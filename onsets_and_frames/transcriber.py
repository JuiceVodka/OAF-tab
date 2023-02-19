"""
A rough translation of Magenta's Onsets and Frames implementation [1].

    [1] https://github.com/tensorflow/magenta/blob/master/magenta/models/onsets_frames_transcription/model.py
"""

import torch
import torch.nn.functional as F
from torch import nn

from .lstm import BiLSTM
from .mel import melspectrogram

class StringSoftmax(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        #frst_dim = (x.shape[0]*x.shape[1]*x.shape[2])//(6*21)
        #print(x.shape)
        x = torch.reshape(x, (x.shape[0] , x.shape[1], 6, 21))#a ta reshape naredi pravilno po vrsticah ali p stolpcih?
        
        """for i in range(6):
            x[:, :, i, :] = torch.softmax(x[:, :, i, :], dim=3)"""
        x = torch.softmax(x, dim=3)
        
        return x
    


class ConvStack(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()

        # input is batch_size * 1 channel * frames * input_features
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(output_features // 16, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(output_features // 16, output_features // 8, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 8),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((output_features // 8) * (input_features // 4), output_features),
            nn.Dropout(0.5)
        )

    def forward(self, mel):
        x = mel.view(mel.size(0), 1, mel.size(1), mel.size(2))
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class OnsetsAndFrames(nn.Module):
    def __init__(self, input_features, output_features, model_complexity=48):
        super().__init__()

        model_size = model_complexity * 16
        sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)

        self.onset_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.offset_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.frame_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.combined_stack = nn.Sequential(
            sequence_model(output_features * 3, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.tab_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, 126),
            #nn.Linear(output_features, 128)
            #nn.Linear(128, 126)
            StringSoftmax()
        )
        self.combined_tab_stack = nn.Sequential(
            sequence_model(output_features * 3, model_size),
            nn.Linear(model_size, 126),
            StringSoftmax()
        )
        """self.velocity_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features)
        )"""

    def forward(self, mel):
        onset_pred = self.onset_stack(mel)
        #print(onset_pred.shape)
        offset_pred = self.offset_stack(mel)
        #print(offset_pred.shape)
        activation_pred = self.frame_stack(mel)
        combined_pred = torch.cat([onset_pred.detach(), offset_pred.detach(), activation_pred], dim=-1)
        frame_pred = self.combined_stack(combined_pred)
        tab_pred = self.tab_stack(mel)
        combined_tab_pred = self.combined_tab_stack(combined_pred)
        #velocity_pred = self.velocity_stack(mel)
        return onset_pred, offset_pred, activation_pred, frame_pred, tab_pred, combined_tab_pred#, velocity_pred

    def run_on_batch(self, batch):
        audio_label = batch['audio']
        onset_label = batch['onset']
        offset_label = batch['offset']
        frame_label = batch['frame']
        tab_label = batch['tab']
        #velocity_label = batch['velocity']

        mel = melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]).transpose(-1, -2)
        onset_pred, offset_pred, _, frame_pred, tab_pred, combined_tab_pred = self(mel)
        
        #tab_pred = tab_pred.reshape(*tab_label.shape)
        #print(onset_label.shape)
        #print(tab_pred.shape)
        #print(tab_label.shape)
        
        #tab_pred = self.softmax_by_string(tab_pred)

        predictions = {
            'onset': onset_pred.reshape(*onset_label.shape),
            'offset': offset_pred.reshape(*offset_label.shape),
            'frame': frame_pred.reshape(*frame_label.shape),
            'tab' : tab_pred.reshape(*tab_label.shape),
            'comb_tab' : combined_tab_pred.reshape(*tab_label.shape),
            'tab_binary' : tab_pred,
            'comb_tab_binary' : combined_tab_pred
        }

        losses = {
            'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
            'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label),
            'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
            'loss/tab' : self.tab_loss(predictions['tab'], tab_label),
            'loss/comb_tab' : self.categorical_cross_entropy_by_string(predictions['comb_tab'], tab_label),
            'loss/tab_binary': F.binary_cross_entropy(predictions['tab_binary'].view(-1, predictions['tab_binary'].size(-1)), tab_label.view(-1, tab_label.size(-1)).float()),
            'loss/comb_tab_binary': F.binary_cross_entropy(predictions['comb_tab_binary'].view(-1, predictions['comb_tab_binary'].size(-1)), tab_label.view(-1, tab_label.size(-1)).float())
        }

        return predictions, losses

    def velocity_loss(self, velocity_pred, velocity_label, onset_label):
        denominator = onset_label.sum()
        if denominator.item() == 0:
            return denominator
        else:
            return (onset_label * (velocity_label - velocity_pred) ** 2).sum() / denominator

    
    def tab_loss(self, tab_pred, tab_label):
        #print(tab_pred.shape)
        #print(tab_label.shape)
        #print(tab_label[:, :, 0, :].argmax(dim=2).shape)
        #loss = 0
        #for i in range(6):
        #    loss += F.cross_entropy(tab_pred[:,:,i,:], tab_label[:,:,i,:].argmax(dim=2), reduction='mean')
        target = tab_label.view(-1, tab_label.size(-1))
        output = tab_pred.view(-1, tab_pred.size(-1))
        loss = F.cross_entropy(output, target.argmax(dim=-1), reduction="mean")
        return loss
    
    
    def categorical_cross_entropy_by_string(self, tab_pred, tab_label):
        #print(tab_pred.shape)
        if(len(tab_pred.shape) == 4):
            batch_size, seq_len, num_strings, num_classes = tab_pred.shape
            
            # Reshape to [batch_size * seq_len * num_strings, num_classes]
            tab_pred_flat = tab_pred.view(-1, num_classes)
            tab_label_flat = tab_label.view(-1, num_classes)
            
            # Compute categorical cross-entropy for each string
            losses = []
            for i in range(num_strings):
                start_idx = i * batch_size * seq_len
                end_idx = (i+1) * batch_size * seq_len
                string_pred = tab_pred_flat[start_idx:end_idx]
                string_label = tab_label_flat[start_idx:end_idx].argmax(dim=1)
                loss = F.cross_entropy(string_pred, string_label, reduction='mean')
                losses.append(loss)
            
            # Return sum of losses for all strings
            return sum(losses)
        else:
            seq_len, num_strings, num_classes = tab_pred.shape
            tab_pred_flat = tab_pred.view(-1, num_classes)
            tab_label_flat = tab_label.view(-1, num_classes)
            losses = []
            for i in range(num_strings):
                start_idx = i*seq_len
                end_idx = (i+1)*seq_len
                string_pred = tab_pred_flat[start_idx:end_idx]
                string_label = tab_label_flat[start_idx:end_idx].argmax(dim=1)
                loss = F.cross_entropy(string_pred, string_label, reduction="mean")
                losses.append(loss)
            return sum(losses)
                
    
    def softmax_by_string(self, x):
        #frst_dim = (x.shape[0]*x.shape[1]*x.shape[2])//(6*21)
        #x = torch.reshape(x, (frst_dim, 6, 21))
        string_sm = []
        for i in range(6):
            string_sm.append(torch.softmax(x[:, i, :], dim=1))
        return torch.cat(string_sm, dim=1)
            

        
