import argparse
import os
import sys
from collections import defaultdict

import numpy as np
from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.transcription_velocity import precision_recall_f1_overlap as evaluate_notes_with_velocity
from mir_eval.util import midi_to_hz
from scipy.stats import hmean
from tqdm import tqdm

import onsets_and_frames.dataset as dataset_module
from onsets_and_frames import *

eps = sys.float_info.epsilon

#tab pitch metrics

def tabToPitch(tab):
    stringsMidi = [40, 45, 50, 55, 59, 64]
    pitchVector = np.zeros(44)
    for string in range(tab.shape[0]):
        fret = tab[string, :]
        played = np.argmax(fret, -1)
        if(played > 0):
            pitch = played + stringsMidi[string] - 41  # to move down by pitch of lower e string + 1 for string not played class
            pitchVector[pitch] = 1
    return pitchVector

def tabToVec(tab):
    fretBoard = np.zeros((6, 20))
    for string in range(tab.shape[0]):
        stringVec = tab[string, :]
        played = np.argmax(stringVec, -1)
        if(played > 0):
            fret = played - 1
            fretBoard[string][fret] = 1
    return fretBoard

def pitchPrecision(prediction, truth):
    pitchPred = np.array(list(map(tabToPitch, prediction)))
    pitchTruth = np.array(list(map(tabToPitch, truth)))
    precision = np.sum(np.multiply(pitchPred, pitchTruth).flatten()) / np.sum(pitchPred.flatten())
    return precision


def pitchRecall(prediction, truth):
    pitchPred = np.array(list(map(tabToPitch, prediction)))
    pitchTruth = np.array(list(map(tabToPitch, truth)))
    recall = np.sum(np.multiply(pitchPred, pitchTruth).flatten()) / np.sum(pitchTruth.flatten())
    return recall

def tabPrecision(prediction, truth):
    tabPred = np.array(list(map(tabToVec, prediction)))
    tabTruth = np.array(list(map(tabToVec, truth)))
    precision = np.sum(np.multiply(tabPred, tabTruth).flatten()) / np.sum(tabPred.flatten())
    return precision


def tabRecall(prediction, truth):
    tabPred = np.array(list(map(tabToVec, prediction)))
    tabTruth = np.array(list(map(tabToVec, truth)))
    recall = np.sum(np.multiply(tabPred, tabTruth).flatten()) / np.sum(tabTruth.flatten())
    return recall



def fMeasure(prediction, truth, isTab):
    precision = tabPrecision(prediction, truth) if isTab else pitchPrecision(prediction, truth)
    recall = tabRecall(prediction, truth) if isTab else pitchRecall(prediction, truth)
    f = (2 * precision * recall) / (precision + recall)
    return f



def evaluate(data, model, onset_threshold=0.5, frame_threshold=0.5, save_path=None):
    metrics = defaultdict(list)

    for label in data:
        #print(label)
        pred, losses = model.run_on_batch(label)

        for key, loss in losses.items():
            metrics[key].append(loss.item())

        for key, value in pred.items():
            value.squeeze_(0).relu_()

        p_ref, i_ref = extract_notes(label['onset'], label['frame'])
        p_est, i_est = extract_notes(pred['onset'], pred['frame'], onset_threshold, frame_threshold)

        t_ref, f_ref = notes_to_frames(p_ref, i_ref, label['frame'].shape)
        t_est, f_est = notes_to_frames(p_est, i_est, pred['frame'].shape)

        scaling = HOP_LENGTH / SAMPLE_RATE

        i_ref = (i_ref * scaling).reshape(-1, 2)
        p_ref = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_ref])
        i_est = (i_est * scaling).reshape(-1, 2)
        p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])

        t_ref = t_ref.astype(np.float64) * scaling
        f_ref = [np.array([midi_to_hz(MIN_MIDI + midi) for midi in freqs]) for freqs in f_ref]
        t_est = t_est.astype(np.float64) * scaling
        f_est = [np.array([midi_to_hz(MIN_MIDI + midi) for midi in freqs]) for freqs in f_est]

        p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est, offset_ratio=None)
        metrics['metric/note/precision'].append(p)
        metrics['metric/note/recall'].append(r)
        metrics['metric/note/f1'].append(f)
        metrics['metric/note/overlap'].append(o)

        p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est)
        metrics['metric/note-with-offsets/precision'].append(p)
        metrics['metric/note-with-offsets/recall'].append(r)
        metrics['metric/note-with-offsets/f1'].append(f)
        metrics['metric/note-with-offsets/overlap'].append(o)

        """p, r, f, o = evaluate_notes_with_velocity(i_ref, p_ref, v_ref, i_est, p_est, v_est,
                                                  offset_ratio=None, velocity_tolerance=0.1)
        metrics['metric/note-with-velocity/precision'].append(p)
        metrics['metric/note-with-velocity/recall'].append(r)
        metrics['metric/note-with-velocity/f1'].append(f)
        metrics['metric/note-with-velocity/overlap'].append(o)

        p, r, f, o = evaluate_notes_with_velocity(i_ref, p_ref, v_ref, i_est, p_est, v_est, velocity_tolerance=0.1)
        metrics['metric/note-with-offsets-and-velocity/precision'].append(p)
        metrics['metric/note-with-offsets-and-velocity/recall'].append(r)
        metrics['metric/note-with-offsets-and-velocity/f1'].append(f)
        metrics['metric/note-with-offsets-and-velocity/overlap'].append(o)"""

        frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)
        metrics['metric/frame/f1'].append(hmean([frame_metrics['Precision'] + eps, frame_metrics['Recall'] + eps]) - eps)

        for key, loss in frame_metrics.items():
            metrics['metric/frame/' + key.lower().replace(' ', '_')].append(loss)

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            label_path = os.path.join(save_path, os.path.basename(label['path']) + '.label.png')
            save_pianoroll(label_path, label['onset'], label['frame'])
            pred_path = os.path.join(save_path, os.path.basename(label['path']) + '.pred.png')
            save_pianoroll(pred_path, pred['onset'], pred['frame'])
            midi_path = os.path.join(save_path, os.path.basename(label['path']) + '.pred.mid')
            #save_midi(midi_path, p_est, i_est, v_est)
            
            
        predTab = pred['tab'].cpu().numpy()
        labelTab = label['tab'].cpu().numpy()
        
        metrics['metric/tab/pitch-precision'].append(pitchPrecision(predTab, labelTab))
        metrics['metric/tab/pitch-recall'].append(pitchRecall(predTab, labelTab))
        metrics['metric/tab/pitch-fScore'].append(fMeasure(predTab, labelTab, False))
        metrics['metric/tab/tab-precision'].append(tabPrecision(predTab, labelTab))
        metrics['metric/tab/tab-recall'].append(tabRecall(predTab, labelTab))
        metrics['metric/tab/tab-fScore'].append(fMeasure(predTab, labelTab, True))
        
        
        predTabComb = pred['comb_tab'].cpu().numpy()
        
        metrics['metric/tab-comb/pitch-precision'].append(pitchPrecision(predTabComb, labelTab))
        metrics['metric/tab-comb/pitch-recall'].append(pitchRecall(predTabComb, labelTab))
        metrics['metric/tab-comb/pitch-fScore'].append(fMeasure(predTabComb, labelTab, False))
        metrics['metric/tab-comn/tab-precision'].append(tabPrecision(predTabComb, labelTab))
        metrics['metric/tab-comb/tab-recall'].append(tabRecall(predTabComb, labelTab))
        metrics['metric/tab-comb/tab-fScore'].append(fMeasure(predTabComb, labelTab, True))
        
        """print(f"Tab pitch precission: {pitchPrecision(predTab, labelTab)}")
        print(f"Tab pitch recall: {pitchRecall(predTab, labelTab)}")
        print(f"Tab pitch f-score: {fMeasure(predTab, labelTab, False)}")
        
        print(f"Tab precision: {tabPrecision(predTab, labelTab)}")
        print(f"Tab recall: {tabRecall(predTab, labelTab)}")
        print(f"Tab f-score: {fMeasure(predTab, labelTab, True)}")"""
        

    return metrics


def evaluate_file(model_file, dataset, dataset_group, sequence_length, save_path,
                  onset_threshold, frame_threshold, device):
    dataset_class = getattr(dataset_module, dataset)
    kwargs = {'sequence_length': sequence_length, 'device': device}
    if dataset_group is not None:
        kwargs['groups'] = [dataset_group]
    dataset = dataset_class(**kwargs)

    model = torch.load(model_file, map_location=device).eval()
    summary(model)

    metrics = evaluate(tqdm(dataset), model, onset_threshold, frame_threshold, save_path)

    for key, values in metrics.items():
        if key.startswith('metric/'):
            _, category, name = key.split('/')
            print(f'{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=str)
    parser.add_argument('dataset', nargs='?', default='GuitarSet')
    parser.add_argument('dataset_group', nargs='?', default=None)
    parser.add_argument('--save-path', default=None)
    parser.add_argument('--sequence-length', default=None, type=int)
    parser.add_argument('--onset-threshold', default=0.5, type=float)
    parser.add_argument('--frame-threshold', default=0.5, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        evaluate_file(**vars(parser.parse_args()))
