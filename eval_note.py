import argparse
import os
import librosa
import numpy as np
import mir_eval

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_path', type=str)
    parser.add_argument('--result_path', type=str)
    parser.add_argument('--onset-win', type=float, default=0.05)
    parser.add_argument('--offset-win', type=float, default=0.05)
    parser.add_argument('--offset', action='store_true', default=False)
    return parser.parse_args()

def get_notes(filepath):
    print(filepath)
    notes = []
    with open(filepath) as f:
        for idx, line in enumerate(f):
            onset, offset, pitch = line.strip().split()[:3]
            onset = float(onset)
            offset = float(offset)
            pitch = float(pitch)
            if onset>=offset:
              print(idx+1)
            notes.append([onset, offset, pitch])
    notes.sort(key=lambda x:x[0])
    return np.array(notes)

if __name__=='__main__':
    args = parse_args()
    label_path = args.label_path
    result_path = args.result_path
    onset_win = args.onset_win
    offset_win = args.offset_win
    offset = args.offset
    filenames=sorted([filename for filename in os.listdir(result_path) if filename.endswith(".txt")])

    op = [[], [], []]
    oop = [[], [], []]
    for filename in filenames:
        labelpath = os.path.join(label_path, filename)
        resultpath = os.path.join(result_path, filename)
        label = get_notes(labelpath)
        result = get_notes(resultpath)
        p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
            label[:, :2], librosa.midi_to_hz(label[:, 2]), result[:, :2], librosa.midi_to_hz(result[:, 2]),
            onset_tolerance = onset_win, offset_ratio =None
        )

        op[0].append(f)
        op[1].append(p)
        op[2].append(r)
        print(filename)
        print('op F1: %4.2f%% P: %4.2f%% R:  %4.2f%%'%(f*100, p*100, r*100))

        p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
            label[:, :2], librosa.midi_to_hz(label[:, 2]), result[:, :2], librosa.midi_to_hz(result[:, 2]),
            onset_tolerance=onset_win, offset_min_tolerance = offset_win
        )
        oop[0].append(f)
        oop[1].append(p)
        oop[2].append(r)
        if offset:
            print('opo F1: %4.2f%% P: %4.2f%% R:  %4.2f%%'%(f*100, p*100, r*100))

    mean_op = [round(np.mean(x)*100, 2) for x in op]
    mean_oop = [round(np.mean(x)*100, 2) for x in oop]
    print('\nmean op F1: %4.2f%% P: %4.2f%% R:  %4.2f%%'%tuple(mean_op))
    if offset:
        print('mean opo F1: %4.2f%% P: %4.2f%% R:  %4.2f%%'%tuple(mean_oop))