import os
import pandas as pd
import librosa
import numpy as np
import argparse
from multiprocessing import Pool
from functools import partial
import soundfile as snd
from note_seq import midi_io
from note_seq import sequences_lib
from configs import *


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('data_root', type=str,
                        help='the root path to maestrov1')
    parser.add_argument('dest_root', type=str,
                        help='the root path to the split dataset')

    return parser.parse_args()


def label_process_one_file(midi_path):
    ns = midi_io.midi_file_to_note_sequence(midi_path)
    sequence = sequences_lib.apply_sustain_control_changes(ns)
    pedal_extend_path = midi_path.replace('.midi', '_extend.txt')

    notes = []
    for note in sequence.notes:
        onset, offset, pitch, velocity = note.start_time, note.end_time, \
            note.pitch, note.velocity
        notes.append([onset, offset, pitch, velocity])
    notes.sort(key=lambda note: note[0])

    with open(pedal_extend_path, 'wt', encoding='utf8') as f:
        for onset, offset, pitch, velocity in notes:
            f.write("%-8.4f  %-8.4f  %-3d  %-3d\n" %
                    (onset, offset, pitch, velocity))
    
    print('process label', pedal_extend_path)


def generate_magenta_label(data_root):

    csv_path = os.path.join(data_root, 'maestro-v1.0.0.csv')
    data_info = pd.read_csv(csv_path)
    wav_names = list(data_info['audio_filename'])
    wav_paths = [os.path.join(data_root, wav_name) for wav_name in wav_names]
    midi_paths = [filepath.replace('.wav', '.midi') for filepath in wav_paths]

    pool = Pool(16)
    res = pool.map_async(label_process_one_file, midi_paths)
    print(res.get())
    pool.close()
    pool.join()


def split_process_one_file(inputs, save_root):
    wav_path, label_path = inputs
    audio, _ = librosa.load(wav_path, mono=True, sr=SAMPLING_RATE)
    ns = np.loadtxt(label_path, dtype=np.float32).reshape([-1, 4])

    def find_splits(ns):
        splits = [0]
        note_seqs = []
        temp_notes = [ns[0]]
        for i in range(1, len(ns)):
            mid_time = (ns[i-1, 1]+ns[i, 0])/2
            if ns[i, 0] >= ns[i-1, 1] and mid_time-splits[-1] >= SPLIT_AUDIO_MIN_LENGTH:
                splits.append(mid_time)
                note_seqs.append(np.stack(temp_notes, axis=0))
                temp_notes = [ns[i]]
            else:
              temp_notes.append(ns[i])

        splits.append(-1)
        if temp_notes:
            note_seqs.append(np.stack(temp_notes, axis=0))
        return splits, note_seqs

    splits, note_seqs = find_splits(ns)
    n_segments = len(splits)-1

    basename = os.path.basename(wav_path).replace('.wav', '')
    for seg in range(n_segments):
        start, end = splits[seg], splits[seg+1]
        start_frame = int(start*SAMPLING_RATE+0.5)
        end_frame = int(end*SAMPLING_RATE + 0.5) if end > 0 else audio.shape[0]
        audio_split = audio[start_frame:end_frame]
        note_seqs[seg][:, :2] = note_seqs[seg][:, :2]-start
        label_split = note_seqs[seg]

        audio_split_path = os.path.join(
            save_root, basename+'_%03d.wav' % seg)
        label_split_path = os.path.join(
            save_root, basename+'_%03d.txt' % seg)

        snd.write(audio_split_path, audio_split, SAMPLING_RATE)
        with open(label_split_path, 'wt', encoding='utf8') as f:
            for onset, offset, pitch, velocity in label_split:
                f.write("%-8.4f  %-8.4f  %-3d  %-3d\n" %
                        (onset, offset, pitch, velocity))

    print('process split', wav_path)


def generate_data_splits(data_root, data_type, save_root):
    data_info = pd.read_csv(os.path.join(data_root, 'maestro-v1.0.0.csv'))
    select_info = data_info.loc[data_info['split'] == data_type]
    wav_names = list(select_info['audio_filename'])
    wav_paths = [os.path.join(data_root, wav_name) for wav_name in wav_names]
    label_paths = [filepath.replace('.wav', '_extend.txt') for filepath in wav_paths]

    pool = Pool(16)
    func = partial(split_process_one_file, save_root=save_root)
    inputs = list(zip(wav_paths, label_paths))
    # func(inputs[0])
    res = pool.map_async(func, inputs)
    print(res.get())
    pool.close()
    pool.join()


if __name__ == '__main__':
    args = parse_args()
    generate_magenta_label(args.data_root)
    generate_data_splits(args.data_root, 'test', args.dest_root)
