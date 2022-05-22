import os
import scipy.io.wavfile
import six
import numpy as np
from torch.utils.data import Dataset
from configs import *
import copy
import torch
import wave
import contextlib
from torch.nn.utils.rnn import pad_sequence
np.random.seed(997)

def pad_collate(batch):
  (enc_inputs, dec_inputs, dec_outputs) = zip(*batch)
  enc_input_lens = [(len(enc_input)-FFT_SIZE)//HOP_WIDTH+1 for enc_input in enc_inputs]

  enc_inputs_pad = pad_sequence(enc_inputs, batch_first=True, padding_value=0)
  dec_inputs_pad = pad_sequence(dec_inputs, batch_first=True, padding_value=PAD_IDX)
  dec_outputs_pad = pad_sequence(dec_outputs, batch_first=True, padding_value=PAD_IDX)

  return enc_inputs_pad, dec_inputs_pad, dec_outputs_pad, enc_input_lens

class MaestroDataset(Dataset):
    def __init__(self, config) -> None:
        self.train_dir = config.train_dir
        self.data_paths = self.get_filepairs(self.train_dir)
        self.data_num = len(self.data_paths)
    
    def get_filepairs(self, data_dir):
        wav_paths = []
        for filename in os.listdir(data_dir):
            if filename.endswith('.wav'):
                wav_path = os.path.join(data_dir, filename)
                with contextlib.closing(wave.open(wav_path, 'r')) as f:
                    frames = f.getnframes()
                if frames>=HOP_WIDTH*INPUT_LENGTH:
                    wav_paths.append(wav_path)
        label_paths = [wav_path.replace('.wav', '.txt') for wav_path in wav_paths]
        return list(zip(wav_paths, label_paths))

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        wav_path, label_path = self.data_paths[idx]
        wav_data = open(wav_path, 'rb').read()
        _, audio = scipy.io.wavfile.read(six.BytesIO(wav_data))
        label = np.loadtxt(label_path, dtype=np.float32).reshape([-1, 4])[:, :3]

        # padded the audio
        n_segments = int(np.ceil(audio.shape[0]/HOP_WIDTH))
        pad_audio = (n_segments-1)*HOP_WIDTH+FFT_SIZE - audio.shape[0]
        audio = np.pad(audio, ((0, pad_audio), ), mode='constant')

        # padding the data and label
        # start_segment = np.random.randint(0, n_segments)
        start_segment = np.random.randint(0, n_segments-INPUT_LENGTH+1)
        # segment_length = np.random.randint(1, min(n_segments-start_segment+1, INPUT_LENGTH+1))
        # segment_length = min(n_segments-start_segment, INPUT_LENGTH)
        segment_length = INPUT_LENGTH
        start_pos = start_segment*HOP_WIDTH
        end_pos = start_pos + (segment_length-1)*HOP_WIDTH+FFT_SIZE-1
        enc_inputs = audio[start_pos: end_pos+1]

        start_t = start_pos/SAMPLING_RATE
        end_t = start_t + segment_length*HOP_WIDTH/SAMPLING_RATE
        dec_inputs, dec_outputs = self._get_segment_ns(label, start_t, end_t)

        return torch.from_numpy(enc_inputs), torch.from_numpy(dec_inputs), \
                torch.from_numpy(dec_outputs) # variable length

    def _get_segment_ns(self, label, start_t, end_t):
        # onset notes, offset notes
        # dict: {'time_idx': [[note1, note2, ...], [note1, note2, ...]]} 
        onoff_events = dict() 
        tie_events = []
        # 量化onsets和offsets
        for onset, offset, pitch in label:
            pitch_idx = map_event_to_idx(NOTE_EVENT, int(pitch))
            onset_event_idx = map_event_to_idx(TIME_EVENT, onset-start_t)
            offset_event_idx = map_event_to_idx(TIME_EVENT, offset-start_t)
            if onset>=start_t and onset<=end_t:
                if onset_event_idx in onoff_events.keys():
                    onoff_events[onset_event_idx][0].append(pitch_idx)
                else:
                    onoff_events[onset_event_idx] = [[pitch_idx], []]
            if offset>=start_t and offset<=end_t:
                if offset_event_idx in onoff_events.keys():
                    onoff_events[offset_event_idx][1].append(pitch_idx)
                else:
                    onoff_events[offset_event_idx] = [[], [pitch_idx]]
            if onset<start_t and offset>start_t:
                tie_events.append(pitch_idx)
        
        onoff_keys = sorted(onoff_events.keys())
        tie_events.sort()

        begin_time_idx = map_event_to_idx(TIME_EVENT, 0)
        events = []
        time_type = 'initial'
        # time_idx, type_idx, pitch_idx
        if len(tie_events)>0:
            events.append(begin_time_idx)
            events.append(ETS_IDX)
            for pitch_idx in sorted(tie_events):
                events.append(pitch_idx)
        
            if len(onoff_keys)>0 and onoff_keys[0]==begin_time_idx:
                onsets, offsets = onoff_events[begin_time_idx]
                onsets.sort()
                offsets.sort()
                if len(offsets)>0:
                    events.append(OFFSET_IDX)
                    time_type = 'off'
                    for pitch_idx in offsets:
                        events.append(pitch_idx)
                if len(onsets)>0:
                    events.append(ONSET_IDX)
                    time_type = 'on'
                    for pitch_idx in onsets:
                        events.append(pitch_idx)
                
                del onoff_keys[0]

        for time_idx in onoff_keys:
            events.append(time_idx)
            onsets, offsets = onoff_events[time_idx]
            onsets.sort()
            offsets.sort()
            if len(offsets)>0:
                if time_type!='off':
                    events.append(OFFSET_IDX)
                    time_type='off'
                for pitch_idx in offsets:
                    events.append(pitch_idx)
            if len(onsets)>0:
                if time_type!='on':
                    events.append(ONSET_IDX)
                    time_type='on'
                for pitch_idx in onsets:
                    events.append(pitch_idx)

        dec_outputs = copy.copy(events)
        dec_inputs = copy.copy(events)
        dec_outputs.append(EOS_IDX)
        dec_inputs.insert(0, SOS_IDX)

        return np.array(dec_inputs, dtype=np.int16), np.array(dec_outputs, dtype=np.int16)