import os
import scipy.io.wavfile
import six
import numpy as np
from torch.utils.data import Dataset
from configs import *

np.random.seed(997)

class MaestroDataset(Dataset):
    def __init__(self, config) -> None:
        self.train_dir = config.train_dir
        self.data_paths = self.get_filepairs(self.train_dir)
        self.data_num = len(self.data_paths)
    
    def get_filepairs(self, data_dir):
        wav_paths = []
        for filename in os.listdir(data_dir):
            if filename.endswith('.wav'):
                wav_paths.append(os.path.join(data_dir, filename))
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
        # print(n_segments, INPUT_LENGTH)
        start_segment = np.random.randint(0, n_segments-INPUT_LENGTH+1)
        # segment_length = np.random.randint(1, min(n_segments-start_segment+1, INPUT_LENGTH+1))
        # segment_length = min(n_segments-start_segment, INPUT_LENGTH)
        segment_length = INPUT_LENGTH
        start_pos = start_segment*HOP_WIDTH
        end_pos = start_pos + (segment_length-1)*HOP_WIDTH+FFT_SIZE-1
        enc_inputs = audio[start_pos: end_pos+1]
        enc_input_mask = np.ones([segment_length, ], dtype=np.int8)

        start_t = start_pos/SAMPLING_RATE
        end_t = start_t + segment_length*HOP_WIDTH/SAMPLING_RATE
        dec_inputs, dec_outputs = self._get_segment_ns(label, start_t, end_t)

        padded_enc_inputs, padded_enc_input_mask, padded_dec_inputs, padded_dec_outputs = \
                self._pad_segment(enc_inputs, enc_input_mask, segment_length, dec_inputs, dec_outputs)

        return padded_enc_inputs, padded_enc_input_mask, padded_dec_inputs, padded_dec_outputs # [HOP_WIDTH*INPUT_LENGTH, ] [INPUT_LENGTH, ] [OUTPUT_LENGTH, ] [OUTPUT_LENGTH, ]

    def _get_segment_ns(self, label, start_t, end_t):
        events = []
        for onset, offset, pitch in label:
            if onset>=start_t and onset<=end_t:
                events.append([onset-start_t, pitch])
            if offset>=start_t and offset<=end_t:
                events.append([offset-start_t, pitch])
        events.sort(key=lambda x: x[0])
        dec_outputs = np.zeros((len(events)*2+1, ), dtype=np.int16)
        dec_inputs = np.zeros((len(events)*2+1, ), dtype=np.int16)
        dec_inputs[0] = map_event_to_idx(SOS_EVENT)
        for i in range(len(events)):
            dec_outputs[2*i] = map_event_to_idx(TIME_EVENT, events[i][0])
            dec_outputs[2*i+1] = map_event_to_idx(NOTE_EVENT, events[i][1])
            dec_inputs[2*i+1] = map_event_to_idx(TIME_EVENT, events[i][0])
            dec_inputs[2*i+2] = map_event_to_idx(NOTE_EVENT, events[i][1])
        dec_outputs[len(events)*2] = map_event_to_idx(EOS_EVENT)
        return dec_inputs, dec_outputs

    def _pad_segment(self, enc_inputs, enc_input_mask, segment_length, dec_inputs, dec_outputs): # [length1, ], [length2, ]

        padded_enc_length = HOP_WIDTH*(INPUT_LENGTH-1)+FFT_SIZE - enc_inputs.shape[0]
        padded_enc_inputs = np.pad(enc_inputs, ((0, padded_enc_length), ), mode='constant')
        padded_enc_mask_length = INPUT_LENGTH - segment_length
        padded_enc_input_mask = np.pad(enc_input_mask, ((0, padded_enc_mask_length), ), mode='constant')
        padded_dec_length = OUTPUT_LENGTH - dec_inputs.shape[0]
        # print(padded_dec_length, OUTPUT_LENGTH, dec_inputs.shape[0])
        padded_dec_inputs = np.pad(dec_inputs, ((0, padded_dec_length), ), mode='constant', constant_values=PAD_IDX)
        padded_dec_outputs = np.pad(dec_outputs, ((0, padded_dec_length), ), mode='constant', constant_values=PAD_IDX)

        return padded_enc_inputs, padded_enc_input_mask, padded_dec_inputs, padded_dec_outputs