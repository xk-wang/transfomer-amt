import os
import pandas as pd
import librosa
import numpy as np
import argparse
from multiprocessing import Pool
from functools import partial
from transformer import Transformer
import soundfile as snd
from configs import *
import scipy.io.wavfile
import six
import torch


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_path', type=str,
                        help='the path to checkpoint')
    parser.add_argument('data_root', type=str,
                        help='the root path to maestrov1')
    parser.add_argument('dest_root', type=str,
                        help='the root path to save resample audio')
    parser.add_argument('res_root', type=str,
                        help='the root path to save res')

    return parser.parse_args()

def process_one_file(wav_path, dest_root):
    basename = os.path.basename(wav_path)
    save_path = os.path.join(dest_root, basename)
    audio, _ = librosa.load(wav_path, sr=SAMPLING_RATE, mono=True)
    snd.write(save_path, audio, SAMPLING_RATE)
    return save_path

def resample_test_audio(data_root, save_root):

    data_info = pd.read_csv(os.path.join(data_root, 'maestro-v1.0.0.csv'))
    select_info = data_info.loc[data_info['split'] == 'test']
    wav_names = list(select_info['audio_filename'])
    wav_paths = [os.path.join(data_root, wav_name) for wav_name in wav_names]

    pool = Pool(16)
    func = partial(process_one_file, dest_root=save_root)
    # func(wav_paths[0])
    res = pool.map_async(func, wav_paths)
    # print(res.get())
    pool.close()
    pool.join()

class InferModel:
  def __init__(self, ckpt_path, device='cuda') -> None:
      self.model = Transformer().eval().to(device)
      self.model.load_state_dict(torch.load(ckpt_path))
      self.device = device
    # .to(device)
    #   self.model.eval()

  def predict(self, wav_path):
    print(wav_path)
    wav_data = open(wav_path, 'rb').read()
    _, audio = scipy.io.wavfile.read(six.BytesIO(wav_data))
    audio_length = (INPUT_LENGTH-1)*HOP_WIDTH+FFT_SIZE
    begin = 0
    while begin<audio.shape[0]:
        end = begin+audio_length
        chunk = audio[begin:end]
        input_mask = int((chunk.shape[0]-FFT_SIZE)/HOP_WIDTH+1.5) if chunk.shape[0]>=FFT_SIZE else 1
        input_mask = np.ones((input_mask, ), np.int8)
        if chunk.shape[0]<audio_length:
            chunk = np.pad(chunk, (0, audio_length - audio.shape[0]), mode='constant')
            input_mask = np.pad(input_mask, (0, INPUT_LENGTH - input_mask.shape[0]), mode='constant')

        chunk = torch.from_numpy(chunk.reshape([1, -1])).to(self.device)
        input_mask = torch.from_numpy(input_mask.reshape([1, -1])).to(self.device)
        enc_inputs = self.model.Spec(chunk)

        enc_outputs, _ = self.model.Encoder(enc_inputs, input_mask)
        dec_inputs = PAD_IDX*torch.ones((1, OUTPUT_LENGTH), dtype=torch.int32).to(self.device)
        next_symbol = SOS_IDX
        for i in range(OUTPUT_LENGTH):
            dec_inputs[0, i] = next_symbol
            dec_outputs, _, _ = self.model.Decoder(dec_inputs, input_mask, enc_outputs)
            projected = self.model.projection(dec_outputs)
            next_symbol = projected.squeeze(0).max(dim=-1, keepdim=False)[1].data[i]
            # raise ValueError(next_symbol)
            print(round(map_idx_to_event(next_symbol)[1], 2), end=' ')
            if next_symbol==EOS_IDX:
                break
        break
    
    
if __name__ == '__main__':
    args = parse_args()
    # resample_test_audio(args.data_root, args.dest_root)
    wav_paths = [os.path.join(args.dest_root, filename) for filename in os.listdir(args.dest_root)]
    wav_paths.sort(key=lambda x: x.lower())

    infer_model = InferModel(args.ckpt_path)
    
    for wav_path in wav_paths[:1]:
        infer_model.predict(wav_path)