import numpy as np
import argparse
from multiprocess import Pool
import os

interval = 0.01
window_len = 407 # for the input audio of 4.096s

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('label_path', type=str, help='path to save the split labels')
  return parser.parse_args()


def count_max_num(labelpath):
  label = np.loadtxt(labelpath, dtype=np.float32).reshape([-1, 4])[:, :2]
  max_time = label[:, 1].max()
  max_frame = int(np.ceil(max_time/interval)+1)
  piano_roll = np.zeros((max_frame, 1), dtype=np.int32)

  for idx, (onset, offset) in enumerate(label):
    start_frame = int(onset/interval+0.5)
    end_frame = int(offset/interval+0.5)    
    
    piano_roll[start_frame] += 1 # time event and pitch event
    try:
      piano_roll[end_frame] += 1
    except:
      raise ValueError(labelpath, label.shape, label[label.shape[0]-1], max_time, offset, end_frame, max_frame)


  max_count = initial_count = sum(piano_roll[: window_len])

  for i in range(1, max_frame-window_len+1):
    initial_count = initial_count-piano_roll[i-1]+piano_roll[i+window_len-1]
    max_count = max(max_count, initial_count)
  
  print('%-4d'%max_count, labelpath)
  
  return max_count


def count_global_max(label_dir):
  filepaths = []
  for filename in os.listdir(label_dir):
    if filename.endswith('.txt'):
      filepaths.append(os.path.join(label_dir, filename))

  # filepaths = ['/home/data/wxk/master-graduate/maestro-v1.0.0-spits/train/MIDI-Unprocessed_18_R1_2006_01-05_ORIG_MID--AUDIO_18_R1_2006_04_Track04_wav_022.txt']
  
  # for filepath in filepaths:
  #   count_max_num(filepath)
  
  pool = Pool(16)
  res = pool.map_async(count_max_num, filepaths)
  counts = res.get()
  pool.close()
  pool.join()
  print('global max_count', max(counts)) # 469 for maestro training set 425 for maestro test dataset

if __name__ == '__main__':
  args = parse_args()
  count_global_max(args.label_path)