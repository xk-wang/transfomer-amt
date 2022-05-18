SPLIT_AUDIO_MIN_LENGTH = 30
TIME_INTERVAL = 0.01
SAMPLING_RATE = 16000
FFT_SIZE = 2048
HOP_WIDTH = 128
MEL_LO_HZ = 20.0
MEL_HI_HZ = 8000.0
INPUT_LENGTH = 512
OUTPUT_LENGTH = 1024
TOTAL_EVENT_NUM = 512
EOS_EVENT_NUM = 1
NOTE_EVENT_NUM = 88
NUM_MEL_BINS = 512
INT16_MAX = 32767
TIME_EVENT_NUM = TOTAL_EVENT_NUM-EOS_EVENT_NUM-NOTE_EVENT_NUM # the max time can only reach  410

SOS_EVENT = 0
EOS_EVENT = 1
NOTE_EVENT = 2
TIME_EVENT = 3
PAD_EVENT = 4

d_model = 512   # 字 Embedding 的维度
d_ff = 1024     # 前向传播隐藏层维度
d_k = d_v = 64  # K(=Q), V的维度
n_layers = 6    # 有多少个encoder和decoder
n_heads = 8     # Multi-Head Attention设置为8


def map_event_to_idx(event_type, event=None): # note event in [21, 108], time event [0, 4.088]
  if event_type==SOS_EVENT:
    return 0
  if event_type==EOS_EVENT:
    return 1
  if event_type==NOTE_EVENT:
    return event-19 # 2-89
  if event_type==TIME_EVENT: # 90-499
    return int(event/TIME_INTERVAL+0.5)+90
  if event_type==PAD_EVENT:
    return 500
  raise ValueError('not implement event type!')

def map_idx_to_event(idx):
  idx = int(idx)
  if idx==0:
    return [SOS_EVENT, None]
  if idx==1:
    return [EOS_EVENT, None]
  if idx>=2 and idx<=89:
    return [NOTE_EVENT, idx+19]
  if idx>=90 and idx<=499:
    return [TIME_EVENT, (idx-90)*TIME_INTERVAL]
  if idx==500:
    return [PAD_EVENT, None]
  raise ValueError('idx out of event range!')

PAD_IDX = map_event_to_idx(PAD_EVENT)
SOS_IDX = map_event_to_idx(SOS_EVENT)
EOS_IDX = map_event_to_idx(EOS_EVENT)