SPLIT_AUDIO_MIN_LENGTH = 30
TIME_INTERVAL = 0.01
SAMPLING_RATE = 16000
FFT_SIZE = 2048
HOP_WIDTH = 128
MEL_LO_HZ = 20.0
MEL_HI_HZ = 8000.0
INPUT_LENGTH = 512
OUTPUT_LENGTH = 512 # varaible length

NUM_MEL_BINS = 512
INT16_MAX = 32767

INITIAL_EVENT = 'initial'
EOS_EVENT = 'eos'
ETS_EVENT = 'ets'
ONSET_EVENT = 'on'
OFFSET_EVENT = 'off'
NOTE_EVENT = 'note'
TIME_EVENT = 'time'
SOS_EVENT = 'sos' # not int the output type
PAD_EVENT = 'pad' # not int the output type

EOS_EVENT_NUM = 1
ETS_EVENT_NUM = 1
ONSET_EVENT_NUM = 1
OFFSET_EVENT_NUM = 1
NOTE_EVENT_NUM = 88
TIME_EVENT_NUM = 410
PAD_EVENT_NUM = 1  # must need
# SOS_EVENT_NUM = 1 # only occurs in dec_inputs
# 503
TOTAL_EVENT_NUM = EOS_EVENT_NUM+ETS_EVENT_NUM+ONSET_EVENT_NUM+OFFSET_EVENT_NUM+ \
                  NOTE_EVENT_NUM+TIME_EVENT_NUM+PAD_EVENT_NUM


d_model = 512   # 字 Embedding 的维度
d_ff = 1024     # 前向传播隐藏层维度
d_k = d_v = 64  # K(=Q), V的维度
n_layers = 6    # 有多少个encoder和decoder
n_heads = 8     # Multi-Head Attention设置为8


def map_event_to_idx(event_type, event=None): # note event in [21, 108], time event [0, 4.088]
  if event_type==EOS_EVENT:
    return 0
  if event_type==ETS_EVENT:
    return 1
  if event_type==ONSET_EVENT:
    return 2
  if event_type==OFFSET_EVENT:
    return 3
  if event_type==NOTE_EVENT:
    return event-17 # 4-91
  if event_type==TIME_EVENT: # 92-501
    return int(event/TIME_INTERVAL+0.5)+92
  if event_type==SOS_EVENT:
    return 502
  if event_type==PAD_EVENT:
    return 503
  raise ValueError('not implement event type!')

def map_idx_to_event(idx):
  if idx==0:
    return [EOS_EVENT, None]
  if idx==1:
    return [ETS_EVENT, None]
  if idx==2:
    return [ONSET_EVENT, None]
  if idx==3:
    return [OFFSET_EVENT, None]
  if idx>=4 and idx<=91:
    return [NOTE_EVENT, idx+17]
  if idx>=92 and idx<=501:
    return [TIME_EVENT, (idx-92)*TIME_INTERVAL]
  if idx==502:
    return [SOS_EVENT, None]
  if idx==503:
    return [PAD_EVENT, None]
  raise ValueError('idx out of event range!')

SOS_IDX = map_event_to_idx(SOS_EVENT)
EOS_IDX = map_event_to_idx(EOS_EVENT)
ETS_IDX = map_event_to_idx(ETS_EVENT)
PAD_IDX = map_event_to_idx(PAD_EVENT)
ONSET_IDX = map_event_to_idx(ONSET_EVENT)
OFFSET_IDX = map_event_to_idx(OFFSET_EVENT)