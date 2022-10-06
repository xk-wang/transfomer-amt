
import numpy as np

class VocabIndexTransform:

  def __init__(self, time_interval) -> None:
    super(VocabIndexTransform, self).__init__()
    self.vocabs2index_dict = self.vocab2index()
    self.vocabs_len = len(self.vocabs2index_dict)
    self.index2vocabs_dict = self.index2vocab()
    self.time_interval = time_interval

  def noteseq2indexs(self, noteseq, start_t, end_t):
    '''
    inputs:
      noteseq (onset, offset, pitch, velocity)
      start_t begin time
      end_t end time
    outputs:
      note event index sequence 1 for decoder inputs
      note event index sequence 2 for decoder outputs
    '''
    if isinstance(noteseq, np.ndarray):
      noteseq = noteseq.reshape(-1, 4)
    
    events = []
    for onset, offset, pitch, velocity in noteseq:
      if onset>=start_t and offset<=end_t:
        events.append([onset, velocity, pitch])
      if offset>=start_t and offset<=end_t:
        events.append([offset, 0, pitch])
    events.sort(key=lambda x: x[0])

    indexs = []
    for t, v, p in events:
      offset_t = t - start_t
      t_index = int(np.round(offset_t/self.time_interval))
      indexs.append(self.vocabs2index_dict['time ' + str(t_index)])
      indexs.append(self.vocabs2index_dict['velocity ' + str(v)])
      indexs.append(self.vocabs2index_dict['pitch ' + str(p)])
    indexs1 = [self.vocabs2index_dict['bos']] + indexs
    indexs2 = indexs + [self.vocabs2index_dict['eos']]
    return indexs1, indexs2
      
  def indexs2noteseq(self, indexs):
    '''
    inputs:
      note event index sequence
    outputs:
      noteseq (onset, offset, pitch, velocity)
    '''

    # some time events will be dropped!
    # if the output sentence is perfect then will no problem
    sentence = self.index2sentence(indexs)
    onset_events = []
    offset_events = []
    for idx, word in enumerate(sentence):
      if word.startswith('velocity'):
        t = int(sentence[idx-1])*self.time_interval
        pitch = int(sentence[idx+1])
        # offset events
        if word.split('velocity ')[1] == '0':
          offset_events.append([t, pitch])
        # onset events
        else:
          onset_events.append([t, pitch, int(word.split('velocity ')[1])])
    
    onset_events.sort(key=lambda x: x[0])
    offset_events.sort(key=lambda x: x[0])
    note_events = []
    visit = [False for i in range(len(offset_events))]
    # merge notes
    for i in range(len(onset_events)):
      for j in range(len(offset_events)):
        if not visit[j] and offset_events[j][1] == onset_events[i][1] and \
          offset_events[j][0]>onset_events[i][0]:
          visit[j] = True
          note_events.append([onset_events[i][0], offset_events[j][0]] + onset_events[1:])
    return note_events


  def vocab2index(self):
    notes = {
      'pitch ' + str(i+21): i for i in range(128)
    }
    total = len(notes)

    velocities = {
      'velocity ' + str(i): i+total for i in range(128)
    }
    total += len(velocities)

    times = {
      'time ' + str(i): i+total for i in range(6000)
    }
    # times = {
    #   'time ' + str(i): i+total for i in range(410)
    # }
    total += len(times)

    other_symbols = {
      'bos': total,
      'eos': total+1
    }
    total += len(other_symbols)

    vocabs2index_dict = {**notes, **velocities, **times, **other_symbols}

    assert max(vocabs2index_dict.values())==total-1, (max(vocabs2index_dict.values()), total)

    return vocabs2index_dict

  def index2vocab(self):

    index2vocabs_dict = dict()
    for key, value in self.vocabs2index_dict.items():
      index2vocabs_dict[value] = key
    
    return index2vocabs_dict

  def sentence2index(self, strs):
    return [self.vocabs2index_dict[st] for st in strs]
  
  def index2sentence(self, indexs):
    return [self.index2vocabs_dict[i] for i in indexs]


def construct_global(time_interval):

  vocabs = VocabIndexTransform(time_interval).vocabs2index_dict
  vocabs_len = len(vocabs)
  eos_index = vocabs['eos']

  return vocabs, vocabs_len, eos_index