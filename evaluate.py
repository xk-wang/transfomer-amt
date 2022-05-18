import torch
import configs
eps = 1e-7

def compute_metrics(logits, labels):
  mask = (labels==configs.PAD_IDX).type(torch.LongTensor).to(logits.device)
  pred = torch.argmax(logits, dim=-1)
  comp = (labels == pred).type(torch.LongTensor).to(logits.device)
  TP = torch.sum(comp * mask)
  TOTAL = torch.sum(mask)
  acc = TP / ( TOTAL + eps)
  return {'acc': acc}
