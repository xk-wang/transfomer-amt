import numpy as np
import torch.nn as nn
import torch
import einops
from spafe.fbanks import mel_fbanks
from util import construct_global

# vocabs = None
# vocabs_len = None
# eos_index = None

class STFT(nn.Module):
    def __init__(self, win_len, hop_len, fft_len, win_type):
        super(STFT, self).__init__()
        self.win, self.hop = win_len, hop_len
        self.nfft = fft_len
        window = {
            "hann": torch.hann_window(win_len),
            "hamm": torch.hamming_window(win_len),
        }
        assert win_type in window.keys()
        self.window = window[win_type]

    def transform(self, inp):
        """
        inp: B N
        """
        cspec = torch.stft(inp, self.nfft, self.hop, self.win, 
                            self.window.to(inp.device), return_complex=False)
        cspec = einops.rearrange(cspec, "b f t c -> b c f t")
        return cspec

    def inverse(self, real, imag):
        """
        real, imag: B F T
        """
        inp = torch.stack([real, imag], dim=-1)
        return torch.istft(inp, self.nfft, self.hop, self.win, self.window.to(real.device))


class Banks(nn.Module):
    def __init__(self, nfilters, nfft, fs, low_freq=None, high_freq=None, learnable=False):
        super(Banks, self).__init__()
        self.nfilters, self.nfft, self.fs = nfilters, nfft, fs
        filter, _ = mel_fbanks.mel_filter_banks(
            nfilts=self.nfilters,
            nfft=self.nfft,
            low_freq=low_freq,
            high_freq=high_freq,
            fs=self.fs,
        )
        filter = torch.from_numpy(filter).float()
        if not learnable:
            #  30% energy compensation.
            self.register_buffer('filter', filter*1.3)
            self.register_buffer('filter_inv', torch.pinverse(filter))
        else:
            self.filter = nn.Parameter(filter)
            self.filter_inv = nn.Parameter(torch.pinverse(filter))

    def amp2bank(self, amp):
        amp_feature = torch.einsum("bcft,kf->bckt", amp, self.filter.to(amp.device))
        return amp_feature

    def bank2amp(self, inputs):
        return torch.einsum("bckt,fk->bcft", inputs, self.filter_inv.to(inputs.device))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super(PositionalEncoding, self).__init__()
        
        pos_table = np.array([
            [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
            if pos != 0 else np.zeros(d_model) for pos in range(max_len)])
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])           # 字嵌入维度为偶数时
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])           # 字嵌入维度为奇数时
        self.pos_table = torch.FloatTensor(pos_table)               # enc_inputs: [seq_len, d_model]

    def forward(self, enc_inputs):                                  # enc_inputs: [batch_size, seq_len, d_model]
        enc_inputs += self.pos_table[:enc_inputs.size(1), :].to(enc_inputs.device)
        return enc_inputs


def get_attn_pad_mask(len_q, len_k, valid_lens):   
    
    batch_size = valid_lens.size(0)
    pad_attn_mask = torch.arange(len_k, device=valid_lens.device).unsqueeze(0).repeat(batch_size, 1)
    pad_attn_mask[pad_attn_mask < valid_lens] = True
    pad_attn_mask[pad_attn_mask >= valid_lens] = False
    pad_attn_mask = pad_attn_mask.to(torch.bool)
    pad_attn_mask = pad_attn_mask.unsqueeze(1).repeat(1, len_q, 1)

    return pad_attn_mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dk = torch.sqrt(torch.tensor(d_k+0.0))
        self.soft_max = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, attn_mask):
        # Q: [batch_size, n_heads, len_q, d_k]
        # # K: [batch_size, n_heads, len_k, d_k]
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        # attn_mask: [batch_size, n_heads, len_q, len_k]                      

        # scores : [batch_size, n_heads, len_q, len_k]    
        scores = torch.matmul(Q, K.transpose(-1, -2)) / self.sqrt_dk.to(Q.device)
        scores.masked_fill_(attn_mask, -1e9)
        attn = self.soft_max(scores)
        context = torch.matmul(attn, V)   # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.layernorm = nn.LayerNorm(d_model)
        self.product = ScaledDotProductAttention(d_k)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads

    def forward(self, input_Q, input_K, input_V, attn_mask): 
                                                        
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)    # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)    # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)    # V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)                      # attn_mask : [batch_size, n_heads, seq_len, seq_len]
       
        context, attn = self.product(Q, K, V, attn_mask)                                      # context: [batch_size, n_heads, len_q, d_v]
                                                                                              # attn: [batch_size, n_heads, len_q, len_k]
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        output = self.fc(context)
        return self.layernorm(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False))
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, inputs):                    # inputs: [batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)

        return self.layernorm(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, d_k, d_v, n_heads):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)

        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, d_input, d_model, d_ff, d_k, d_v, n_heads, n_layers, dropout=0.1):
        super(Encoder, self).__init__()
        self.eos_emb = nn.Embedding(vocabs_len, d_model) 
        self.src_emb = nn.Linear(d_input, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.pos_emb = PositionalEncoding(d_model) 
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_ff, d_k, d_v, n_heads) for _ in range(n_layers)])

    def forward(self, enc_inputs, valid_lens):                                   # enc_inputs: [batch_size, src_len]
        eos_index_tensor = eos_index*torch.ones((enc_inputs.size(0), 1), dtype=torch.long, device=enc_inputs.device)
        eos_embedding = self.eos_emb(eos_index_tensor)
        pos_outputs = self.src_emb(enc_inputs)                                   # enc_outputs: [batch_size, src_len, d_model]
        pos_outputs = self.dropout(pos_outputs)
        pos_outputs[:, -1] = eos_embedding
        enc_outputs = self.pos_emb(pos_outputs)                                  # enc_outputs: [batch_size, src_len, d_model]
        src_len = enc_outputs.size(1)
        enc_self_attn_mask = get_attn_pad_mask(src_len, src_len, valid_lens)     # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_self_attn_mask = enc_self_attn_mask.to(enc_outputs.device)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)  # enc_outputs :   [batch_size, src_len, d_model],
                                                                                 # enc_self_attn : [batch_size, n_heads, src_len, src_len]
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_heads):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.dec_enc_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
                                                                               
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)     
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)                           
        dec_outputs = self.pos_ffn(dec_outputs)                      

        return dec_outputs, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self, clss, d_model, d_ff, d_k, d_v, n_heads, n_layers):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(clss, d_model)
        self.pos_emb = PositionalEncoding(d_model)

        self.layers = nn.ModuleList([DecoderLayer(d_model, d_ff, d_k, d_v, n_heads) for _ in range(n_layers)])

    def forward(self, dec_inputs, dec_valid_lens, enc_outputs, enc_valid_lens):                 

        batch_size, len_q = dec_inputs.shape
        dec_outputs = self.tgt_emb(dec_inputs)                      
        dec_outputs = self.pos_emb(dec_outputs)  
        dec_self_attn_pad_mask = get_attn_pad_mask(len_q, len_q, dec_valid_lens).to(dec_inputs.device)
        dec_self_attn_subsequence_mask = self.get_attn_subsequence_mask(batch_size, len_q)  
        dec_self_attn_subsequence_mask = dec_self_attn_subsequence_mask.to(dec_inputs.device)
        
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0)   
        len_k = enc_outputs.shape[1]
        dec_enc_attn_mask = get_attn_pad_mask(len_q, len_k, enc_valid_lens) 
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns

    def get_attn_subsequence_mask(self, batch_size, seq_len):                                
        attn_shape = [batch_size, seq_len, seq_len]
        subsequence_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
        return subsequence_mask


class Transformer(nn.Module):
    def __init__(self, win_len, hop_len, nfilters, sr, low_freq, high_freq,
                    d_input, d_model, d_ff, d_k, d_v, n_heads, n_layers, clss, dropout=0.1):
        super(Transformer, self).__init__()
        
        self.stft = STFT(win_len, hop_len, win_len, "hann")
        self.bank = Banks(nfilters, win_len, sr, low_freq, high_freq)
        self.Encoder = Encoder(d_input, d_model, d_ff, d_k, d_v, n_heads, n_layers)
        self.Decoder = Decoder(clss, d_model, d_ff, d_k, d_v, n_heads, n_layers)
        self.projection = nn.Linear(d_model, clss, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, audio, enc_valid_lens, dec_inputs, dec_valid_lens):

        stft = self.stft.transform(audio)
        mag = torch.norm(stft, dim=1, keepdim=True)
        banks = self.bank.amp2bank(mag**2).squeeze(dim=1)
        banks = 10*torch.log10(banks).permute(0, 2, 1)
        enc_outputs, enc_self_attns = self.Encoder(banks, enc_valid_lens)                                     
        dec_outputs, dec_self_attns, dec_enc_attns = self.Decoder(
            dec_inputs, dec_valid_lens, enc_outputs, enc_valid_lens)                                                                                      
        dec_logits = self.projection(dec_outputs)
        dec_logits = self.dropout(dec_logits)

        return (dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns)


if __name__ == '__main__':
    from ptflops import get_model_complexity_info

    global vocabs
    global vocabs_len
    global eos_index
    vocabs, vocabs_len, eos_index = construct_global(0.01)

    device = 'cpu'
    # model = Transformer(win_len=2048, hop_len=128, nfilters=512, sr=16000, low_freq=20, 
    #                     high_freq=8000, d_input=512, d_model=512, d_ff=1024, d_k=64, 
    #                     d_v=64, n_heads=6, n_layers=8, clss=vocabs_len)
    model = Transformer(win_len=2048, hop_len=128, nfilters=512, sr=16000, low_freq=20, 
                        high_freq=8000, d_input=512, d_model=512, d_ff=1024, d_k=64, 
                        d_v=64, n_heads=6, n_layers=8, clss=vocabs_len)
    model = model.eval()
    model = model.to(device)

    def prepare_input(shapes):
        audio = torch.zeros((1, int(16000*4.088)), dtype=torch.float32).to(device)
        enc_valid_lens = 500*torch.ones((1, 1), dtype=torch.int32).to(device)
        dec_inputs = torch.zeros((1, 1024), dtype=torch.long).to(device)
        dec_valid_lens = 1024*torch.ones((1, 1), dtype=torch.int32).to(device)

        return {'audio': audio, 'enc_valid_lens': enc_valid_lens, 'dec_inputs': dec_inputs,
                'dec_valid_lens': dec_valid_lens }

    macs, params = get_model_complexity_info(model, input_res=(1, 16000),input_constructor=prepare_input, print_per_layer_stat=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    