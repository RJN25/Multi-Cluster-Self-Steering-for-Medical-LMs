import torch

def last_hidden_last_token(outputs, layer_idx=-1):
    # outputs.hidden_states: tuple(len=L+1)[batch, seq, d]
    hs = outputs.hidden_states[layer_idx]           # [B, T, d]
    h_last = hs[:, -1, :]                           # last token per sample
    return h_last                                   # [B, d]
