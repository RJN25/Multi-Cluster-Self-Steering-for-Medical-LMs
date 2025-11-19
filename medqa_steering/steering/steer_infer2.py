# steering/steer_infer2.py

import torch
import torch.nn.functional as F

from model.hooks import last_hidden_last_token
from data.medqa_dataset import LETTER
from config import DEVICE, TARGET_LAYER, ALPHA


@torch.no_grad()
def steer_and_score_letters(tok, model, prompt, proj_head, vec_dict, cosine_gate=None):
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    out = model(**inputs)

    h = last_hidden_last_token(out, TARGET_LAYER).float()  # [1,d]
    proj = proj_head(h)                                   # [1,d]
    proj = proj / (proj.norm(dim=-1, keepdim=True) + 1e-8)

    best_k = None
    best_cos = -1e9
    best_v = None

    for k, v in vec_dict.items():
        v = v.to(DEVICE).float()
        v = v / (v.norm() + 1e-8)
        cos = F.cosine_similarity(proj, v.unsqueeze(0)).item()
        if cos > best_cos:
            best_cos = cos
            best_k = k
            best_v = v

    last = out.hidden_states[-1].clone()

    if (cosine_gate is None) or (best_cos >= cosine_gate):
        last[:, -1, :] = last[:, -1, :] + ALPHA * best_v
    logits = model.lm_head(last[:, -1, :])

    ids = [tok.convert_tokens_to_ids(x) for x in LETTER]
    z4 = logits[:, ids].squeeze(0)
    probs = torch.softmax(z4.to(torch.float32), dim=-1)

    info = dict(best_class=best_k, cosine=best_cos)
    return probs, info
