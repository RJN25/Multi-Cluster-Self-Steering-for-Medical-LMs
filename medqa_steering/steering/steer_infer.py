# steering/steer_infer.py
# Inference-time steering: normalize h → project → select best class vector → modify hidden state

import torch
import torch.nn.functional as F

from model.hooks import last_hidden_last_token
from data.medqa_dataset import LETTER
from config import DEVICE, TARGET_LAYER, ALPHA


@torch.no_grad()
def steer_and_score_letters(tok, model, prompt, proj_head, vec_dict, cosine_gate=None):
    """
    Steps:
    1. Tokenize + run LM forward
    2. Extract last-token hidden state (layer TARGET_LAYER)
    3. Normalize h → project via projection head
    4. Compute cosine similarity(P(h_norm), v_class) for 4 class steering vectors
    5. Inject best vector: h' = h + α v_best  (optional gate)
    6. Recompute logits for A/B/C/D
    7. Return probabilities + best_k + best_cosine
    """

    # 1. Forward pass
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    out = model(**inputs)

    # 2. Extract hidden state [1,d]
    h = last_hidden_last_token(out, TARGET_LAYER).float()  # shape: [1,d]

    # 3. Normalize h → project
    hn = h / (h.norm(dim=-1, keepdim=True) + 1e-8)         # [1,d]
    proj = proj_head(hn)                                    # [1,d]
    proj = proj / (proj.norm(dim=-1, keepdim=True) + 1e-8)  # normalized proj

    # 4. Select best steering vector among A/B/C/D
    best_k = None
    best_cos = -1e9
    best_v = None

    for k in LETTER:
        if k not in vec_dict:   # skip centroid entries like A_pos/A_neg
            continue
        v = vec_dict[k].to(DEVICE).float()
        v = v / (v.norm() + 1e-8)

        cos = F.cosine_similarity(proj, v.unsqueeze(0)).item()
        if cos > best_cos:
            best_cos = cos
            best_k = k
            best_v = v

    # 5. Inject steering vector into hidden state
    last = out.hidden_states[-1].clone()   # full hidden states [1,T,d]

    if cosine_gate is None or best_cos >= cosine_gate:
        last[:, -1, :] = last[:, -1, :] + ALPHA * best_v  # apply steering

    # 6. Compute logits at final position
    logits = model.lm_head(last[:, -1, :])  # shape [1,V]

    # Extract logits for A/B/C/D tokens
    ids = [tok.convert_tokens_to_ids(x) for x in LETTER]
    z4 = logits[:, ids].squeeze(0)
    probs = torch.softmax(z4.to(torch.float32), dim=-1)

    # 7. Return probabilities + steering info
    info = {
        "best_class": best_k,
        "best_cosine": best_cos,
    }
    return probs, info
