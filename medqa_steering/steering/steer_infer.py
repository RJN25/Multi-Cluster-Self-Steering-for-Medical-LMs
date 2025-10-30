import torch, torch.nn.functional as F
from model.hooks import last_hidden_last_token
from config import TARGET_LAYER, ALPHA, DEVICE
from data.medqa_dataset import LETTER

def steer_and_score_letters(tok, model, prompt, proj_head, vec_dict, cosine_gate=None):
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    out = model(**inputs)

    # hidden before steering
    h = last_hidden_last_token(out, TARGET_LAYER)           # [1,d]
    with torch.no_grad():
        pvec = proj_head(h)                                 # [1,d]
        # choose v* with max cosine
        best_k, best_cos, best_v = None, -1e9, None
        for k, v in vec_dict.items():
            v = v.to(DEVICE).unsqueeze(0)
            cos = F.cosine_similarity(pvec, v).item()
            if cos > best_cos:
                best_k, best_cos, best_v = k, cos, v
        if (cosine_gate is None) or (best_cos < cosine_gate):
            # inject steering by modifying last layer's last token state -> rebuild logits
            mod_last = out.hidden_states[-1].clone()
            mod_last[:, -1, :] = mod_last[:, -1, :] + ALPHA * best_v
            logits = model.lm_head(mod_last[:, -1, :])      # [1,V]
        else:
            logits = model.lm_head(out.hidden_states[-1][:, -1, :])

        ids = [tok.convert_tokens_to_ids(x) for x in LETTER]
        sel = logits[:, ids]                                # [1,4]
        probs = sel.softmax(dim=-1).squeeze(0)             # [4]
    return probs, dict(best_class=best_k, cosine=best_cos)
