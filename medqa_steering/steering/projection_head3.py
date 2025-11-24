# steering/projection_head3.py

import os, csv, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.medqa_dataset import MedQADataset, LETTER
from data.prompt_builder import build_prompt
from model.loader import load_model
from model.hooks import last_hidden_last_token
from steering.io import load_vectors, save_proj
from config import DEVICE, TARGET_LAYER, LOG_DIR, BATCH_SIZE, PROJ_PATH

LOG_CSV = os.path.join(LOG_DIR, "projection_head3_loss.csv")
os.makedirs(LOG_DIR, exist_ok=True)


class ProjHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.P = nn.Linear(dim, dim, bias=False)
    def forward(self, h):
        return self.P(h)


@torch.no_grad()
def hidden_and_probs(tok, model, stem, choices):
    flat = []
    for c in choices:
        if isinstance(c, (list, tuple)) and len(c) == 1:
            flat.append(c[0])
        else:
            flat.append(c)
    if len(flat) < 4:
        flat = flat + [flat[-1]] * (4 - len(flat))
    elif len(flat) > 4:
        flat = flat[:4]

    prompt = build_prompt(stem, flat)
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    out = model(**inputs)

    h = last_hidden_last_token(out, TARGET_LAYER).squeeze(0).float()
    logits = model.lm_head(out.hidden_states[-1][:, -1, :])
    ids = [tok.convert_tokens_to_ids(x) for x in LETTER]
    z4 = logits[:, ids].squeeze(0)
    probs = z4.softmax(dim=-1)
    return h, probs


def train(split="train", epochs=3):
    tok, model = load_model()
    vec_dict = load_vectors()
    dim = next(iter(vec_dict.values())).numel()

    head = ProjHead(dim).to(DEVICE)
    opt = optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-4)
    crit = nn.CosineEmbeddingLoss(margin=0.0)

    ds = MedQADataset(split=split)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    with open(LOG_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "step", "loss", "pos_frac"])

        for epoch in range(epochs):
            pbar = tqdm(loader, desc=f"Proj3 epoch {epoch}")
            for step, batch in enumerate(pbar):
                H_list, V_list, t_list = [], [], []

                for stem, choices, label in zip(
                    batch["stem"], batch["choices"], batch["label"]
                ):
                    h, probs = hidden_and_probs(tok, model, stem, choices)
                    y = int(label)
                    pred = int(torch.argmax(probs).item())
                    key = LETTER[y]
                    if key not in vec_dict:
                        continue

                    is_pos = 1.0 if pred == y else -1.0
                    H_list.append(h)
                    V_list.append(vec_dict[key])
                    t_list.append(is_pos)

                if not H_list:
                    continue

                H = torch.stack(H_list).to(DEVICE).float()
                V = torch.stack(V_list).to(DEVICE).float()
                t = torch.tensor(t_list, dtype=torch.float32, device=DEVICE)

                Hn = H / (H.norm(dim=-1, keepdim=True) + 1e-8)
                Vn = V / (V.norm(dim=-1, keepdim=True) + 1e-8)

                proj = head(Hn)
                proj = proj / (proj.norm(dim=-1, keepdim=True) + 1e-8)

                loss = crit(proj, Vn, t)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                pos_frac = float((t > 0).float().mean().item())
                pbar.set_postfix(loss=float(loss), pos=pos_frac)
                w.writerow([epoch, step, float(loss), pos_frac])

    save_proj(head)
    print(f"[INFO] Saved projection head3 → {PROJ_PATH}")
    return head
