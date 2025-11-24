# steering/projection_head.py
# Train a projection head P: h_norm -> steering space for classes A/B/C/D

import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import tqdm

from data.medqa_dataset import MedQADataset, LETTER
from data.prompt_builder import build_prompt
from model.loader import load_model
from model.hooks import last_hidden_last_token
from steering.io import load_vectors, save_proj
from config import (
    DEVICE,
    TARGET_LAYER,
    LOG_DIR,
    BATCH_SIZE,
    PROJ_PATH,
    NUM_WORKERS,
)

LOG_CSV = os.path.join(LOG_DIR, "projection_head_loss.csv")
os.makedirs(LOG_DIR, exist_ok=True)


class ProjHead(nn.Module):
    """
    Simple linear projection: h_norm (d,) -> projected (d,)
    Trained with cosine embedding loss toward class steering vectors.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.P = nn.Linear(dim, dim, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B,d] or [d]
        return self.P(h)


@torch.no_grad()
def hidden_and_probs(tok, model, stem, choices):
    """
    Forward pass to get hidden state h and class probabilities over A/B/C/D.
    """
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

    h = last_hidden_last_token(out, TARGET_LAYER).squeeze(0).float()  # [d]

    logits = model.lm_head(out.hidden_states[-1][:, -1, :])           # [1,V]
    ids = [tok.convert_tokens_to_ids(x) for x in LETTER]
    z4 = logits[:, ids].squeeze(0).float()                            # [4]
    probs = z4.softmax(dim=-1)

    return h, probs


def train(split: str = "train", epochs: int = 3) -> ProjHead:
    """
    Train projection head so that P(h_norm) aligns (cosine) with class steering vectors.

    Target y:
        +1 if model prediction == true label  (positive example)
        -1 if model prediction != true label  (negative example)
    """
    print(f"[INFO] Training ProjHead on split={split} for {epochs} epoch(s)")

    tok, model = load_model()
    vec_dict = load_vectors()  # contains A_pos, A_neg, A, B_pos, ..., D

    # only need the 4 steering vectors A/B/C/D for supervision.
    steering_vecs = {}
    for k in LETTER:
        if k in vec_dict:
            steering_vecs[k] = vec_dict[k].float()
    if not steering_vecs:
        raise RuntimeError(
            "[ERROR] No class steering vectors (A/B/C/D) found in loaded vec_dict."
        )

    dim = next(iter(steering_vecs.values())).numel()
    print(f"[INFO] Steering / hidden dim = {dim}")

    head = ProjHead(dim).to(DEVICE)
    opt = optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-4)
    crit = nn.CosineEmbeddingLoss(margin=0.0)

    ds = MedQADataset(split=split)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    print(f"[INFO] Dataset size for projection training: {len(ds)}")

    with open(LOG_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "step", "loss", "pos_frac", "batch_size"])

        for epoch in range(epochs):
            pbar = tqdm(loader, desc=f"ProjHead epoch {epoch}")
            for step, batch in enumerate(pbar):
                H_list, V_list, t_list = [], [], []

                for stem, choices, label in zip(
                    batch["stem"], batch["choices"], batch["label"]
                ):
                    h, probs = hidden_and_probs(tok, model, stem, choices)
                    y = int(label)
                    pred = int(torch.argmax(probs).item())
                    key = LETTER[y]

                    if key not in steering_vecs:
                        continue

                    # Target: +1 for correctly predicted sample, -1 otherwise
                    t_val = 1.0 if pred == y else -1.0

                    H_list.append(h)
                    V_list.append(steering_vecs[key])
                    t_list.append(t_val)

                if not H_list:
                    continue

                H = torch.stack(H_list, dim=0).to(DEVICE).float()   # [B,d]
                V = torch.stack(V_list, dim=0).to(DEVICE).float()   # [B,d]
                t = torch.tensor(t_list, dtype=torch.float32, device=DEVICE)  # [B]

                # Normalize H and V to unit-length for cosine embedding
                Hn = H / (H.norm(dim=-1, keepdim=True) + 1e-8)
                Vn = V / (V.norm(dim=-1, keepdim=True) + 1e-8)

                proj = head(Hn)                                      # [B,d]
                proj = proj / (proj.norm(dim=-1, keepdim=True) + 1e-8)

                loss = crit(proj, Vn, t)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                pos_frac = float((t > 0).float().mean().item())
                pbar.set_postfix(loss=float(loss), pos=pos_frac)

                writer.writerow([epoch, step, float(loss), pos_frac, len(H_list)])

    save_proj(head)
    print(f"[INFO] Saved projection head → {PROJ_PATH}")
    return head


if __name__ == "__main__":
    train("train", epochs=3)
