import os, sys, csv, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from data.medqa_dataset import MedQADataset, LETTER
from data.prompt_builder import build_prompt
from model.loader import load_model
from model.hooks import last_hidden_last_token
from steering.io import load_vectors, save_proj
from config import DEVICE, TARGET_LAYER, LOG_DIR, BATCH_SIZE

LOG_PATH = f"{LOG_DIR}/projection_train.log"

# tee logging
class Tee:
    def __init__(self, *files): self.files = files
    def write(self, obj):
        for f in self.files: f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()

os.makedirs(LOG_DIR, exist_ok=True)
log_f = open(LOG_PATH, "a")
sys.stdout = Tee(sys.stdout, log_f)
sys.stderr = Tee(sys.stderr, log_f)
print("\n===== NEW PROJECTION RUN:", datetime.now(), "=====\n")

        
class ProjHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.P = nn.Linear(dim, dim, bias=False)

    def forward(self, h):
        return self.P(h)

# h state + softmax
@torch.no_grad()
def extract_h_and_probs(tok, model, stem, choices):
    choices = list(choices)
    if len(choices) < 4:
        choices += [choices[-1]] * (4 - len(choices))
    elif len(choices) > 4:
        choices = choices[:4]

    prompt = build_prompt(stem, choices)
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    out = model(**inputs)

    h = last_hidden_last_token(out, TARGET_LAYER).squeeze(0).float()
    logits = model.lm_head(out.hidden_states[-1][:, -1, :])
    ids = [tok.convert_tokens_to_ids(x) for x in LETTER]
    p = logits[:, ids].softmax(dim=-1).squeeze(0)
    return h, p

# train
def train(epochs=3, batch_size=16):
    tok, model = load_model()
    vec_dict = load_vectors()

    # steering vector dimension d
    d = next(iter(vec_dict.values())).numel()
    head = ProjHead(d).to(DEVICE)

    ds = MedQADataset(split="train")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    opt = optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-4)
    crit = nn.CosineEmbeddingLoss()

    # log files (loss + pos/neg true-class probability)
    loss_log = open(f"{LOG_DIR}/projection_loss.csv", "w", newline="")
    prob_log = open(f"{LOG_DIR}/projection_probs.csv", "w", newline="")
    w_loss, w_prob = csv.writer(loss_log), csv.writer(prob_log)
    w_loss.writerow(["epoch", "step", "loss"])
    w_prob.writerow(["epoch", "step", "is_pos", "true_class_prob"])

    print(f"[INFO] Steering vector dimension: {d}")
    print(f"[INFO] Training projection matrix for {epochs} epochs\n")

    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"Projection epoch {epoch}")
        for step, batch in enumerate(pbar):
            stems = batch["stem"]
            choices_all = batch["choices"]
            labels = batch["label"]

            H, Vt, tgt, pos_list, prob_list = [], [], [], [], []

            for stem, choices, y in zip(stems, choices_all, labels):
                y = int(y)
                h, probs = extract_h_and_probs(tok, model, stem, choices)

                pred = torch.argmax(probs).item()
                is_pos = 1 if pred == y else -1
                v = vec_dict[LETTER[y]].to(DEVICE).float()

                H.append(h)
                Vt.append(v)
                tgt.append(is_pos)
                pos_list.append(1 if is_pos == 1 else 0)
                prob_list.append(float(probs[y].item()))

            H = torch.stack(H).to(DEVICE)
            Vt = torch.stack(Vt).to(DEVICE)
            tgt = torch.tensor(tgt, dtype=torch.float32, device=DEVICE)

            proj = head(H)
            loss = crit(proj, Vt, tgt)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            pbar.set_postfix(loss=float(loss))
            w_loss.writerow([epoch, step, float(loss)])
            for isp, pt in zip(pos_list, prob_list):
                w_prob.writerow([epoch, step, isp, pt])

    loss_log.close()
    prob_log.close()
    save_proj(head)

    print(f"\n[✓] Saved projection matrix → steering/projection.pt")
    print(f"[✓] Full log written to → {LOG_PATH}")
    return head
