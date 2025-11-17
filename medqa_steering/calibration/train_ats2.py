import os, sys, logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.medqa_dataset import MedQADataset, LETTER
from data.prompt_builder import build_prompt
from model.loader import load_model
from model.hooks import last_hidden_last_token
from calibration.ats_head import ATSHead
from config import DEVICE, TARGET_LAYER, ATS_PATH, BATCH_SIZE, LOG_DIR


# tee logging

LOG_TXT = os.path.join(LOG_DIR, "ats_train.log")
os.makedirs(LOG_DIR, exist_ok=True)

class Tee:
    def __init__(self, *files): self.files = files
    def write(self, obj):
        for f in self.files: f.write(obj); f.flush()
    def flush(self): [f.flush() for f in self.files]

log_f = open(LOG_TXT, "a")
sys.stdout = Tee(sys.stdout, log_f)
sys.stderr = Tee(sys.stderr, log_f)
print("\n===== START NEW ATS RUN:", datetime.now(), "=====\n")


# selective loss
def selective_loss(q_logits, y_idx, alpha=0.5):
    ce = nn.CrossEntropyLoss(reduction="none")
    base = ce(q_logits, y_idx)

    with torch.no_grad():
        correct = (q_logits.argmax(dim=-1) == y_idx).float()

    uni = -torch.log_softmax(q_logits, dim=-1).mean(dim=-1)

    return torch.where(correct > 0, (1-alpha)*base, alpha*uni).mean()


# extract hidden
# logits
@torch.no_grad()
def hidden_and_logits(tok, model, batch):
    stems  = batch["stem"]
    ch_all = batch["choices"]

    prompts = []
    for choices, stem in zip(ch_all, stems):
        # flatten + sanitize
        flat = []
        for c in choices:
            if isinstance(c, (list, tuple)) and len(c)==1:
                flat.append(c[0])
            else:
                flat.append(c)

        if len(flat) < 4: flat += [flat[-1]]*(4-len(flat))
        elif len(flat) > 4: flat = flat[:4]

        prompts.append(build_prompt(stem, flat))

    # batch tokenization
    inputs = tok(prompts, return_tensors="pt", padding=True).to(DEVICE)
    out = model(**inputs)

    # hidden states for last token of each sample
    H = last_hidden_last_token(out, TARGET_LAYER).float()     # [B, d]

    # take logits only for A/B/C/D tokens
    logits = model.lm_head(out.hidden_states[-1][:, -1, :])   # [B, V]
    ids = [tok.convert_tokens_to_ids(x) for x in LETTER]
    Z = logits[:, ids]                                        # [B, 4]

    return H.cpu(), Z.cpu()


# train
def train(split="validation"):
    print(f"[INFO] Starting ATS on split='{split}'")

    tok, model = load_model()
    ds = MedQADataset(split=split)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    # infer hidden size
    h0, _ = hidden_and_logits(tok, model, next(iter(loader)))
    hid_dim = h0.shape[-1]
    print(f"[INFO] Hidden dim: {hid_dim}")

    head = ATSHead(hid_dim).to(DEVICE)
    opt = optim.AdamW(head.parameters(), lr=5e-5)

    for epoch in range(2):
        pbar = tqdm(loader, desc=f"ATS epoch {epoch}")

        for batch in pbar:
            H, Z = hidden_and_logits(tok, model, batch)

            H = H.to(DEVICE)         # [B, d]
            Z = Z.to(DEVICE)         # [B, 4]
            y = batch["label"].to(DEVICE).long()

            tau = head(H)            # [B, 1]
            Zcal = Z / tau           # [B, 4]

            loss = selective_loss(Zcal, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.set_postfix(loss=float(loss))

    torch.save(head.state_dict(), ATS_PATH)
    print(f"[INFO] Saved ATS → {ATS_PATH}")

    return head
