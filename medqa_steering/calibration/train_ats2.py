import os, sys, logging, csv
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
LOSS_CSV = os.path.join("logs", "ats_loss.csv")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

log_f = open(LOG_TXT, "a")
sys.stdout = Tee(sys.stdout, log_f)
sys.stderr = Tee(sys.stderr, log_f)
print("\n===== START NEW ATS RUN:", datetime.now(), "=====\n")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# loss

def selective_loss(q_logits, y_idx, alpha=0.5):
    ce = nn.CrossEntropyLoss(reduction="none")
    base = ce(q_logits, y_idx)                # [B]
    with torch.no_grad():
        correct = (q_logits.argmax(dim=-1) == y_idx).float()  # [B]
    uni = -torch.log_softmax(q_logits, dim=-1).mean(dim=-1)   # [B]
    return torch.where(correct > 0, (1 - alpha) * base, alpha * uni).mean()


# feature & logit extraction 

@torch.no_grad()
def hidden_and_logits(tok, model, batch):
    H, Z = [], []

    for stem, choices in zip(batch["stem"], batch["choices"]):

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
        z = logits[:, ids].squeeze(0)

        H.append(h)
        Z.append(z)

    return torch.stack(H), torch.stack(Z)


# training

def train(split="validation"):
    print(f"\n===== START NEW ATS RUN: {datetime.now()} =====\n")

    tok, model = load_model()
    print(f"[INFO] Loading MedQA dataset split: {split}")

    ds = MedQADataset(split=split)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    # derive hidden dimension using one batch
    h0, _ = hidden_and_logits(tok, model, next(iter(loader)))
    hid_dim = h0.shape[-1]
    print(f"[INFO] Hidden dim for ATS head: {hid_dim}")

    head = ATSHead(hid_dim).to(DEVICE)
    opt = optim.AdamW(head.parameters(), lr=5e-5)

    for epoch in range(2):
        pbar = tqdm(loader, desc=f"ATS epoch {epoch}")

        for batch in pbar:
            # === FIX: use the helper that returns perfectly aligned tensors ===
            H, Z = hidden_and_logits(tok, model, batch)
            H = H.to(DEVICE).float()        # [B, d]
            Z = Z.to(DEVICE).float()        # [B, 4]
            y = batch["label"].to(DEVICE).long()  # [B]

            tau = head(H)                   # [B, 1]
            Zcal = Z / tau                  # [B, 4]
            loss = selective_loss(Zcal, y, alpha=0.5)

            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.set_postfix(loss=float(loss))

    torch.save(head.state_dict(), ATS_PATH)
    print(f"[INFO] Saved ATS head → {ATS_PATH}")

    return head

