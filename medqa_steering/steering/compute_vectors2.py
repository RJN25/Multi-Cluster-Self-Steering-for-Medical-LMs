# steering/compute_vectors2.py

import os, sys, logging
from datetime import datetime
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.medqa_dataset import MedQADataset, LETTER
from data.prompt_builder import build_prompt
from model.loader import load_model
from model.hooks import last_hidden_last_token
from steering.io import save_vectors
from config import DEVICE, TARGET_LAYER, LOG_DIR

LOG_PATH = os.path.join(LOG_DIR, "compute_vectors2.log")
os.makedirs(LOG_DIR, exist_ok=True)


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


log_f = open(LOG_PATH, "a")
sys.stdout = Tee(sys.stdout, log_f)
sys.stderr = Tee(sys.stderr, log_f)
print("\n===== NEW VECTORS2 RUN:", datetime.now(), "=====\n")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
    logits = model.lm_head(out.hidden_states[-1][:, -1, :])  # [1,V]
    ids = [tok.convert_tokens_to_ids(x) for x in LETTER]
    z4 = logits[:, ids].squeeze(0)                           # [4]
    probs = z4.softmax(dim=-1)
    return h, probs


def run(split="train", max_items=None):
    print(f"[INFO] Loading model for vectors2 on split={split}")
    tok, model = load_model()

    ds = MedQADataset(split=split)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    print(f"[INFO] MedQA size = {len(ds)}")

    pos = defaultdict(list)
    neg = defaultdict(list)

    n = 0
    for item in tqdm(loader, total=len(ds)):
        stem = item["stem"][0]
        raw_choices = item["choices"]
        if isinstance(raw_choices, list) and len(raw_choices) == 1:
            raw_choices = raw_choices[0]

        choices = [
            c[0] if isinstance(c, (list, tuple)) and len(c) == 1 else c
            for c in raw_choices
        ]

        y = int(item["label"][0])
        h, probs = hidden_and_probs(tok, model, stem, choices)
        pred = int(torch.argmax(probs).item())
        k = LETTER[y]

        if pred == y:
            pos[k].append(h.cpu())
        else:
            neg[k].append(h.cpu())

        n += 1
        if max_items and n >= max_items:
            break

    vecs = {}
    used = []
    for k in LETTER:
        if len(pos[k]) == 0 or len(neg[k]) == 0:
            logger.warning(f"class {k} has pos={len(pos[k])} neg={len(neg[k])}")
            continue
        hp = torch.stack(pos[k]).mean(0)
        hn = torch.stack(neg[k]).mean(0)
        v = hp - hn
        v = v - v.mean()
        v = v / (v.norm() + 1e-8)
        vecs[k] = v
        used.append(v.unsqueeze(0))

    if used:
        M = torch.cat(used, dim=0)
        g = M.mean(0)
        for k in vecs:
            v = vecs[k] - g
            vecs[k] = v / (v.norm() + 1e-8)

    norms = {k: float(v.norm().item()) for k, v in vecs.items()}
    logger.info(f"steering norms2: {norms}")
    print(f"[INFO] steering norms2: {norms}")

    save_vectors(vecs)
    print("[INFO] Saved steering vectors2 to VEC_PATH")
    print(f"[INFO] Full log → {LOG_PATH}")
    return vecs
