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


# -------------------- Logging --------------------
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


# -------------------- Forward pass --------------------
@torch.no_grad()
def hidden_and_probs(tok, model, stem, choices):
    """
    Returns hidden-state vector h ∈ R^d and probability distribution over A/B/C/D.
    """
    # Always enforce 4 choices properly
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

    # Hidden state from target layer
    h = last_hidden_last_token(out, TARGET_LAYER).squeeze(0).float()  # [d]

    # Extract logits for A/B/C/D tokens
    logits = model.lm_head(out.hidden_states[-1][:, -1, :])           # [1,V]
    ids = [tok.convert_tokens_to_ids(x) for x in LETTER]
    z4 = logits[:, ids].squeeze(0).float()                            # [4]
    probs = z4.softmax(dim=-1)

    return h, probs


# -------------------- Main compute --------------------
def run(split="train", max_items=None):
    print(f"[INFO] Loading model for compute_vectors2 split={split}")
    tok, model = load_model()

    ds = MedQADataset(split=split)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    print(f"[INFO] Dataset contains {len(ds)} samples.")

    # pos['A'] = list of h where predicted = true_label = A
    # neg['A'] = list of h where predicted != A but true_label = A
    pos = defaultdict(list)
    neg = defaultdict(list)

    n = 0
    for item in tqdm(loader, total=len(ds)):
        stem = item["stem"][0]
        raw_choices = item["choices"]

        # Clean choices
        choices = []
        for c in raw_choices:
            if isinstance(c, (list, tuple)) and len(c) == 1:
                choices.append(c[0])
            else:
                choices.append(c)

        y = int(item["label"][0])          # correct class index
        true_letter = LETTER[y]

        # Run model
        h, probs = hidden_and_probs(tok, model, stem, choices)
        pred = int(torch.argmax(probs).item())

        # Correct or incorrect relative to true class
        if pred == y:
            pos[true_letter].append(h.cpu())
        else:
            neg[true_letter].append(h.cpu())

        n += 1
        if max_items and n >= max_items:
            break

    # -------------------- Compute centroids + steering vectors --------------------
    vecs = {}
    for k in LETTER:
        pos_list = pos[k]
        neg_list = neg[k]

        if len(pos_list) == 0 or len(neg_list) == 0:
            logger.warning(f"[WARN] Class {k}: pos={len(pos_list)} neg={len(neg_list)} — skipping steering vector.")
            continue

        hp = torch.stack(pos_list, dim=0).mean(0)   # positive centroid
        hn = torch.stack(neg_list, dim=0).mean(0)   # negative centroid

        # Save raw centroids
        vecs[f"{k}_pos"] = hp.clone()
        vecs[f"{k}_neg"] = hn.clone()

        # Steering vector = (pos - neg), mean-centered, normalized
        v = hp - hn
        v = v - v.mean()
        v = v / (v.norm() + 1e-8)

        vecs[k] = v.clone()

    # -------------------- Save & log --------------------
    norms = {k: float(v.norm().item()) for k, v in vecs.items()}
    logger.info(f"[INFO] Steering/centroid norms = {norms}")
    print(f"[INFO] Steering/centroid norms = {norms}")

    save_vectors(vecs)
    print("[INFO] Saved vectors (pos/neg + steering) to:", LOG_PATH.replace(".log",""))
    print(f"[INFO] Full log → {LOG_PATH}")

    return vecs


if __name__ == "__main__":
    run("train")
