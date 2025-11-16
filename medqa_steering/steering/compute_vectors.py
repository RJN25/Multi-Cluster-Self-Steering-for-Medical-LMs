import torch, logging, sys
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime

from data.medqa_dataset import MedQADataset, LETTER
from data.prompt_builder import build_prompt
from model.loader import load_model
from model.hooks import last_hidden_last_token
from steering.io import save_vectors
from config import DEVICE, TARGET_LAYER, LOG_DIR


log_path = f"{LOG_DIR}/compute_vectors.log"

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

log_f = open(log_path, "a")
sys.stdout = Tee(sys.stdout, log_f)
sys.stderr = Tee(sys.stderr, log_f)
print("\n===== START NEW RUN:", datetime.now(), "=====\n")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@torch.no_grad()
def score_logits_letters(tok, model, prompt):
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    out = model(**inputs)
    h = last_hidden_last_token(out, TARGET_LAYER)
    logits = model.lm_head(out.hidden_states[-1][:, -1, :])
    ids = [tok.convert_tokens_to_ids(x) for x in LETTER]
    sel = logits[:, ids]
    probs = sel.softmax(dim=-1).squeeze(0)
    return h.squeeze(0), probs


def run(split="train", max_items=None):
    print("[INFO] Loading model…")
    tok, model = load_model()

    print(f"[INFO] Loading MedQA split: {split}")
    ds = MedQADataset(split=split)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

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
        prompt = build_prompt(stem, choices)
        h, probs = score_logits_letters(tok, model, prompt)
        pred = probs.argmax().item()

        (pos if pred == y else neg)[LETTER[y]].append(h.detach().cpu())
        n += 1
        if max_items and n >= max_items:
            break

    vecs = {}
    for k in LETTER:
        if len(pos[k]) == 0 or len(neg[k]) == 0:
            logger.warning(f"Class {k}: insufficient examples (pos={len(pos[k])}, neg={len(neg[k])})")
            continue
        hp = torch.stack(pos[k]).mean(0)
        hn = torch.stack(neg[k]).mean(0)
        vecs[k] = hp - hn

    save_vectors(vecs)
    logger.info({k: v.norm().item() for k, v in vecs.items()})

    print(f"[INFO] Saved steering vectors → steering/class_vectors.pt")
    print(f"[INFO] Full terminal log written to: {log_path}")

    return vecs
