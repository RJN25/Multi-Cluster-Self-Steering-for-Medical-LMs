import os, sys, logging, csv
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm

from data.medqa_dataset import MedQADataset, LETTER
from data.prompt_builder import build_prompt
from model.loader import load_model
from model.hooks import last_hidden_last_token

from steering.io import load_vectors
from steering.projection_head2 import ProjHead
from steering.steer_infer import steer_and_score_letters

from calibration.ats_head import ATSHead
from get_tag_contents import extract_all
from eval.metrics import brier_multiclass, ece_multiclass, macro_auroc_ovr

from config import DEVICE, TARGET_LAYER, PROJ_PATH, ATS_PATH, LOG_DIR


# tee logging
LOG_TXT = os.path.join(LOG_DIR, "eval.log")
CSV_OUT = os.path.join(LOG_DIR, "eval_results.csv")
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

log_f = open(LOG_TXT, "a")
sys.stdout = Tee(sys.stdout, log_f)
sys.stderr = Tee(sys.stderr, log_f)

# print("\n===== START NEW EVAL RUN:", datetime.now(), "=====\n")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# gen with cot
@torch.no_grad()
def generate_with_cot(tok, model, prompt):
    enc = tok(prompt, return_tensors="pt").to(DEVICE)
    out = model.generate(
        **enc,
        max_new_tokens=256,
        do_sample=False
    )
    return tok.decode(out[0], skip_special_tokens=True)


# ats head load
def load_ats_head(dim):
    head = ATSHead(dim).to(DEVICE)
    state = torch.load(ATS_PATH, map_location=DEVICE)
    head.load_state_dict(state)
    head.eval()
    return head


# hidden and logits
@torch.no_grad()
def hidden_and_logits(tok, model, prompt):
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    out = model(**inputs)

    h = last_hidden_last_token(out, TARGET_LAYER).squeeze(0).float()
    logits = model.lm_head(out.hidden_states[-1][:, -1, :])  # [1, vocab]
    ids = [tok.convert_tokens_to_ids(x) for x in LETTER]
    z4 = logits[:, ids][0]  # [4]

    return h, z4


# main eval
def evaluate(split="test"):
    tok, model = load_model()

    vec_dict = load_vectors()
    dim = next(iter(vec_dict.values())).numel()
    print(f"[INFO] Hidden dim = {dim}")

    proj = ProjHead(dim).to(DEVICE)
    proj.load_state_dict(torch.load(PROJ_PATH, map_location=DEVICE))
    proj.eval()

    ats = load_ats_head(dim)

    ds = MedQADataset(split=split)
    print(f"[INFO] Loaded {len(ds)} samples for evaluation.")

    probs_all = []
    labels = []
    confidences = []
    corrects = []

    f_csv = open(CSV_OUT, "w", newline="")
    writer = csv.writer(f_csv)
    writer.writerow([
        "qid",
        "true_label",
        "pred_label",
        "correct",
        "calibrated_confidence",
        "cot_raw",
        "cot_answer",
        "cot_confidence",
        "cot_analysis"
    ])

    for item in tqdm(ds):
        stem = item["stem"]
        choices = item["choices"]
        y = int(item["label"])

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

        # steering pass
        p_pre, info = steer_and_score_letters(tok, model, prompt, proj, vec_dict)

        # ats calibration
        h, z4 = hidden_and_logits(tok, model, prompt)
        tau = ats(h.unsqueeze(0))          # [1,1] or [1]
        tau_scalar = tau.squeeze()         # []
        zcal = z4 / tau_scalar             # [4]
        p_cal = torch.softmax(zcal, dim=-1)

        pred = int(torch.argmax(p_cal).item())
        conf = float(p_cal[pred].item())
        correct = int(pred == y)

        probs_all.append(p_cal.cpu().numpy())
        labels.append(y)
        confidences.append(conf)
        corrects.append(correct)

        cot_raw = generate_with_cot(tok, model, prompt)
        cot_ans, cot_conf, cot_analysis = extract_all(cot_raw)

        qid = item["qid"] if "qid" in item else ""
        writer.writerow([
            qid,
            y,
            pred,
            correct,
            conf,
            cot_raw,
            cot_ans,
            cot_conf,
            cot_analysis
        ])

        print(f"[INFO] {qid}: pred={pred} correct={correct} conf={conf:.3f}")

    f_csv.close()

    probs_all = np.stack(probs_all)
    labels_np = np.array(labels)

    acc = float(np.mean(corrects))
    mean_conf = float(np.mean(confidences))
    brier = float(brier_multiclass(probs_all, labels_np))
    ece = float(ece_multiclass(probs_all, labels_np))
    auroc = float(macro_auroc_ovr(probs_all, labels_np))

    print("\n=== METRICS ===")
    print(f"Accuracy {acc:.4f}")
    print(f"Mean Conf {mean_conf:.4f}")
    print(f"Brier {brier:.4f}")
    print(f"ECE {ece:.4f}")
    print(f"AUROC {auroc:.4f}")
    print(f"Results saved → {CSV_OUT}")

    return dict(
        acc=acc,
        mean_conf=mean_conf,
        brier=brier,
        ece=ece,
        auroc=auroc,
    )
