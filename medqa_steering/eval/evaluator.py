import os, sys, logging, csv
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from data.medqa_dataset import MedQADataset, LETTER
from data.prompt_builder import build_prompt
from model.loader import load_model

from steering.io import load_vectors
from steering.steer_infer import steer_and_score_letters
from steering.projection_head import ProjHead
from get_tag_contents import extract_all

from config import DEVICE, PROJ_PATH, LOG_DIR, ALPHA, COSINE_GATE

# log files
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

print("\n===== START NEW EVAL RUN:", datetime.now(), "=====\n")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@torch.no_grad()
def generate_with_cot(tok, model, prompt: str) -> str:
    """
    Generate CoT-style output (with <think>, <answer>, <analysis>, <confidence> tags)
    for logging only. Does NOT affect metrics.
    """
    enc = tok(prompt, return_tensors="pt").to(DEVICE)
    out = model.generate(
        **enc,
        max_new_tokens=256,
        do_sample=False
    )
    return tok.decode(out[0], skip_special_tokens=True)


def _compute_auroc(corrects, confidences) -> float:
    """
    Compute AUROC from binary labels (correct: 0/1) and confidence scores in [0,1].
    Uses rank-based formula to avoid external deps.
    """
    labels = np.array(corrects, dtype=np.int32)
    scores = np.array(confidences, dtype=np.float64)

    # Handle degenerate cases
    pos = labels.sum()
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")

    # Rank-based AUROC: equivalent to Mann–Whitney U statistic normalized
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)

    pos_ranks_sum = ranks[labels == 1].sum()
    auc = (pos_ranks_sum - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def evaluate(split: str = "test"):
    tok, model = load_model()

    # Load steering vectors (A/B/C/D + centroids) and projection head
    vec_dict = load_vectors()
    dim = next(iter(vec_dict.values())).numel()
    print(f"[INFO] Hidden dim (from vectors) = {dim}")

    proj = ProjHead(dim).to(DEVICE)
    proj.load_state_dict(torch.load(PROJ_PATH, map_location=DEVICE))
    proj.eval()

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
        "mcq_confidence",
        "best_class",
        "best_cosine",
        "cot_raw",
        "cot_answer",
        "cot_conf_tag",
        "cot_analysis",
    ])

    for item in tqdm(ds):
        stem = item["stem"]
        choices = item["choices"]
        y = int(item["label"])

        # Flatten choices to 4 options A/B/C/D
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

        # ---- 1) Steering + MCQ probabilities (our main method) ----
        with torch.no_grad():
            probs, info = steer_and_score_letters(
                tok, model, prompt, proj, vec_dict, cosine_gate=COSINE_GATE
            )  # probs: [4] over A/B/C/D

        pred = int(torch.argmax(probs).item())
        conf = float(probs[pred].item())
        correct = int(pred == y)

        probs_all.append(probs.detach().float().cpu().numpy())
        labels.append(y)
        confidences.append(conf)
        corrects.append(correct)

        # ---- 2) Chain-of-Thought generation AFTER steering decision (for logging ONLY) ----
        cot_raw = generate_with_cot(tok, model, prompt)
        cot_ans, cot_conf_tag, cot_analysis = extract_all(cot_raw)

        qid = item.get("qid", "NA") if isinstance(item, dict) else "NA"

        writer.writerow([
            qid,
            y,
            pred,
            correct,
            conf,
            info.get("best_class", None),
            info.get("best_cosine", None),
            cot_raw,
            cot_ans,
            cot_conf_tag,
            cot_analysis,
        ])

        print(
            f"[INFO] {qid}: pred={pred} correct={correct} "
            f"conf={conf:.3f} best={info.get('best_class')} "
            f"cos={info.get('best_cosine'):.3f if info.get('best_cosine') is not None else float('nan')}"
        )

    f_csv.close()

    probs_all = np.stack(probs_all)
    labels_np = np.array(labels)

    # ---- Metrics: Accuracy, Mean Conf, Brier, ECE, AUROC ----
    acc = float(np.mean(corrects))
    mean_conf = float(np.mean(confidences))

    # Brier score for multi-class
    eye = np.eye(4)[labels_np]  # one-hot
    brier = float(np.mean(np.sum((probs_all - eye) ** 2, axis=1)))

    # ECE with 10 bins
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    conf_arr = np.array(confidences)
    corr_arr = np.array(corrects)
    for i in range(10):
        mask = (conf_arr >= bins[i]) & (conf_arr < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc_bin = float(corr_arr[mask].mean())
        conf_bin = float(conf_arr[mask].mean())
        ece += (mask.sum() / len(conf_arr)) * abs(acc_bin - conf_bin)

    auroc = _compute_auroc(corrects, confidences)

    print("\n=== METRICS ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Mean Conf: {mean_conf:.4f}")
    print(f"Brier: {brier:.4f}")
    print(f"ECE: {ece:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print(f"Results saved → {CSV_OUT}")

    return dict(acc=acc, mean_conf=mean_conf, brier=brier, ece=ece, auroc=auroc)


if __name__ == "__main__":
    evaluate("test")
