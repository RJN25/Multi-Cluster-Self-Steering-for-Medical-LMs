import os, sys, logging, csv
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from data.medqa_dataset import MedQADataset, LETTER
from data.prompt_builder import build_prompt
from model.loader import load_model
from model.hooks import last_hidden_last_token

from steering.io import load_vectors
from steering.projection_head2 import ProjHead
from calibration.ats_head import ATSHead
from get_tag_contents import extract_all

from config import DEVICE, TARGET_LAYER, PROJ_PATH, ATS_PATH, LOG_DIR, ALPHA

# log files
LOG_TXT = os.path.join(LOG_DIR, "eval.log")
CSV_OUT = os.path.join(LOG_DIR, "eval_results.csv")

os.makedirs(LOG_DIR, exist_ok=True)

class Tee:
    def __init__(self, *files): self.files = files
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
def generate_with_cot(tok, model, prompt):
    enc = tok(prompt, return_tensors="pt").to(DEVICE)
    out = model.generate(
        **enc,
        max_new_tokens=256,
        do_sample=False
    )
    return tok.decode(out[0], skip_special_tokens=True)


def load_ats_head(dim):
    head = ATSHead(dim).to(DEVICE)
    head.load_state_dict(torch.load(ATS_PATH, map_location=DEVICE))
    head.eval()
    return head


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
        "best_class",
        "best_cosine",
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
        inputs = tok(prompt, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            out = model(**inputs)

            h = last_hidden_last_token(out, TARGET_LAYER).float()      # [1,d]
            pvec = proj(h)                                             # [1,d]

            best_k, best_cos, best_v = None, -1e9, None
            for k, v in vec_dict.items():
                v_dev = v.to(DEVICE).unsqueeze(0)                       # [1,d]
                cos = F.cosine_similarity(pvec, v_dev).item()
                if cos > best_cos:
                    best_k, best_cos, best_v = k, cos, v_dev

            last = out.hidden_states[-1].clone()
            last[:, -1, :] = last[:, -1, :] + ALPHA * best_v           # [1,d]
            logits = model.lm_head(last[:, -1, :])                     # [1,V]

            ids = [tok.convert_tokens_to_ids(x) for x in LETTER]
            z4 = logits[:, ids][0]                                     # [4]

            tau = ats(h).view(-1)[0]                                   # scalar
            zcal = z4 / tau                                            # [4]
            p_cal = torch.softmax(zcal, dim=-1)                        # [4]

        pred = int(torch.argmax(p_cal).item())
        conf = float(p_cal[pred].item())
        correct = int(pred == y)

        probs_all.append(p_cal.detach().cpu().numpy())
        labels.append(y)
        confidences.append(conf)
        corrects.append(correct)

        cot_raw = generate_with_cot(tok, model, prompt)
        cot_ans, cot_conf, cot_analysis = extract_all(cot_raw)

        qid = item.get("qid", "NA") if isinstance(item, dict) else "NA"

        writer.writerow([
            qid,
            y,
            pred,
            correct,
            conf,
            best_k,
            best_cos,
            cot_raw,
            cot_ans,
            cot_conf,
            cot_analysis
        ])

        print(f"[INFO] {qid}: pred={pred} correct={correct} conf={conf:.3f} best={best_k} cos={best_cos:.3f}")

    f_csv.close()

    probs_all = np.stack(probs_all)
    labels = np.array(labels)

    acc = float(np.mean(corrects))
    mean_conf = float(np.mean(confidences))

    eye = np.eye(4)[labels]
    brier = float(np.mean(np.sum((probs_all - eye) ** 2, axis=1)))

    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    conf_arr = np.array(confidences)
    corr_arr = np.array(corrects)
    for i in range(10):
        mask = (conf_arr >= bins[i]) & (conf_arr < bins[i+1])
        if mask.sum() == 0:
            continue
        acc_bin = float(corr_arr[mask].mean())
        conf_bin = float(conf_arr[mask].mean())
        ece += (mask.sum() / len(conf_arr)) * abs(acc_bin - conf_bin)

    print("\n=== METRICS ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Mean Conf: {mean_conf:.4f}")
    print(f"Brier: {brier:.4f}")
    print(f"ECE: {ece:.4f}")
    print(f"Results saved → {CSV_OUT}")

    return dict(acc=acc, mean_conf=mean_conf, brier=brier, ece=ece)


if __name__ == "__main__":
    evaluate("test")
