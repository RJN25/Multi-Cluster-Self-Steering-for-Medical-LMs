import os
import re
import csv
from datasets import load_dataset

LOG_PATH = "logs/eval.log"
CSV_PATH = "logs/eval_results.csv"

# Load MedQA test set
ds = load_dataset("GBaker/MedQA-USMLE-4-options-hf")["test"]

# Updated regex for new log format
# Example line:
# test-00000: Correct=0 | Conf=0.296 | Brier=0.0718 | ECE=0.2960
line_re = re.compile(
    r"(?:Q|test-)(\d+): Correct=(\d) \| Conf=([\d.]+)(?: \| Brier=([\d.]+))?(?: \| ECE=([\d.]+))?"
)

rows = []

with open(LOG_PATH, "r") as f:
    for line in f:
        m = line_re.search(line)
        if not m:
            continue

        idx = int(m.group(1))
        correct_flag = int(m.group(2))
        conf = float(m.group(3))
        brier = float(m.group(4)) if m.group(4) else None
        ece = float(m.group(5)) if m.group(5) else None

        if idx < len(ds):
            entry = ds[idx]
            qid = entry.get("id", f"test-{idx:05d}")
            question = entry["sent1"]
            options = [entry[f"ending{i}"] for i in range(4)]
            true_label = entry["label"]
        else:
            qid, question, options, true_label = f"test-{idx:05d}", "", ["", "", "", ""], None

        pred = ["A","B","C","D"][true_label] if correct_flag == 1 else "?"

        rows.append({
            "qid": qid,
            "question": question,
            "option_A": options[0],
            "option_B": options[1],
            "option_C": options[2],
            "option_D": options[3],
            "true_answer": ["A","B","C","D"][true_label] if true_label is not None else "",
            "predicted_answer": pred,
            "correct": correct_flag,
            "confidence": conf,
            "brier": brier,
            "ece": ece,
        })

# Safety guard
if not rows:
    print("No matching log lines found. Check your logs/eval.log format.")
    exit()

# Save CSV
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
with open(CSV_PATH, "w", newline='', encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"[✓] Exported {len(rows)} rows to {CSV_PATH}")
