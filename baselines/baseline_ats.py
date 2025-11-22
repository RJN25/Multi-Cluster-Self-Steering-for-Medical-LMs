import csv
import os
from pathlib import Path
from typing import List, Optional

import torch, numpy as np
from tqdm import tqdm

from baselines.calibration2.apply_ats2 import load_ats, apply_ats
from baselines.steering2.config2 import ATS_PATH, LOG_DIR, DEVICE, TARGET_LAYER
from baselines.data2.medqa_dataset2 import MedQADataset, LETTER
from baselines.data2.prompt_builder2 import build_prompt
from baselines.eval2.metrics2 import brier_multiclass, ece_multiclass, macro_auroc_ovr, mean_confidence
from baselines.eval2.logging_setup2 import setup_logger
from baselines.model2.hooks2 import last_hidden_last_token
from baselines.model2.loader2 import load_model


def _ensure_log_dir():
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

def _plot_metric_trends(history: List[dict], split: str) -> Optional[Path]:
    if not history:
        return None

    # Import lazily to avoid the heavy matplotlib dependency unless plotting.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    steps = [point["step"] for point in history]
    metric_keys = [
        ("accuracy", "Accuracy"),
        ("brier", "Brier score"),
        ("ece", "ECE"),
        ("auroc", "Macro AUROC"),
    ]

    fig, axes = plt.subplots(len(metric_keys), 1, figsize=(8, 8), sharex=True)

    for ax, (key, label) in zip(axes, metric_keys):
        values = [point.get(key) for point in history]
        valid = [
            (step, value)
            for step, value in zip(steps, values)
            if value is not None and not np.isnan(value)
        ]

        if valid:
            valid_steps, valid_values = zip(*valid)
            ax.plot(valid_steps, valid_values, label=label)
        else:
            ax.text(0.5, 0.5, "Not available", ha="center", va="center", transform=ax.transAxes)

        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))

    axes[-1].set_xlabel("Evaluated samples")
    fig.tight_layout()

    plot_path = Path(LOG_DIR) / f"baseline_{split}_metrics.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    return plot_path



def _prompt_hidden_logits(tokenizer, model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    # Generate output
    generation_output = model.generate(**inputs, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(generation_output[0], skip_special_tokens=True)

    outputs = model(**inputs)
    hidden = last_hidden_last_token(outputs, TARGET_LAYER)
    last_layer = outputs.hidden_states[-1][:, -1, :]
    option_ids = [tokenizer.convert_tokens_to_ids(x) for x in LETTER]
    logits4 = model.lm_head(last_layer)[:, option_ids]
    return hidden, logits4, generated_text


def evaluate_baseline(split="validation", csv_name="baseline_ats.csv"):
    dataset = MedQADataset(split)
    tokenizer, model = load_model()
    ats_head = None
    if os.path.exists(ATS_PATH):
        ats_head = load_ats(model.config.hidden_size)
    else:
        raise FileNotFoundError(f"ATS weights not found at {ATS_PATH}. Train or supply a head before evaluation.")

    _ensure_log_dir()
    csv_path = Path(LOG_DIR) / csv_name
    logger = setup_logger("baseline", f"baseline_{split}.log")

    logger.info(f"Loaded MedQA {split} dataset with {len(dataset)} samples.")
    logger.info("Loaded model: Qwen/Qwen2.5-3B-Instruct")
    logger.info(f"Starting baseline evaluation for {len(dataset)} MedQA samples...")

    probs_all, labels, preds = [], [], []
    rows = []
    history: List[dict] = []
    correct_so_far = 0

    for idx, example in enumerate(tqdm(dataset, desc=f"Evaluating {split} baseline"), start=1):
        prompt = build_prompt(example["stem"], list(example["choices"]))
        h_last, logits4, generated_text = _prompt_hidden_logits(tokenizer, model, prompt)
        # Ensure tensors are in the dtype expected by the ATS head.
        h_last = h_last.to(dtype=torch.float32)
        logits4 = logits4.to(dtype=torch.float32)
        probs = apply_ats(ats_head, h_last, logits4).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        label = example["label"]
        confidence = float(probs[pred_idx])

        probs_all.append(probs)
        labels.append(label)
        preds.append(pred_idx)

        probs_arr = np.stack(probs_all, axis=0)
        labels_arr = np.array(labels)
        running_brier = brier_multiclass(probs_arr, labels_arr)
        running_ece = ece_multiclass(probs_arr, labels_arr)
        running_auroc = macro_auroc_ovr(probs_arr, labels_arr)
        running_mean_conf = mean_confidence(probs_arr)

        correct_flag = "1" if pred_idx == label else "0"
        correct_so_far += int(pred_idx == label)
        running_accuracy = correct_so_far / idx
        if np.isnan(running_auroc):
            auroc_str = "nan"
        else:
            auroc_str = f"{running_auroc:.3f}"
        logger.info(
            f"{example['qid']}: Correct={correct_flag} | Conf={confidence:.3f} | MeanConf={running_mean_conf:.3f} | Brier={running_brier:.3f} | ECE={running_ece:.3f} | AUROC={auroc_str}"
        )
        logger.info(f"Generated output for {example['qid']}: {generated_text}")

        history.append(
            {
                "step": idx,
                "accuracy": running_accuracy,
                "brier": running_brier,
                "ece": running_ece,
                "auroc": running_auroc if not np.isnan(running_auroc) else None,
                "mean_conf": running_mean_conf,
            }
        )

        row = {
            "qid": example["qid"],
            "label": label,
            "prediction": pred_idx,
            "confidence": confidence,
            "generated_output": generated_text,
        }
        for i, letter in enumerate(LETTER):
            row[f"p_{letter}"] = float(probs[i])
        rows.append(row)

    probs_arr = np.stack(probs_all, axis=0)
    labels_arr = np.array(labels)
    preds_arr = np.array(preds)

    accuracy = float((preds_arr == labels_arr).mean())
    brier = brier_multiclass(probs_arr, labels_arr)
    ece = ece_multiclass(probs_arr, labels_arr)
    auroc = macro_auroc_ovr(probs_arr, labels_arr)
    final_mean_conf = mean_confidence(probs_arr)

    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        fieldnames = ["qid", "label", "prediction", "confidence", "generated_output"] + [f"p_{l}" for l in LETTER]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metrics = {
        "accuracy": accuracy,
        "brier": brier,
        "ece": ece,
        "macro_auroc": auroc,
    }

    logger.info("Model inference complete.")
    logger.info(
        "ACCURACY={:.4f} | AUROC={} | Brier={:.4f} | ECE={:.4f} | MeanConf={:.4f}".format(
            accuracy,
            f"{auroc:.4f}" if not np.isnan(auroc) else "nan",
            brier,
            ece,
            final_mean_conf
        )
    )
    logger.info(f"Saved per-sample probabilities to {csv_path}")

    plot_path = _plot_metric_trends(history, split)
    if plot_path is not None:
        logger.info(f"Saved metric trend plot to {plot_path}")

    return metrics, csv_path, plot_path


if __name__ == "__main__":
    metrics, path, plot = evaluate_baseline()
    print(f"Saved per-sample probabilities to {path}")
    if plot is not None:
        print(f"Saved metric trend plot to {plot}")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, (int, float)) and not np.isnan(value) else f"{key}: {value}")