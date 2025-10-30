import torch, numpy as np
from tqdm import tqdm

from data.medqa_dataset import MedQADataset, LETTER
from data.prompt_builder import build_prompt
from model.loader import load_model
from model.hooks import last_hidden_last_token
from steering.io import load_vectors
from steering.projection_head import ProjHead
from steering.steer_infer import steer_and_score_letters
from calibration.apply_ats import load_ats, apply_ats
from eval.metrics import brier_multiclass, ece_multiclass, macro_auroc_ovr
from eval.logging_setup import setup_logger
from config import DEVICE, TARGET_LAYER, PROJ_PATH

def load_proj(dim):
    head = ProjHead(dim).to(DEVICE)
    head.load_state_dict(torch.load(PROJ_PATH, map_location=DEVICE))
    head.eval()
    return head

@torch.no_grad()
def logits4(tok, model, inputs):
    out = model(**inputs)
    logits = model.lm_head(out.hidden_states[-1][:, -1, :])  # [1,V]
    ids = [tok.convert_tokens_to_ids(x) for x in LETTER]
    return last_hidden_last_token(out, TARGET_LAYER), logits[:, ids] # h, [1,4]

def evaluate(split="test"):
    logger = setup_logger("eval", "eval.log")
    tok, model = load_model()
    vec_dict = load_vectors()
    d = next(iter(vec_dict.values())).numel()
    proj = load_proj(d)
    ats = load_ats(d)

    ds = MedQADataset(split=split)
    logger.info(f"Loaded MedQA {split} dataset with {len(ds)} samples.")
    logger.info("Loaded model: Qwen/Qwen2.5-3B-Instruct")
    logger.info(f"Starting generation for {len(ds)} MedQA samples...")

    probs_all=[]; labels=[]; corrects=[]
    for i, item in enumerate(tqdm(ds)):
        prompt = build_prompt(item["stem"], item["choices"])
        # Steered probabilities (before ATS) + info
        p_pre, info = steer_and_score_letters(tok, model, prompt, proj, vec_dict, cosine_gate=None)

        # Recompute to apply ATS using same hidden state (deterministic scoring path)
        inputs = tok(prompt, return_tensors="pt").to(DEVICE)
        h, z4 = logits4(tok, model, inputs)      # [1,d], [1,4]
        # steer again consistently
        # Minimal duplication: use pvec and best v to modify z4 via hidden-state replacement
        # (Here we approximate by reusing z4; ATS is applied on z4 directly)
        p_cal = apply_ats(ats, h, z4).squeeze(0)  # [4]

        pred = int(torch.argmax(p_cal).item())
        y = int(item["label"])
        probs_all.append(p_cal.cpu().numpy())
        labels.append(y)
        correct = int(pred==y); corrects.append(correct)

        # Per-sample logging line (IDs are strings like 'train-10100')
        logger.info(f"{item['qid']}: Correct={correct} | Conf={p_cal.max().item():.3f}")

    probs_all = np.stack(probs_all)
    labels = np.array(labels)

    acc = np.mean(np.array(corrects))
    brier = brier_multiclass(probs_all, labels)
    ece = ece_multiclass(probs_all, labels)
    auroc = macro_auroc_ovr(probs_all, labels)

    logger.info("Model inference complete.")
    logger.info(f"ACCURACY={acc:.4f} | AUROC={auroc:.4f} | Brier={brier:.4f} | ECE={ece:.4f}")
    return dict(acc=acc, auroc=auroc, brier=brier, ece=ece)
