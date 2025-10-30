import torch, logging
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm

from data.medqa_dataset import MedQADataset, LETTER
from data.prompt_builder import build_prompt
from model.loader import load_model
from model.hooks import last_hidden_last_token
from get_tag_contents import extract_all
from steering.io import save_vectors
from config import DEVICE, TARGET_LAYER, LOG_DIR

logging.basicConfig(filename=f"{LOG_DIR}/compute_vectors.log", level=logging.INFO) # independent of main script and file location
logger = logging.getLogger(__name__)

@torch.no_grad()
def score_logits_letters(tok, model, prompt): ## pre logit processing
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    out = model(**inputs)
    h = last_hidden_last_token(out, TARGET_LAYER)
    logits = model.lm_head(out.hidden_states[-1][:, -1, :])
    ids = [tok.convert_tokens_to_ids(x) for x in LETTER]
    sel = logits[:, ids]  # [1,4]
    probs = sel.softmax(dim=-1).squeeze(0)  # [4]
    return h.squeeze(0), probs

def run(split="train", max_items=None):
    tok, model = load_model()
    ds = MedQADataset(split=split)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    pos = defaultdict(list)  # per true class
    neg = defaultdict(list)

    n=0
    for item in tqdm(loader, total=len(ds)):
        stem = item["stem"][0]
        raw_choices = item["choices"]
    
        # Handle batch_size=1 wrapping and flatten inner tuples
        if isinstance(raw_choices, list) and len(raw_choices) == 1:
            raw_choices = raw_choices[0]
    
        # Flatten inner one-element tuples like ('Ampicillin',)
        choices = [c[0] if isinstance(c, (list, tuple)) and len(c) == 1 else c for c in raw_choices]
    
        y = int(item["label"][0])
        prompt = build_prompt(stem, choices)
        h, probs = score_logits_letters(tok, model, prompt)
        pred = probs.argmax().item()

        # group by CLASS = y (ground truth); pos if pred==y else neg
        (pos if pred==y else neg)[LETTER[y]].append(h.detach().cpu())
        n+=1
        if max_items and n>=max_items: break 


    vecs = {}
    for k in LETTER:
        if len(pos[k])==0 or len(neg[k])==0:
            logger.warning(f"Class {k}: insufficient examples (pos={len(pos[k])}, neg={len(neg[k])})")
            continue
        hp = torch.stack(pos[k]).mean(0)
        hn = torch.stack(neg[k]).mean(0)
        vecs[k] = hp - hn

    save_vectors(vecs)
    logger.info({k: v.norm().item() for k,v in vecs.items()})
    return vecs
