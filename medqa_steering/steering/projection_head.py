import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.medqa_dataset import MedQADataset, LETTER
from data.prompt_builder import build_prompt
from model.loader import load_model
from model.hooks import last_hidden_last_token
from steering.io import load_vectors, save_proj
from config import DEVICE, TARGET_LAYER, BATCH_SIZE

class ProjHead(nn.Module):
    def __init__(self, dim): 
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=False)
    def forward(self, h): 
        return self.lin(h)

@torch.no_grad()
def batch_hidden_and_targets(tok, model, batch, vec_dict):
    H=[]; T=[]
    for stem, choices, label in zip(batch["stem"], batch["choices"], batch["label"]):
        h, _ = score_once(tok, model, stem, list(choices))
        y = LETTER[int(label)]
        if y in vec_dict:
            H.append(h); T.append(vec_dict[y])
    if not H: return None, None
    return torch.stack(H), torch.stack(T)

@torch.no_grad()
def score_once(tok, model, stem, choices):
    prompt = build_prompt(stem, choices)
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    out = model(**inputs)
    h = last_hidden_last_token(out, TARGET_LAYER).squeeze(0)
    return h, None

def train():
    tok, model = load_model()
    vec_dict = load_vectors()
    any_vec = next(iter(vec_dict.values()))
    dim = any_vec.numel()

    head = ProjHead(dim).to(DEVICE)
    opt = optim.AdamW(head.parameters(), lr=3e-4, weight_decay=0.0)
    crit = nn.MSELoss()

    ds = MedQADataset(split="train")
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(2):
        pbar = tqdm(loader, desc=f"Proj epoch {epoch}")
        for batch in pbar:
            H,T = batch_hidden_and_targets(tok, model, batch, vec_dict)
            if H is None: continue
            H = H.to(DEVICE); T = T.to(DEVICE)
            pred = head(H)
            loss = crit(pred, T)
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix(loss=float(loss))
    save_proj(head)
    return head
