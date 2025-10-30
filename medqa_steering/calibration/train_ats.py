import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.medqa_dataset import MedQADataset, LETTER
from data.prompt_builder import build_prompt
from model.loader import load_model
from model.hooks import last_hidden_last_token
from calibration.ats_head import ATSHead
from config import DEVICE, TARGET_LAYER, ATS_PATH, BATCH_SIZE

def selective_loss(q_logits, y_idx, alpha=0.5):
    # q_logits: [B, 4] calibrated logits (after scaling), y_idx: [B]
    ce = nn.CrossEntropyLoss(reduction="none")
    base = ce(q_logits, y_idx)  # per-sample
    with torch.no_grad():
        correct = (q_logits.argmax(dim=-1) == y_idx).float()
    uni = -torch.log_softmax(q_logits, dim=-1).mean(dim=-1)  # uniform target
    return torch.where(correct>0, (1-alpha)*base, alpha*uni).mean()

@torch.no_grad()
def hidden_and_logits(tok, model, batch):
    H=[]; Z=[]
    for stem, choices in zip(batch["stem"], batch["choices"]):
        prompt = build_prompt(stem, list(choices))
        inputs = tok(prompt, return_tensors="pt").to(DEVICE)
        out = model(**inputs)
        h = last_hidden_last_token(out, TARGET_LAYER).squeeze(0)     # [d]
        logits = model.lm_head(out.hidden_states[-1][:, -1, :])      # [1,V]
        ids = [tok.convert_tokens_to_ids(x) for x in LETTER]
        z = logits[:, ids].squeeze(0)                                # [4]
        H.append(h.cpu()); Z.append(z.cpu())
    return torch.stack(H), torch.stack(Z)

def train(split="validation"):
    tok, model = load_model()
    ds = MedQADataset(split=split)  # use a held-out slice as calibration set
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    # infer hidden size
    h0, _ = hidden_and_logits(tok, model, next(iter(loader)))
    dim = h0.shape[-1]
    head = ATSHead(dim).to(DEVICE)

    opt = optim.AdamW(head.parameters(), lr=5e-5, betas=(0.9,0.999), weight_decay=0.0)

    for epoch in range(2):
        pbar = tqdm(loader, desc=f"ATS epoch {epoch}")
        for batch in pbar:
            H, Z = hidden_and_logits(tok, model, batch)
            H = H.to(DEVICE); Z = Z.to(DEVICE)
            y = torch.tensor(batch["label"], dtype=torch.long, device=DEVICE)

            tau = head(H)                    # [B,1]
            Zcal = Z / tau                   # temperature scaling
            loss = selective_loss(Zcal, y, alpha=0.5)

            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix(loss=float(loss))
    torch.save(head.state_dict(), ATS_PATH)
    return head
