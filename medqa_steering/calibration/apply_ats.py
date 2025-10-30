import torch
from calibration.ats_head import ATSHead
from config import ATS_PATH, DEVICE

def load_ats(dim):
    head = ATSHead(dim).to(DEVICE)
    head.load_state_dict(torch.load(ATS_PATH, map_location=DEVICE))
    head.eval()
    return head

@torch.no_grad()
def apply_ats(head, h, logits4):
    # h: [1,d]; logits4: [1,4]
    tau = head(h).clamp(min=1e-3, max=50.0)  # [1,1]
    z = logits4 / tau
    return z.softmax(dim=-1)
