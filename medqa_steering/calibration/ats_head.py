import torch, torch.nn as nn

class ATSHead(nn.Module):
    """
    Predicts scalar temperature τ from hidden state h (last-layer last-token),
    then scales logits as z / exp(τ). We output τ_raw and use exp to ensure positivity.
    """
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim//2),
            nn.GELU(),
            nn.Linear(dim//2, 1),
        )
    def forward(self, h):       # h: [B,d]
        tau = torch.exp(self.mlp(h)).clamp(min=1e-3, max=50.0)  # [B,1]
        return tau
