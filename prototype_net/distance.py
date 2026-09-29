# prototype_net/distance.py
# ──────────────────────────────────────────────────────────────
from typing import Literal
import torch
import torch.nn.functional as F


def pairwise_dist(x: torch.Tensor,
                  y: torch.Tensor,
                  metric: Literal["euclidean", "cosine"] = "euclidean",
                  squared: bool = True          # ← NEW
                  ) -> torch.Tensor:
    """
    x : (..., d) , y : (..., d)  →  distances (..., |x|, |y|)
        • squared=True  → ||x-y||²      (par défaut, rétro-compatible)
        • squared=False → ||x-y||      (norme L2)
    """
    if metric == "euclidean":
        diff = x.unsqueeze(-2) - y.unsqueeze(-3)          # (..., N, M, d)
        dist2 = torch.sum(diff * diff, dim=-1)            # (..., N, M)
        return dist2 if squared else torch.sqrt(dist2 + 1e-12)

    if metric == "cosine":
        x_n = F.normalize(x, dim=-1)
        y_n = F.normalize(y, dim=-1)
        # (..., N, d)  @  (..., M, d)ᵀ  →  (..., N, M)
        return 1 - torch.matmul(x_n, y_n.transpose(-1, -2))

    raise ValueError(f"Unknown metric: {metric}")
