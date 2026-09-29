# loss.py  ──────────────────────────────────────────────────────────────
"""
Pertes Prototype, avec MASQUAGE des positions PAD.

Toutes les fonctions attendent :
    • logits  : (B, S, C)
    • labels  : (B, S)
    • embed   : (B, S, D)   (== sent_repr)
    • protos  : (Q, D)
    • mask    : (B, S)      (0 = PAD, 1 = actif)

Seules les positions actives contribuent aux pertes.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from prototype_net.distance import pairwise_dist


# ----------------------------------------------------------------------
def _active(t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Renvoie *uniquement* les éléments où mask == 1.

    • t : (B,S, …)  ou (B,S)
    • mask : (B,S)
    """
    flat = t.reshape(-1, *t.shape[2:])          # (B·S, …)
    mask_f = mask.reshape(-1).bool()            # (B·S,)
    return flat[mask_f]                         # (N, …)  N = Σ mask


# ----------------------------------------------------------------------
def loss_classification(logits: torch.Tensor,
                        labels: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
    """
    CE masquée -– identique à celle de LinearSoftmaxOutputLayer.
    """
    return F.cross_entropy(
        _active(logits, mask),                  # (N,C)
        _active(labels, mask)                   # (N,)
    )


# ----------------------------------------------------------------------
def loss_clustering(embed: torch.Tensor,
                    protos: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    """
    L<sub>c</sub>  =  mean  min<sub>k</sub> ‖ e - P<sub>k</sub> ‖²   (sur les tokens actifs)
    """
    embed_a = F.normalize(_active(embed, mask), dim=-1)  # (N,D)
    protos = F.normalize(protos, dim=-1)  # (Q,D)

    if embed_a.numel() == 0:
        return torch.tensor(0., device=embed.device)

    # print(f"embed_a shape: {embed_a.shape}")
    # print(f"embed_a: {embed_a[0][:5]}")
    # print(f"protos shape: {protos.shape}")
    # print(f"protos: {protos[0][:5]}")
    d = pairwise_dist(embed_a, protos)          # (N,Q)
    # print(f"distance : {d.shape}")
    # print(f"loss : {torch.mean(torch.min(d, dim=-1).values)}" )
    # exit()
    return torch.mean(torch.min(d, dim=-1).values)


# ----------------------------------------------------------------------
def loss_separation(protos: torch.Tensor) -> torch.Tensor:
    """
    L_s = moyenne des distances inter-prototypes (pour les éloigner)
    """
    # Normalisation des prototypes
    protos = F.normalize(protos, dim=-1)

    d = pairwise_dist(protos, protos)           # (Q,Q)
    Q = protos.size(0)

    return torch.mean(d[~torch.eye(Q, dtype=torch.bool, device=protos.device)])
