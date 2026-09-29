# prototype_net/layer.py
# ──────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from prototype_net.distance import pairwise_dist


def _mini_kmeans(x: torch.Tensor, k: int, iters: int = 10) -> torch.Tensor:
    """
    x : (N,D)  – retourne (k,D) centroids, aucune grad.
    """
    N, D = x.size()
    # init: échantillon aléatoire
    centroids = x[torch.randperm(N)[:k]].clone()        # (k,D)

    for _ in range(iters):
        dist = pairwise_dist(x, centroids, metric="euclidean", squared=False)  # (N,k)
        assign = torch.argmin(dist, dim=1)                                     # (N,)
        for j in range(k):
            mask = (assign == j)
            if mask.any():
                centroids[j] = x[mask].mean(0)
    return centroids


class PrototypeLayer(nn.Module):
    """
    forward →
        logits_cls : (B,S,C)   (Linear sur embeddings)
        dist_full  : (B,S,Q)   distances aux prototypes
    """

    def __init__(self,
                 n_prototypes: int,
                 n_classes: int,
                 d_embed: int,
                 dist: str = "euclidean",
                 init_method: str = "sample"):          # "sample" | "kmeans"
        super().__init__()
        self.prototypes   = nn.Parameter(torch.empty(n_prototypes, d_embed))
        nn.init.normal_(self.prototypes, std=1.)        # valeur provisoire
        self.initialized = False                        # ← flag lazy-init
        self.init_method = init_method
        self.dist_metric = dist
        self.norm_dist = nn.LayerNorm(n_prototypes)

        # ① tête linéaire sur *embeddings*  (pour la CE standard)
        self.fc_cls = nn.Linear(d_embed, n_classes, bias=True)
        # ② tête linéaire sur *−distances*  (pour la prédiction finale PBN)
        self.fc_dist = nn.Linear(n_prototypes, n_classes, bias=True)

    # -----------------------------------------------------------
    def _lazy_init(self, x: torch.Tensor):
        """
        Initialise les prototypes à partir de x (B,S,D) – sans grad.
        """
        with torch.no_grad():
            flat = x.reshape(-1, x.size(-1))            # (B·S, D)
            if self.init_method == "kmeans":
                centroids = _mini_kmeans(flat, k=self.prototypes.size(0))
            else:                                       # 'sample'
                idx = torch.randperm(flat.size(0))[: self.prototypes.size(0)]
                centroids = flat[idx]

            self.prototypes.copy_(centroids)
            self.initialized = True

    # -----------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        x : (B,S,D)
        """
        if not self.initialized:
            self._lazy_init(x)                          # first batch only

        # (1) logits pour la loss CE (indépendants des prototypes)
        logits_cls = self.fc_cls(x)  # (B,S,C)

        dist = pairwise_dist(x, self.prototypes, self.dist_metric, squared=False)  # (B,S,P)


        # ➜ option : apply instance norm like in original paper
        dist = self.norm_dist(dist)


        logits = self.fc_dist(-dist)  # Plus proche → plus logit positif


        return logits, dist, logits_cls


