# context_fusion.py
import torch, torch.nn as nn, torch.nn.functional as F
from typing import Optional          # ← ajouter

class ConcatProjection(nn.Module):
    """[sent, ctx] → Linear → même dim que sent"""
    def __init__(self, d_sent: int, d_ctx: int):
        super().__init__()
        self.proj = nn.Linear(d_sent + d_ctx, d_sent)

    def forward(self, sent, ctx):
        x = torch.cat([sent, ctx], dim=-1)
        return self.proj(x)


class GatedAdd(nn.Module):
    """sent + σ( Wg·[sent,ctx] ) ⊙ Wc·ctx"""
    def __init__(self, d_sent: int, d_ctx: int):
        super().__init__()
        self.w_ctx = nn.Linear(d_ctx, d_sent, bias=False)
        self.w_gate = nn.Linear(d_sent + d_ctx, d_sent)

    def forward(self, sent, ctx):
        gate = torch.sigmoid(self.w_gate(torch.cat([sent, ctx], dim=-1)))
        ctx_p = self.w_ctx(ctx)
        return sent + gate * ctx_p


# context_fusion.py
import torch, torch.nn as nn, torch.nn.functional as F

# … tes classes existantes …

class FiLMModulation(nn.Module):
    """
    Applique une modulation FiLM : sent' = γ ⊙ sent + β
    γ et β sont déduits du vecteur ctx via un petit MLP.
    """

    def __init__(self,
                 d_sent: int,
                 d_ctx: int,
                 hidden: Optional[int] = None):  # ← au lieu de  int | None

        super().__init__()
        h = hidden or max(128, d_ctx // 2)                 # ex. 384 par défaut si d_ctx=768
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_ctx),
            nn.Linear(d_ctx, h),
            nn.ReLU(),
            nn.Linear(h, 2 * d_sent)                       # → γ‖β
        )

    def forward(self, sent: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """
        sent : (B,S,d_sent)
        ctx  : (B,S,d_ctx)  (broadcasté / déjà aligné phrase⇄prototype)
        """
        gamma_beta = self.mlp(ctx)                         # (B,S,2*d_sent)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return gamma * sent + beta



class CrossAttentionFusion(nn.Module):
    """
    sent' = LN( sent + W_o · MultiHeadAttn(Q=sent, K=ctx, V=ctx) )
    • d_sent peut être très grand (≈22 k) ; on projette donc vers un
      goulot d'étranglement d_mid≪d_sent avant l'attention.
    """
    def __init__(self,
                 d_sent: int,
                 d_ctx: int,
                 d_mid: int = 1024,
                 num_heads: int = 8):
        super().__init__()
        self.q_proj = nn.Linear(d_sent, d_mid, bias=False)
        self.k_proj = nn.Linear(d_ctx, d_mid, bias=False)
        self.v_proj = nn.Linear(d_ctx, d_mid, bias=False)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_mid,
            num_heads=num_heads,
            batch_first=True
        )
        self.out_proj = nn.Linear(d_mid, d_sent, bias=False)
        self.ln = nn.LayerNorm(d_sent)

    def forward(self, sent, ctx):
        """
        sent : (B, S, d_sent)    – représentation de phrase
        ctx  : (B, S, d_ctx)     – prototype aligné
        """
        q = self.q_proj(sent)          # (B,S,d_mid)
        k = self.k_proj(ctx)
        v = self.v_proj(ctx)
        attn_out, _ = self.attn(q, k, v, need_weights=False)
        fused = self.out_proj(attn_out)           # (B,S,d_sent)
        return self.ln(sent + fused)



# -----------------------------------------------------------
#  Conditional LayerNorm Fusion
# -----------------------------------------------------------
class ConditionalLayerNorm(nn.Module):
    """
    sent' = γ(p) ⊙ LN(sent) + β(p)
    où γ,β sont produits par un MLP sur le prototype p.
    """
    def __init__(self, d_sent: int, d_ctx: int, hidden: Optional[int] = None):
        super().__init__()
        self.ln = nn.LayerNorm(d_sent, elementwise_affine=False)
        h = hidden or max(128, d_ctx // 2)
        self.mlp = nn.Sequential(
            nn.Linear(d_ctx, h),
            nn.ReLU(),
            nn.Linear(h, 2 * d_sent)
        )

    def forward(self, sent, ctx):                  # (B,S,·)
        g_b = self.mlp(ctx)                        # (B,S,2*d_sent)
        gamma, beta = g_b.chunk(2, dim=-1)
        return gamma * self.ln(sent) + beta

# -----------------------------------------------------------
#  ReZero wrapper pour n’importe quel fusor existant
# -----------------------------------------------------------
class ReZeroFusion(nn.Module):
    """
    sent' = sent + α * Fuse(sent, ctx)
    α est initialisé à 0 → au début, on reproduit le baseline.
    """
    def __init__(self, inner_fusor: nn.Module):
        super().__init__()
        self.fusor = inner_fusor
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, sent, ctx):
        return sent + self.alpha * self.fusor(sent, ctx)



import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentiveProtoFusion(nn.Module):
    """
    Module de fusion attentionnelle entre les embeddings de phrases
    (shape: B x S x d_sent) et un ensemble de prototypes (B x S x P x d_ctx).
    """

    def __init__(self, d_sent, d_ctx):
        super().__init__()
        self.query_proj = nn.Linear(d_sent, d_ctx)
        self.key_proj = nn.Linear(d_ctx, d_ctx)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sent_vecs, proto_vecs):
        """
        Args:
            sent_vecs   : (B, S, d_sent)     — embeddings des phrases
            proto_vecs : (B, S, P, d_ctx)    — prototypes associés à chaque phrase

        Returns:
            fusionnés  : (B, S, d_sent)      — phrase enrichie par attention sur prototypes
        """
        B, S, P, d_ctx = proto_vecs.shape
        d_sent = sent_vecs.shape[-1]

        # (B, S, d_sent) → (B, S, 1, d_ctx)
        q = self.query_proj(sent_vecs).unsqueeze(2)      # (B, S, 1, d_ctx)

        # (B, S, P, d_ctx)
        k = self.key_proj(proto_vecs)

        # Attention par produit scalaire
        attn_scores = torch.sum(q * k, dim=-1)           # (B, S, P)
        attn_weights = self.softmax(attn_scores)         # (B, S, P)

        # Appliquer attention aux prototypes
        ctx_fused = torch.sum(proto_vecs * attn_weights.unsqueeze(-1), dim=2)  # (B, S, d_ctx)

        # Optionnel : concat ou addition avec l'embedding original
        # return sent_vecs + ctx_fused  # gated_add (option simple)
        return ctx_fused  # ou à injecter dans un autre fusor ensuite


import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentiveProtoSelector(nn.Module):
    """
    Sélecteur de prototype par attention dure : sélectionne uniquement le
    prototype ayant le plus fort score d’attention pour chaque phrase.

    Entrée :
      - sentence_repr : (B, S, D)
      - prototypes    : (B, S, P, D)
    Sortie :
      - ctx_vector    : (B, S, D)
    """
    def __init__(self, d_query, d_key):
        super().__init__()
        self.d_query = d_query
        self.d_key = d_key

        self.query_proj = nn.Linear(d_query, d_key)
        self.key_proj = nn.Linear(d_key, d_key)
        self.scale = d_key ** 0.5

    def forward(self, sentence_repr, prototypes):
        """
        Arguments :
            sentence_repr : Tensor (B, S, D)
            prototypes    : Tensor (B, S, P, D)
        Retour :
            ctx_vector    : Tensor (B, S, D)
        """
        B, S, P, D = prototypes.shape

        # Projeter query (phrase) et keys (prototypes)
        Q = self.query_proj(sentence_repr)          # (B, S, Dk)
        K = self.key_proj(prototypes)               # (B, S, P, Dk)

        # Calcule des scores d’attention
        Q = Q.unsqueeze(2)                          # (B, S, 1, Dk)
        scores = torch.matmul(Q, K.transpose(-2, -1)).squeeze(2) / self.scale  # (B, S, P)

        # Indice du prototype avec score max
        max_indices = torch.argmax(scores, dim=-1)  # (B, S)

        # Récupère le vecteur du prototype le plus pertinent
        ctx_vector = torch.gather(
            prototypes, dim=2, index=max_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, D)
        ).squeeze(2)  # (B, S, D)


        return ctx_vector