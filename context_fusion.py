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
