"""
model.py — Core model architecture for MiniLLM-MoE.

Contains: ModelConfig, RMSNorm, MoERouter, SwiGLU, MoEFFN,
GroupedQueryAttention, TransformerBlock, and MiniLLM.

Self-contained: no imports from other project files.
Architecture: decoder-only transformer with MoE FFN (LLaMA-2 + MoE).
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """All architectural hyperparameters for MiniLLM-MoE."""
    vocab_size:        int   = 50257   # GPT-2 BPE vocabulary
    context_len:       int   = 1024    # Max sequence length in tokens
    n_layers:          int   = 12      # Number of transformer blocks
    n_heads:           int   = 12      # Query attention heads
    n_kv_heads:        int   = 12      # Key/value heads (GQA; must divide n_heads)
    hidden_dim:        int   = 768     # Embedding and residual stream dimension
    ffn_dim:           int   = 2048    # Hidden dim inside each expert FFN
    n_experts:         int = 8       # Total MoE expert networks
    n_experts_active:  int = 2       # Top-K experts selected per token
    aux_loss_coeff:    float = 0.01    # Load-balancing loss coefficient
    dropout:           float = 0.1     # Dropout (set 0.0 for 1B+)
    norm_eps:          float = 1e-5    # RMSNorm epsilon
    tie_embeddings:    bool  = True    # Tie input/output embedding weights

    @classmethod
    def from_preset(cls, name: str) -> "ModelConfig":
        """
        Return a ModelConfig for a named scale preset.

        MoE models have more *total* params than the dense equivalent because
        there are N expert FFN networks, but only top-K are active per token,
        so compute is proportional to K/N of total FFN params.
        """
        presets = {
            "30m": dict(
                n_layers=6,  n_heads=6,  n_kv_heads=6,
                hidden_dim=384,  ffn_dim=1024,  n_experts=4,
                n_experts_active=2, context_len=512, dropout=0.1,
            ),
            "70m": dict(
                n_layers=8,  n_heads=8,  n_kv_heads=8,
                hidden_dim=512,  ffn_dim=1536,  n_experts=8,
                n_experts_active=2, context_len=1024, dropout=0.1,
            ),
            "125m": dict(
                n_layers=12, n_heads=12, n_kv_heads=12,
                hidden_dim=768,  ffn_dim=2048,  n_experts=8,
                n_experts_active=2, context_len=1024, dropout=0.1,
            ),
            "350m": dict(
                n_layers=24, n_heads=16, n_kv_heads=8,
                hidden_dim=1024, ffn_dim=2048,  n_experts=8,
                n_experts_active=2, context_len=2048, dropout=0.05,
            ),
            "1b": dict(
                n_layers=32, n_heads=16, n_kv_heads=8,
                hidden_dim=2048, ffn_dim=3072,  n_experts=8,
                n_experts_active=2, context_len=2048, dropout=0.0,
            ),
            "1.5b": dict(
                n_layers=32, n_heads=16, n_kv_heads=8,
                hidden_dim=2048, ffn_dim=4096,  n_experts=8,
                n_experts_active=2, context_len=2048, dropout=0.0,
            ),
        }
        if name not in presets:
            raise ValueError(f"Unknown preset '{name}'. Choose from: {list(presets)}")
        
        return cls(**presets[name])


# ─── RoPE helpers ─────────────────────────────────────────────────────────────

def precompute_rope_freqs(head_dim: int, max_seq_len: int,
                          theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute complex exponentials for Rotary Position Embeddings (RoPE).

    RoPE encodes position by rotating pairs of dimensions in Q and K.
    Using complex arithmetic: each position's freqs_cis vector is applied
    via element-wise multiplication after reshaping the head as complex numbers.

    Returns: (max_seq_len, head_dim // 2) complex64 tensor.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)                      # (T, head_dim/2)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex exponentials


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE to tensor x of shape (B, T, n_heads, head_dim).

    Viewing pairs of real dimensions as complex numbers lets position
    encoding be an exact rotation — a single complex multiply per token.
    """
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # freqs_cis: (T, head_dim/2) → broadcast over batch and heads
    freqs = freqs_cis[: x.shape[1]].unsqueeze(0).unsqueeze(2)  # (1, T, 1, head_dim/2)
    return torch.view_as_real(x_complex * freqs).flatten(3).type_as(x)


# ─── Building blocks ──────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalisation.

    Simpler than LayerNorm: no mean subtraction, no bias.
    Faster and equally effective — used in LLaMA-2/3 and Mistral.
    Formula: x / rms(x) * weight  where  rms(x) = sqrt(mean(x²) + eps).
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to fp32 for numerical stability, then cast back to input dtype
        xf = x.float()
        norm = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight).type_as(x)


class MoERouter(nn.Module):
    """
    Mixture of Experts router.

    A single linear layer maps each token's hidden state to a distribution
    over experts. We select the top-K experts and use their softmax
    probabilities (re-normalised) as combination weights.

    Also computes the auxiliary load-balancing loss to prevent expert collapse.
    Without this, the model learns to always use the same 1-2 experts.

    Aux loss formula (from Switch Transformer / Mixtral):
        aux_loss = coeff * n_experts * Σ_i (fraction_i * mean_prob_i)
    where fraction_i = fraction of tokens that selected expert i,
    and mean_prob_i = mean router softmax probability for expert i.
    """

    def __init__(self, hidden_dim: int, n_experts: int,
                 n_experts_active: int, aux_loss_coeff: float):
        super().__init__()
        self.n_experts        = n_experts
        self.n_experts_active = n_experts_active
        self.aux_loss_coeff   = aux_loss_coeff
        # No bias: routing is purely content-driven
        self.gate = nn.Linear(hidden_dim, n_experts, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (N, hidden_dim) — tokens flattened (N = B * T)
        Returns:
            top_k_weights: (N, K) — re-normalised combination weights
            top_k_indices: (N, K) — expert indices
            aux_loss:      scalar — load-balancing penalty
        """
        router_logits = self.gate(x)                        # (N, n_experts)
        router_probs  = F.softmax(router_logits, dim=-1)    # (N, n_experts)

        top_k_weights, top_k_indices = torch.topk(
            router_probs, self.n_experts_active, dim=-1
        )
        # Re-normalise so the K weights sum to 1 per token
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # Aux loss: count fraction of tokens each expert received,
        # multiply by mean router probability → encourage uniform load
        n_tokens = x.shape[0]
        expert_mask = torch.zeros(
            n_tokens, self.n_experts, device=x.device, dtype=x.dtype
        )
        expert_mask.scatter_(1, top_k_indices, 1.0)

        fraction_tokens  = expert_mask.mean(dim=0)          # (n_experts,)
        mean_router_prob = router_probs.mean(dim=0)         # (n_experts,)

        aux_loss = (
            self.aux_loss_coeff
            * self.n_experts
            * (fraction_tokens * mean_router_prob).sum()
        )
        return top_k_weights, top_k_indices, aux_loss


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward network — one MoE expert.

    Output = down_proj(silu(gate_proj(x)) * up_proj(x)).

    SwiGLU outperforms ReLU and GELU FFNs at all scales studied.
    Three linear projections, all bias=False (PaLM / LLaMA convention).
    """

    def __init__(self, hidden_dim: int, ffn_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.up_proj   = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEFFN(nn.Module):
    """
    Mixture of Experts Feed-Forward Network.

    Contains N SwiGLU expert networks and one MoERouter.
    Each token is independently routed to its top-K experts;
    expert outputs are weighted-summed to produce the final output.

    Token dispatch: for each expert, gather the tokens assigned to it,
    run the expert, scatter results back and accumulate weighted sums.
    This avoids holding N separate variable-length tensors simultaneously.
    """

    def __init__(self, hidden_dim: int, ffn_dim: int, n_experts: int,
                 n_experts_active: int, aux_loss_coeff: float):
        super().__init__()
        self.n_experts        = n_experts
        self.n_experts_active = n_experts_active
        self.experts = nn.ModuleList(
            [SwiGLU(hidden_dim, ffn_dim) for _ in range(n_experts)]
        )
        self.router = MoERouter(hidden_dim, n_experts, n_experts_active, aux_loss_coeff)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, hidden_dim)
        Returns:
            output:   (B, T, hidden_dim)
            aux_loss: scalar
        """
        B, T, D = x.shape
        x_flat = x.view(B * T, D)                           # flatten batch+seq

        top_k_weights, top_k_indices, aux_loss = self.router(x_flat)

        output = torch.zeros_like(x_flat)

        # Outer loop over experts (not K slots) to avoid calling each expert
        # more than once per step — each expert sees all its assigned tokens at once.
        for e in range(self.n_experts):
            for k in range(self.n_experts_active):
                token_mask = top_k_indices[:, k] == e      # (B*T,) bool
                if not token_mask.any():
                    continue
                expert_out = self.experts[e](x_flat[token_mask])  # (n, D)
                output[token_mask] += (
                    top_k_weights[token_mask, k : k + 1] * expert_out
                )

        return output.view(B, T, D), aux_loss


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) with Rotary Position Embeddings (RoPE).

    GQA reduces KV cache size at inference by using fewer K/V heads than Q heads.
    Each group of (n_heads // n_kv_heads) query heads shares one K/V head.
    K/V heads are expanded via repeat_interleave before the attention dot product.

    Uses F.scaled_dot_product_attention which auto-selects Flash Attention 2
    when torch >= 2.0 and a compatible GPU is present — no extra code needed.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_heads % config.n_kv_heads == 0, (
            "n_heads must be divisible by n_kv_heads for GQA"
        )
        self.n_heads    = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep      = config.n_heads // config.n_kv_heads
        self.head_dim   = config.hidden_dim // config.n_heads
        self.dropout    = config.dropout

        self.q_proj = nn.Linear(
            config.hidden_dim, config.n_heads    * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_dim, config.n_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_dim, config.n_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.n_heads * self.head_dim, config.hidden_dim, bias=False
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads,    self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        # Apply RoPE to Q and K (not V — position affects similarity, not values)
        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        # Expand KV heads so each matches its group of Q heads
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)

        # SDPA expects (B, n_heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        dropout_p = self.dropout if self.training else 0.0
        # is_causal=True auto-applies causal mask when no explicit mask provided
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=dropout_p,
            is_causal=(mask is None),
        )

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    """
    One full transformer block (pre-norm architecture).

    Execution order:
      RMSNorm → GQA → residual add → RMSNorm → MoEFFN → residual add.

    Returns (hidden_states, aux_loss) so the caller can accumulate
    aux losses from all layers before computing the final total loss.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_dim, config.norm_eps)
        self.attn  = GroupedQueryAttention(config)
        self.norm2 = RMSNorm(config.hidden_dim, config.norm_eps)
        self.moe   = MoEFFN(
            hidden_dim=config.hidden_dim,
            ffn_dim=config.ffn_dim,
            n_experts=config.n_experts,
            n_experts_active=config.n_experts_active,
            aux_loss_coeff=config.aux_loss_coeff,
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.norm1(x), freqs_cis, mask)
        moe_out, aux_loss = self.moe(self.norm2(x))
        x = x + moe_out
        return x, aux_loss


# ─── Top-level model ──────────────────────────────────────────────────────────

class MiniLLM(nn.Module):
    """
    MiniLLM-MoE: decoder-only transformer with Mixture of Experts FFN.

    Architecture:
        Token Embedding → N × TransformerBlock → RMSNorm → LM Head.

    The LM head weight is tied to the embedding table (weight sharing),
    reducing parameters ~10% — standard in GPT-2, LLaMA, etc.

    Gradient checkpointing can be enabled via gradient_checkpointing_enable()
    to trade ~20-30% throughput for ~60% less activation memory.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self._gradient_checkpointing = False

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.blocks       = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.norm_final = RMSNorm(config.hidden_dim, config.norm_eps)
        self.lm_head    = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Precompute RoPE frequencies once; non-persistent so not saved in state_dict
        head_dim   = config.hidden_dim // config.n_heads
        freqs_cis  = precompute_rope_freqs(head_dim, config.context_len)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # Weight init: N(0, 0.02) following GPT-2, with 1/√(2L) scaling on
        # residual projection layers to keep variance stable with depth.
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def gradient_checkpointing_enable(self):
        """
        Enable gradient checkpointing.

        Recomputes activations during backprop instead of storing them,
        saving ~60% of activation memory at the cost of ~20-30% slower
        training. Essential for 1B+ models on 12GB VRAM.
        """
        self._gradient_checkpointing = True

    def count_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: (B, T) long tensor of token indices
            targets:   (B, T) long tensor, same as input_ids shifted left by 1.
                       Pass None for inference (only last-position logits computed).
        Returns:
            logits: (B, T, vocab_size) training | (B, 1, vocab_size) inference
            loss:   lm_loss + Σ aux_losses across all layers, or None if no targets.
        """
        B, T = input_ids.shape
        assert T <= self.config.context_len, (
            f"Sequence length {T} > context_len {self.config.context_len}"
        )

        x         = self.embed_tokens(input_ids)            # (B, T, hidden_dim)
        freqs_cis = self.freqs_cis[:T]

        total_aux_loss = torch.zeros(1, device=x.device, dtype=x.dtype)

        for block in self.blocks:
            if self._gradient_checkpointing and self.training:
                # Closure captures the specific block — avoids loop-variable bug
                def make_fn(b):
                    def fn(x_, fc_):
                        return b(x_, fc_)
                    return fn
                x, aux_loss = torch.utils.checkpoint.checkpoint(
                    make_fn(block), x, freqs_cis, use_reentrant=False
                )
            else:
                x, aux_loss = block(x, freqs_cis)
            total_aux_loss = total_aux_loss + aux_loss

        x = self.norm_final(x)

        if targets is None:
            # Inference: only compute logits for the last token (saves compute)
            return self.lm_head(x[:, -1:, :]), None

        logits   = self.lm_head(x)                          # (B, T, vocab_size)
        lm_loss  = F.cross_entropy(
            logits.view(B * T, -1), targets.reshape(B * T)
        )
        total_loss = lm_loss + total_aux_loss.squeeze()
        return logits, total_loss

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Autoregressive generation with temperature + top-k + top-p (nucleus) sampling.

        Args:
            input_ids:      (B, T) — prompt token ids
            max_new_tokens: tokens to append
            temperature:    < 1 = sharper, > 1 = more random
            top_k:          keep only the top-k logits before sampling (0 = off)
            top_p:          nucleus: keep smallest set of tokens summing to >= top_p
        Returns:
            (B, T + max_new_tokens) — prompt concatenated with generated tokens
        """
        for _ in range(max_new_tokens):
            # Crop to context window if needed
            ctx = input_ids[:, -self.config.context_len :]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / temperature             # (B, vocab_size)

            if top_k > 0:
                k = min(top_k, logits.size(-1))
                kth = torch.topk(logits, k)[0][:, -1].unsqueeze(-1)
                logits = logits.masked_fill(logits < kth, float("-inf"))

            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs > top_p
                # Shift right so the token that crosses top_p is always kept
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0]  = False
                logits = logits.masked_fill(
                    remove.scatter(1, sorted_idx, remove), float("-inf")
                )

            probs      = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)   # (B, 1)
            input_ids  = torch.cat([input_ids, next_token], dim=1)

        return input_ids
