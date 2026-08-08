import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput


# ── Positional Encoding ───────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding — Vaswani et al. (2017)."""
    def __init__(self, d_model: int, max_seq_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


# ── Shared Sublayers ──────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, d_model: int, scale: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * scale)
        self.fc2 = nn.Linear(d_model * scale, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class MHA(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model   = d_model
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.q_proj    = nn.Linear(d_model, d_model, bias=False)
        self.k_proj    = nn.Linear(d_model, d_model, bias=False)
        self.v_proj    = nn.Linear(d_model, d_model, bias=False)
        self.out_proj  = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, E = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, E))


# ── Level 1: Intra-block sublayer gating ─────────────────────────────────────
# Available for later integration into RoutedBlock once routing is verified.

class ScalarGate(nn.Module):
    """
    Bounded exponential gate:
        s(x) = tanh(g(x))  ∈ (-1, 1)
    Used with blend_multiplicative:
        gate * f(x) + (1 - gate) * x   where gate = sigmoid(2s)
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.g = nn.Linear(in_dim, in_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.g(x))


def blend_multiplicative(x: torch.Tensor, f_x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """α·f(x) + β·x  where α=exp(s), β=exp(-s), α·β=1."""
    gate = torch.sigmoid(2 * s)
    return gate * f_x + (1 - gate) * x


# ── Level 2: Intra-block stream mixing ───────────────────────────────────────

class RoutedBlock(nn.Module):
    """
    Pre-norm block with learnable λ blending f_attn and f_mlp.

        x_out = x + λ·f_attn + (1-λ)·f_mlp

        λ → 1 : attention-dominant
        λ → 0 : MLP-dominant
        λ = 0.5 : equal blend at init via sigmoid(0) = 0.5

    ScalarGate (Level 1) wired in after routing is verified.
    """
    def __init__(self, d_model: int, num_heads: int, scale: int = 4):
        super().__init__()
        self.mha   = MHA(d_model, num_heads)
        self.mlp   = MLP(d_model, scale=scale)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.lam   = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5 at init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lam    = torch.sigmoid(self.lam)
        f_attn = self.mha(self.norm1(x))
        f_mlp  = self.mlp(self.norm2(x))
        return x + lam * f_attn + (1 - lam) * f_mlp


# ── Level 3: Intra-layer routing across sequential blocks ────────────────────

class Router(nn.Module):
    """
    Soft routing over num_blocks block outputs.
    Input: last block's output (fully-processed state).

        w = softmax(linear(x_last))  →  [B, T, num_blocks]
    """
    def __init__(self, d_model: int, num_blocks: int):
        super().__init__()
        self.route = nn.Linear(d_model, num_blocks, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.route(x), dim=-1)  # [B, T, num_blocks]


class RoutedLayer(nn.Module):
    """
    One depth position: num_blocks sequential RoutedBlocks.

    Each block receives the previous block's output.
    The router reads the final block's output and produces a weighted
    blend over all intermediate outputs:

        outputs = [Block_0(x), Block_1(Block_0(x)), ...]
        w       = softmax(linear(outputs[-1]))
        out     = Σ_i  w_i · outputs_i

    This lets the router decide how much of each depth of processing
    to surface — shallow vs deep representations within the layer.
    Inter-layer routing emerges from composing RoutedLayers in RoutedDecoder.
    """
    def __init__(self, d_model: int, num_heads: int, num_blocks: int, scale: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList([
            RoutedBlock(d_model, num_heads, scale) for _ in range(num_blocks)
        ])
        self.router = Router(d_model, num_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for block in self.blocks:
            x = block(x)           # sequential: each block sees previous output
            outputs.append(x)

        w   = self.router(outputs[-1])                               # [B, T, num_blocks]
        out = torch.stack(outputs, dim=-1)                           # [B, T, d, num_blocks]
        return (out * w.unsqueeze(-2)).sum(-1)                       # [B, T, d]


# ── Decoder ───────────────────────────────────────────────────────────────────

class RoutedDecoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int,
                 num_blocks: int, scale: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([
            RoutedLayer(d_model, num_heads, num_blocks, scale)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ── Config ────────────────────────────────────────────────────────────────────

class RoutedConfig(PretrainedConfig):
    model_type = "routed_lm"

    def __init__(
        self,
        d_model:    int = 512,
        num_heads:  int = 8,
        num_layers: int = 12,   # sequential RoutedLayer positions (depth)
        num_blocks: int = 2,    # sequential RoutedBlocks per RoutedLayer
        vocab_size: int = 50257,
        scale:      int = 4,
        **kwargs,
    ):
        self.d_model    = d_model
        self.num_heads  = num_heads
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.vocab_size = vocab_size
        self.scale      = scale
        super().__init__(**kwargs)


# ── Model ─────────────────────────────────────────────────────────────────────

class RoutedLM(PreTrainedModel):
    config_class = RoutedConfig

    def __init__(self, config: RoutedConfig):
        super().__init__(config)
        self.pos          = SinusoidalPositionalEncoding(config.d_model)
        self.embedding    = nn.Embedding(config.vocab_size, config.d_model)
        self.decoder      = RoutedDecoder(
            config.d_model, config.num_heads,
            config.num_layers, config.num_blocks, config.scale,
        )
        self.norm         = nn.LayerNorm(config.d_model)
        self.output_layer = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output_layer.weight = self.embedding.weight  # weight tying

        self.post_init()

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # zero-init output projections → identity residual at step 0
        for layer in self.decoder.layers:
            for block in layer.blocks:
                nn.init.zeros_(block.mha.out_proj.weight)
                nn.init.zeros_(block.mlp.fc2.weight)

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels:         Optional[torch.Tensor] = None,
    ) -> CausalLMOutput:
        x      = self.pos(self.embedding(input_ids))
        x      = self.decoder(x)
        x      = self.norm(x)
        logits = self.output_layer(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
            )
        return CausalLMOutput(loss=loss, logits=logits)