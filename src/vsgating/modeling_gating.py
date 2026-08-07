import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput

from vsgating.gating import ScalarGate, blend_multiplicative


class MLP(nn.Module):
    """
    Implements a simple multi-layer perceptron (MLP) with ReLU activations.
    """
    def __init__(self, d_model: int, scale: int = 4):
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_model * scale)
        self.fc2 = nn.Linear(d_model * scale, d_model)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MHA(nn.Module):
    """
    Implements multi-head attention using F.scaled_dot_product_attention.
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_len, d_model)
        B, T, E = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # q, k, v: (batch_size, num_heads, seq_len, head_dim)

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        # attn_out: (batch_size, num_heads, seq_len, head_dim)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, E)
        return attn_out


class GateBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, scale: int = 4):
        super().__init__()
        self.mha = MHA(d_model, num_heads)
        self.mlp = MLP(d_model, scale=scale)
        self.g1 = ScalarGate(d_model, c=0.1)
        self.g2 = ScalarGate(d_model, c=0.1)

    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_len, d_model)
        alpha1, beta1, _ = self.g1(x)
        x = blend_multiplicative(x, self.mha(x), alpha1, beta1)   # residual around self-attention
        alpha2, beta2, _ = self.g2(x)
        x = blend_multiplicative(x, self.mlp(x), alpha2, beta2)   # residual around MLP
        return x


class GateDecoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int, scale: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([GateBlock(d_model, num_heads, scale=scale) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x


# ---- Config ----
class GateConfig(PretrainedConfig):
    model_type = "gate_lm"

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
        vocab_size: int = 50257,
        scale: int = 4,
        device: Optional[str] = None,
        **kwargs,
    ):
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.scale = scale
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(**kwargs)


# ---- Model ----
class GateLM(PreTrainedModel):
    config_class = GateConfig

    def __init__(self, config: GateConfig):
        super().__init__(config)

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.decoder = GateDecoder(
            config.d_model,
            config.num_heads,
            config.num_layers,
            scale=config.scale,
        )
        self.output_layer = nn.Linear(config.d_model, config.vocab_size)


        self.post_init()  # triggers self.apply(self._init_weights)

        # centralized device placement, applied once after full construction
        # dtype is intentionally left to native HF mechanisms:
        #   - TrainingArguments(bf16=True) -> autocast during training
        #   - from_pretrained(..., torch_dtype=...) -> storage dtype at load time
        self.to(device=config.device)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> CausalLMOutput:
        x = self.embedding(input_ids)
        x = self.decoder(x)
        logits = self.output_layer(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return CausalLMOutput(loss=loss, logits=logits)

# config = GateConfig(
#     d_model=64,
#     num_heads=4,
#     num_layers=4,
#     vocab_size=50257,
#     scale=4,
#     device="cpu",        # force cpu for a quick, portable smoke test
#     param_dtype="float32",      # bf16 on cpu can be slow/unsupported for some ops, use fp32 here
# )
# model = GateLM(config)
# print(model)
# # ---- 2. Fake batch, matching your dataset shape ----
# batch_size, seq_len = 2, 16
# input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
# attention_mask = torch.ones_like(input_ids)
# labels = input_ids.clone()

# # ---- 3. Forward pass WITHOUT labels (inference-style) ----
# out_no_labels = model(input_ids=input_ids, attention_mask=attention_mask)
# print("logits shape:", out_no_labels.logits.shape)   # expect (2, 16, 50257)
# print("loss (should be None):", out_no_labels.loss)

# # ---- 4. Forward pass WITH labels (training-style) ----
# out_with_labels = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
# print("loss:", out_with_labels.loss.item())
# assert out_with_labels.loss.requires_grad, "loss should require grad for backward()"

# # ---- 5. Backward pass sanity check ----
# out_with_labels.loss.backward()
# grad_found = any(p.grad is not None for p in model.parameters())
# print("gradients populated:", grad_found)

# # ---- 6. Dtype/device sanity check ----
# print("embedding weight dtype:", model.embedding.weight.dtype)   # expect torch.float32
# print("embedding weight device:", model.embedding.weight.device) # expect cpu        