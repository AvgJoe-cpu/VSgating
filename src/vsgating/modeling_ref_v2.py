import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput
from typing import Optional

from vsgating.modeling_gating import MHA, MLP

class RefBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, scale: int = 4):
        super().__init__()
        self.mha = MHA(d_model, num_heads)
        self.mlp = MLP(d_model, scale=scale)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        x = x + self.mha(self.norm1(x))   # pre-norm residual around self-attention
        x = x + self.mlp(self.norm2(x))   # pre-norm residual around MLP
        return x


class RefDecoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int, scale: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([RefBlock(d_model, num_heads, scale=scale) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

# ---- Config ----

# ---- Config ----
class RefConfig(PretrainedConfig):
    model_type = "ref_lm"

    def __init__(
        self,
        d_model: int = 512,
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
class RefLM(PreTrainedModel):
    config_class = RefConfig

    def __init__(self, config: RefConfig):
        super().__init__(config)

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.decoder = RefDecoder(
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
    
# # ---- 1. Build config + model ----
# config = RefConfig(
#     d_model=512,
#     num_heads=4,
#     num_layers=4,
#     vocab_size=50257,
#     scale=4,
#     device="cpu",        # force cpu for a quick, portable smoke test
#     param_dtype="float32",      # bf16 on cpu can be slow/unsupported for some ops, use fp32 here
# )
# model = RefLM(config)
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
print("loss:", out_with_labels.loss.item())
assert out_with_labels.loss.requires_grad, "loss should require grad for backward()"

# ---- 5. Backward pass sanity check ----
out_with_labels.loss.backward()
grad_found = any(p.grad is not None for p in model.parameters())
print("gradients populated:", grad_found)

# ---- 6. Dtype/device sanity check ----
print("embedding weight dtype:", model.embedding.weight.dtype)   # expect torch.float32
print("embedding weight device:", model.embedding.weight.device) # expect cpu    