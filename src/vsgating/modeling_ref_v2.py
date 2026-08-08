import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput
from typing import Optional

from vsgating.modeling_gating import MHA, MLP, SinusoidalPositionalEncoding

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
        self.pos = SinusoidalPositionalEncoding(config.d_model)
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.decoder = RefDecoder(
            config.d_model,
            config.num_heads,
            config.num_layers,
            scale=config.scale,
        )
        self.norm = nn.LayerNorm(config.d_model)                  # post-decoder norm
        self.output_layer = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output_layer.weight = self.embedding.weight          # tie weights

        self.post_init()

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # zero-init output projections → identity at step 0
        for block in self.decoder.layers:
            nn.init.zeros_(block.mha.out_proj.weight)
            nn.init.zeros_(block.mlp.fc2.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> CausalLMOutput:
        x = self.embedding(input_ids)
        x = self.pos(x)
        x = self.decoder(x)
        x = self.norm(x)                                          # pre-head norm
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
#     d_model=1024,
#     num_heads=16,
#     num_layers=24,
#     vocab_size=50257,
#     scale=4,
#     device="cpu",        # force cpu for a quick, portable smoke test
#     param_dtype="float32",      # bf16 on cpu can be slow/unsupported for some ops, use fp32 here
# )
# model = RefLM(config)
# print(model)

# total_params = sum(p.numel() for p in model.parameters())
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Total parameters: {total_params:,}")
# print(f"Trainable parameters: {trainable_params:,}")