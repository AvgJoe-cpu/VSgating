# from torch import nn
# import torch
# from dataclasses import dataclass

# from vsgating.modeling_gating import MHA, MLP

# class Block(nn.Module):
#     def __init__(self, d_model: int, num_heads: int, scale: int = 4, device: str=None, dtype: torch.dtype=None):
#         super().__init__()
#         self.mha = MHA(d_model, num_heads, device=device, dtype=dtype)
#         self.mlp = MLP(d_model, scale=scale, device=device, dtype=dtype)

#     def forward(self, x: torch.Tensor):
#         x = x + self.mha(x)  # residual around self-attention
#         x = x + self.mlp(x)  # residual around MLP
#         return x


# class Decoder(nn.Module):
#     def __init__(self, d_model: int, num_heads: int, num_layers: int, scale: int = 4, device: str=None, dtype: torch.dtype=None):
#         super().__init__()
#         self.layers = nn.ModuleList([Block(d_model, num_heads, scale=scale, device=device, dtype=dtype) for _ in range(num_layers)])

#     def forward(self, x: torch.Tensor):
#         for layer in self.layers:
#             x = layer(x)
#         return x


# class LM(nn.Module):
#     def __init__(self, d_model: int, num_heads: int, num_layers: int, vocab_size: int, scale: int = 4, device: str=None, dtype: torch.dtype=None):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, d_model)
#         self.decoder = Decoder(d_model, num_heads, num_layers, scale=scale, device=device, dtype=dtype)
#         self.output_layer = nn.Linear(d_model, vocab_size)

#     def forward(self, x: torch.Tensor):
#         x = self.embedding(x)
#         x = self.decoder(x)
#         logits = self.output_layer(x)
#         return logits


# @dataclass
# class RefModelArgs:
#     d_model: int = 64
#     num_heads: int = 4
#     num_layers: int = 4
#     vocab_size: int = 50257  # GPT-2 tokenizer vocab; swap for your tokenizer's size
#     scale: int = 4           # MLP hidden-size multiplier (d_model * scale)
#     device: str = None
#     dtype: torch.dtype = None

#     def __post_init__(self):
#         assert self.d_model % self.num_heads == 0, (
#             f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
#         )
#         if self.device is None:
#             self.device = "cuda" if torch.cuda.is_available() else "cpu"


# def build_ref_model(args: RefModelArgs) -> "LM":
#     return LM(
#         d_model=args.d_model,
#         num_heads=args.num_heads,
#         num_layers=args.num_layers,
#         vocab_size=args.vocab_size,
#         scale=args.scale,
#         device=args.device,
#         dtype=args.dtype,
#     )


# if __name__ == "__main__":
#     args = RefModelArgs()
#     print(args)
#     model = build_ref_model(args)
#     n_params = sum(p.numel() for p in model.parameters())
#     print(f"Total parameters: {n_params:,}")        