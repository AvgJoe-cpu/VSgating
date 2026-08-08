from typing import Any

import torch 
import torch.nn as nn
import torch.nn.functional as F

class ScalarGate(nn.Module):
    """
    Implements the bounded, symmetric, exponential gating mechanism:

        • g : ℝᵈ → ℝᵈ  (learnable per-dim projection)
        • c              (learnable scalar bound, initialised at c_init)
        • s(x) = c · tanh(g(x))
        • α(x) = exp( s(x) )
        • β(x) = exp(−s(x))   (⇒ α · β = 1)
    """
    def __init__(self, in_dim: int):
        """
        Args
        ----
        in_dim : size of the feature vector x
        c_init : initial value of the learnable bound c
        """
        super().__init__()
        self.g = nn.Linear(in_dim, in_dim, bias=False)

    def forward(self, x: torch.Tensor):
        s = torch.tanh(self.g(x))
        return s  # just return s, let the blend use it directly


class ScalarGate2(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.g = nn.Linear(in_dim, in_dim, bias=False)

    def forward(self, x: torch.Tensor, f_x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.g(f_x - x))  # s ∈ (-1, 1), gate sees the sublayer's proposed update

def blend_multiplicative(x, f_x, s):
    gate = torch.sigmoid(2 * s)
    return gate * f_x + (1 - gate) * x


def main() -> None:
    in_dim = 4
    x = torch.randn(2, in_dim)  # batch of 2 samples,
    f_x = torch.randn(2, in_dim)  # corresponding outputs from some function

    gate = ScalarGate(in_dim)
    alpha, beta, s = gate(x)
    print("α:", alpha)
    print("β:", beta)
    print("s:", s)
    blended = blend_multiplicative(x, f_x, alpha, beta)
    print("Blended output:", blended)

if __name__ == "__main__":
    main()