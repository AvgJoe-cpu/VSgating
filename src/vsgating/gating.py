from typing import Any

import torch 
import torch.nn as nn
import torch.nn.functional as F

class ScalarGate(nn.Module):
    """
    Implements the bounded, symmetric, exponential gating mechanism:

        • g : ℝᵈ → ℝ   (learnable scalar – here a linear layer)
        • s(x) = c · tanh(g(x))
        • α(x) = exp( s(x) )
        • β(x) = exp(−s(x))   (⇒ α · β = 1)

    The module **only** returns the scalars (α, β); the actual blending of the input x and the function output f(x) is done in a separate function `blend_multiplicative()`.
    """
    def __init__(self, in_dim: int, c: float = 2.0):
        """
        Args
        ----
        in_dim : size of the feature vector x (g’s input dimension)
        c      : constant range bound (s ∈ [−c, c]  ⇒  α,β ∈ [e^(−c), e^c])
        """
        super().__init__()
        self.g = nn.Linear(in_dim, 1, bias=False)
        self.c = c

    def forward(self, x: torch.Tensor):
        """
        Returns
        -------
        α : torch.Tensor, shape (batch, 1, …)   -- α(x) = exp(s(x))
        β : torch.Tensor, shape (batch, 1, …)   -- β(x) = exp(−s(x))
        s : torch.Tensor, shape (batch, 1, …)   -- s(x) itself for inspection
        """
        # s(x) = c · tanh(g(x))     ← Equation (1)
        s = self.c * torch.tanh(self.g(x))

        # α(x) = exp(s(x)),  β(x) = exp(−s(x))   ← Equation (2)
        alpha = torch.exp(s)
        beta  = torch.exp(-s)        # reciprocal by construction

        return alpha, beta, s


def blend_multiplicative(x, f_x, alpha, beta):
    gate = alpha / (alpha + beta)             # ∈ (0, 1)
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