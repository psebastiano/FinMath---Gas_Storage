"""Utility and training loss."""
from torch import Tensor


def exp_utility(w: Tensor, r: float) -> Tensor:
    """U(w) = (1 - exp(-r w)) / r."""
    raise NotImplementedError


def loss_fn(pnl: Tensor, r: float) -> Tensor:
    """-mean U(W)  (minus sign: the paper's training criterion has a typo)."""
    raise NotImplementedError
