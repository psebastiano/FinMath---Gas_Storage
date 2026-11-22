"""Utility and training loss."""
import torch
from torch import Tensor


def exp_utility(w: Tensor, r: float) -> Tensor:
    """U(w) = (1 - exp(-r w)) / r."""
    #Here w is the final wealth at the end of the year.
    #the error goes backward only one per year at end, not everyday.
    return (1.0 - torch.exp(-r * w))/r


def loss_fn(pnl: Tensor, r: float) -> Tensor:
    """-mean U(W)  (minus sign: the paper's training criterion has a typo)."""
    return -exp_utility(pnl, r).mean()
    
