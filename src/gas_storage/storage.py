"""Storage contract (capacity, rate limits, costs) and the daily bounds on the physical flow."""
from dataclasses import dataclass, field
from typing import Self

import torch
from torch import Tensor


@dataclass
class StorageSpec:
    capacity: float           # c (MWh)
    inj_max: Tensor           # (K,)  u_k >= 0
    wdr_max: Tensor           # (K,)  l_k <= 0
    kappa: float              # proportional cost
    fixed_cost: float         # C
    remaining: Tensor = field(init=False)

    @classmethod
    def paper(cls, K: int = 351, capacity: float = 250_000.0, 
              kappa = 0.0, fixed_cost = 0.0) -> Self:
        """u = 2808 (k <= 200) else 408;  l = -600 (k <= 170) else -3072."""

        #Create the tensors
        k = torch.arange(K)

        inj_max = torch.where(k <= 200.0, 2808.0, 408.0)
        wdr_max = torch.where(k <= 170.0, -600.0, -3072.0)

        return cls(capacity = capacity, inj_max = inj_max, wdr_max = wdr_max,
                   kappa = kappa, fixed_cost = fixed_cost)

    def __post_init__(self):
        self.remaining = self.remaining_withdrawal()

    def remaining_withdrawal(self) -> Tensor:
        """(K+1,)  R_k = sum_{j >= k} |l_j|, with R_K = 0."""

        # I need one value more because both H_k and R_k have one more value than trading days
        K = self.wdr_max.shape[0]
        R = torch.zeros(K + 1)

        # At each index i, R_k holds the sum of abs daily withdrawal limit
        # Last index is already set to zero, so no need to acces here i = K (i = K cause we count from 0 included)
        # Vectorized loop        
        R[:K] = self.wdr_max.abs().flip(0).cumsum(0).flip(0)

        return R

def flow_bounds(spec: StorageSpec, k: int, level: Tensor) -> tuple[Tensor, Tensor]:
    """Bounds (lo, hi), each (B,), on the physical flow q_k into storage on day k.

    lo = max(l_k, -H_k)
    hi = min(u_k, c - H_k, R_{k+1} - H_k)   # last term: strict 'empty at the end' guarantee
    """

    l_k = spec.wdr_max[k]
    u_k = spec.inj_max[k]
    H_k = level
    R_k_1 = spec.remaining[k+1]

    c = spec.capacity

    lo = torch.maximum(l_k, -H_k)
    hi = torch.minimum(torch.minimum(u_k, c - H_k), R_k_1 - H_k)

    return lo, hi
   
