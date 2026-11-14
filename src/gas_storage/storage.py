"""Storage contract (capacity, rate limits, costs) and the daily bounds on the physical flow."""
from dataclasses import dataclass
from typing import Self

from torch import Tensor


@dataclass
class StorageSpec:
    capacity: float           # c (MWh)
    inj_max: Tensor           # (K,)  u_k >= 0
    wdr_max: Tensor           # (K,)  l_k <= 0
    kappa: float = 0.0        # proportional cost
    fixed_cost: float = 0.0   # C

    @classmethod
    def paper(cls, K: int = 351, capacity: float = 250_000.0) -> Self:
        """u = 2808 (k <= 200) else 408;  l = -600 (k <= 170) else -3072."""
        raise NotImplementedError

    def remaining_withdrawal(self) -> Tensor:
        """(K+1,)  R_k = sum_{j >= k} |l_j|, with R_K = 0."""
        raise NotImplementedError


def flow_bounds(spec: StorageSpec, k: int, level: Tensor) -> tuple[Tensor, Tensor]:
    """Bounds (lo, hi), each (B,), on the physical flow q_k into storage on day k.

    lo = max(l_k, -H_k)
    hi = min(u_k, c - H_k, R_{k+1} - H_k)   # last term: strict 'empty at the end' guarantee
    """
    raise NotImplementedError
