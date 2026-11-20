"""Run a strategy through the whole year on a batch of paths (shared by SMod and SFMod)."""
from dataclasses import dataclass

from torch import Tensor

from .config import ModelConfig
from .market import MarketData
from .policy import PolicyNetworks
from .storage import StorageSpec


def build_features(k: int, level: Tensor, market: MarketData, price_scale: float,
                   spec: StorageSpec, cfg: ModelConfig) -> Tensor:
    """(B, n_inputs) normalized inputs, day-k information only (no peeking)."""

    # Noramlization
    

    


@dataclass
class RolloutResult:
    flow: Tensor        # (B, K)    physical flow q_k
    h_spot: Tensor      # (B, K)
    h_fwd: Tensor       # (B, K)    zeros for SMod
    level: Tensor       # (B, K+1)  H_0 ... H_K
    pnl: Tensor         # (B,)      W (scaled)
    pnl_spot: Tensor    # (B,)
    pnl_fwd: Tensor     # (B,)


def rollout(policy: PolicyNetworks, market: MarketData, spec: StorageSpec,
            cfg: ModelConfig) -> RolloutResult:
    raise NotImplementedError
