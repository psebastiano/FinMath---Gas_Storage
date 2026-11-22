"""Run a strategy through the whole year on a batch of paths (shared by SMod and SFMod)."""
from dataclasses import dataclass

import torch
from torch import Tensor

from .config import ModelConfig
from .market import MarketData
from .policy import PolicyNetworks
from .storage import StorageSpec, flow_bounds


def build_features(k: int, level: Tensor, market: MarketData, price_scale: float,
                   spec: StorageSpec, cfg: ModelConfig) -> Tensor:
    """(B, n_inputs) normalized inputs, day-k information only (no peeking)."""

    K = market.n_days
    columns = []

    for name in cfg.features:
        if name == "k":
            columns.append(torch.full_like(level, k / K))
            #This puts the day in [0, 1] - works better with Sigmoids
            #Makes a row per path in B, using level tensor as template.
        
        elif name == "H":
            #Storage level as a ratio of the storage capacity
            columns.append(level / spec.capacity)
        
        elif name == "S":
            #Today spot price, scaled
            columns.append(market.spot[:,k] / price_scale)

        elif name == "F":
            #Today forward month
            fwd_k = torch.nan_to_num(market.fwd[:, k], nan=0.0)
            columns.append(fwd_k / price_scale)
        else:
            raise ValueError(f"Unknown feature: {name}")
        
    # This returns the list of feature rows as a tensor of shape (B,4)
    return torch.stack(columns, dim=1)



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
            cfg: ModelConfig, price_scale: float) -> RolloutResult:
    """Simulate the strategy day by day on all B paths in paralle"""
    B, K = market.spot.shape

    H = torch.zeros(B)
    levels = [H]
    flows = []

    for k in range(K):
        x = build_features(k, H, market, price_scale, spec, cfg)

        a = policy(k, x)[:, 0]

        lo, hi = flow_bounds(spec, k, H)

        q = lo + a * (hi - lo)

        H = H + q

        flows.append(q)
        levels.append(H)

    flow = torch.stack(flows, dim=1)
    level = torch.stack(levels,dim=1)

    h_spot = flow
    h_fwd = torch.zeros_like(flow)

    pnl_spot = -(h_spot * market.spot).sum(dim=1) / cfg.pnl_scale
    pnl_fwd = torch.zeros(B)
    pnl = pnl_spot + pnl_fwd

    return RolloutResult(flow=flow, h_spot=h_spot, h_fwd=h_fwd, level=level,
                         pnl=pnl, pnl_spot=pnl_spot, pnl_fwd=pnl_fwd)