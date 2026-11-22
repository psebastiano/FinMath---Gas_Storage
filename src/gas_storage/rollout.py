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

    # -- Calendar
    month_start = market.month_start
    J = len(month_start) - 1
    month_of_day = torch.zeros(K, dtype=torch.long)
    for j in range(J):
        month_of_day[int(month_start[j]):int(month_start[j+1])] = j
    days_in_month = month_start[1:] - month_start[:-1]

    # -- State
    H = torch.zeros(B)
    d_now = torch.zeros(B) # Daily deliver this month
    d_next = torch.zeros(B) # Daily deliver next month
        
    levels = [H]
    flows, h_spots, h_fwds = [], [], []
    fwd_cash = []

    for k in range(K):
        j = int(month_of_day[k])

        # New month : last month trades become this month delivery
        if k == int(month_start[j]) and k > 0:
            d_now = d_next
            d_next = torch.zeros(B)

        x = build_features(k, H, market, price_scale, spec, cfg)
        out = policy(k, x)

        # Physical Flow
        lo, hi = flow_bounds(spec, k, H)
        q = lo + out[:, 0] * (hi - lo)
        H = H + q

        h_s = q - d_now

        if cfg.trade_forwards and j <= J - 3:
            n_next = float(days_in_month[j + 1])
            cap = cfg.alpha * spec.capacity / n_next #eq. 16
            h_f = cap * (2.0 * out[:,1] - 1.0) #(0,1) -> [-cap, cap]
            d_next = d_next + h_f
            fwd_cash.append(-h_f * market.fwd[:, k] * n_next)
        else:
            h_f = torch.zeros(B)

        flows.append(q)
        levels.append(H)
        h_spots.append(h_s)
        h_fwds.append(h_f)

    flow = torch.stack(flows, dim=1) #(B,K)
    level = torch.stack(levels,dim=1) #(B,K+1) Because it includes the starting level which is zero before any trading
    h_spot = torch.stack(h_spots, dim=1) #(B,K)
    h_fwd = torch.stack(h_fwds, dim=1) #(B,K)

    pnl_spot = -(h_spot * market.spot).sum(dim=1) / cfg.pnl_scale
    
    if fwd_cash:
        pnl_fwd = torch.stack(fwd_cash, dim=1).sum(dim=1) / cfg.pnl_scale
    else:
        pnl_fwd = torch.zeros(B)
    
    pnl = pnl_spot + pnl_fwd

    return RolloutResult(flow=flow, h_spot=h_spot, h_fwd=h_fwd, level=level,
                         pnl=pnl, pnl_spot=pnl_spot, pnl_fwd=pnl_fwd)