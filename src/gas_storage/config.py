"""Configuration dataclasses for the models and the training loop."""
from dataclasses import dataclass
from typing import Self


@dataclass
class ModelConfig:
    trade_forwards: bool = False                 # False = SMod, True = SFMod
    alpha: float = 0.0                           # forward volume cap (eq. 16)
    fwd_cap: str = "cumulative"                  # "per_trade" (paper literal) | "cumulative"
    features: tuple[str, ...] = ("k", "H", "S")  # SFMod: ("k", "H", "S", "F", "D")
    net_sharing: str = "month"                   # "day" (N=K) | "month" (N=12) | "shared" (N=1)
    hidden: int = 16
    n_hidden_layers: int = 1                     # L = 2 in the paper's notation
    risk_aversion: float = 3.0                   # r
    pnl_scale: float = 1e6                       # P&L reported in millions

    @classmethod
    def smod(cls) -> Self:
        raise NotImplementedError

    @classmethod
    def sfmod(cls, alpha: float) -> Self:
        raise NotImplementedError


@dataclass
class TrainConfig:
    epochs: int = 1000
    batch_size: int = 64
    lr: float = 0.05
    fresh_paths: bool = False                    # True: new simulated paths every batch
    seed: int = 0
    device: str = "cpu"
