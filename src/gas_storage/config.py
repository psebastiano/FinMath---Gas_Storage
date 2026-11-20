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


@dataclass
class DataConfig:
    """Size and train/test split of the simulated data. Reproducibility notes: market.make_data."""
    n_paths: int = 1000                          # M: number of simulated paths (rows of spot/fwd)
    train_frac: float = 0.9                      # 0.9 -> 900/100 as in the paper; scales with n_paths
    seed: int = 0                                # seed for the simulation and the train/test split

    def __post_init__(self) -> None:
        if not 0.0 < self.train_frac < 1.0:
            raise ValueError(f"train_frac must be in (0, 1), got {self.train_frac}")
        if self.n_train < 1 or self.n_test < 1:
            raise ValueError(f"n_paths={self.n_paths} too small for train_frac={self.train_frac}")

    @property
    def n_train(self) -> int:
        return round(self.n_paths * self.train_frac)

    @property
    def n_test(self) -> int:
        return self.n_paths - self.n_train
