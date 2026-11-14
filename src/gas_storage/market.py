"""Market data container and price simulators (spot + front-month forward)."""
from dataclasses import dataclass
from typing import Protocol, Self

import torch
from torch import Tensor


@dataclass
class MarketData:
    spot: Tensor          # (M, K)  day-ahead price S_k, EUR/MWh
    fwd: Tensor | None    # (M, K)  price on day k of the contract delivering next month; NaN in last month
    month_start: Tensor   # (J+1,) long, n_0 = 0 < ... < n_J = K

    @property
    def n_paths(self) -> int:
        return self.spot.shape[0]

    @property
    def n_days(self) -> int:
        return self.spot.shape[1]

    def month_of_day(self) -> Tensor:
        """(K,) long: month index of each day."""
        raise NotImplementedError

    def subset(self, idx: Tensor) -> Self:
        """Keep only the paths whose indices are in idx."""
        raise NotImplementedError

    def to(self, device: str | torch.device) -> Self:
        raise NotImplementedError


def train_test_split(data: MarketData, n_train: int, seed: int) -> tuple[MarketData, MarketData]:
    raise NotImplementedError


class MarketSimulator(Protocol):
    def simulate(self, n_paths: int, seed: int) -> MarketData:
        """Return n_paths simulated price paths, reproducible with seed."""
        ...


class DummySimulator:
    """Day-1 placeholder: right shapes, simple seasonal curve + noise."""

    def __init__(self, K: int = 351, n_months: int = 12):
        self.K, self.n_months = K, n_months

    def simulate(self, n_paths: int, seed: int) -> MarketData:
        raise NotImplementedError


@dataclass
class SeasonalOUConfig:
    K: int = 351
    n_months: int = 12
    mean_level: float = 20.0        # EUR/MWh, average seasonal level
    seasonal_amp: float = 5.0       # winter-summer amplitude
    mean_reversion: float = 0.05    # per day
    vol: float = 0.03               # daily volatility of log-price
    fwd_premium: float = 0.0        # drift in forwards (0 = no free forward profit)


class SeasonalOUSimulator:
    """log S_k = seasonal mean + Ornstein-Uhlenbeck deviation."""

    def __init__(self, cfg: SeasonalOUConfig):
        self.cfg = cfg

    def seasonal_mean(self) -> Tensor:
        """(K,) seasonal mean of log-price."""
        raise NotImplementedError

    def simulate(self, n_paths: int, seed: int) -> MarketData:
        raise NotImplementedError
