"""Statistics and plots (paper Figs. 3, 5, 6)."""
from torch import Tensor

from .market import MarketData
from .rollout import RolloutResult
from .storage import StorageSpec


def summary_stats(pnl: Tensor) -> dict:
    """mean, median, std of the terminal P&L."""
    raise NotImplementedError


def plot_paths(market: MarketData, ax=None):
    raise NotImplementedError


def plot_levels(result: RolloutResult, spec: StorageSpec, ax=None):
    raise NotImplementedError


def plot_pnl_hist(pnls: dict[str, Tensor], ax=None):
    raise NotImplementedError


def plot_pnl_box(pnls: dict[str, Tensor], ax=None):
    raise NotImplementedError
