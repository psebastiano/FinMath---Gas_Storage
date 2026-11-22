"""Statistics and plots (paper Figs. 3, 5, 6)."""
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor

from .market import MarketData
from .rollout import RolloutResult
from .storage import StorageSpec


def summary_stats(pnl: Tensor) -> dict:
    """mean, median, std of the terminal P&L."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Data plots (simulated market)
# ---------------------------------------------------------------------------
def add_month_lines(ax, market: MarketData) -> None:
    """Dotted vertical lines at the start of each month."""
    for n in market.month_start[1:-1].tolist():
        ax.axvline(n, color="grey", lw=0.5, ls=":", zorder=0)


def plot_paths(market: MarketData, n_show: int = 10, expected: np.ndarray | None = None, ax=None):
    """Spot price over the days: a few paths, the mean and the 5-95 % band (paper Fig. 3, top left).

    expected: optional (K,) analytical E[S_k], e.g. SeasonalOUSimulator.expected_spot().numpy().
    """
    ax = ax or plt.gca()
    S = market.spot.double().numpy()
    days = np.arange(market.n_days)
    lo, hi = np.percentile(S, [5, 95], axis=0)
    ax.fill_between(days, lo, hi, color="C0", alpha=0.15, label="5-95 % band")
    for i in range(min(n_show, market.n_paths)):
        ax.plot(days, S[i], lw=0.7, alpha=0.7)
    ax.plot(days, S.mean(0), color="k", lw=2, label="mean (simulation)")
    if expected is not None:
        ax.plot(days, expected, color="r", ls="--", lw=1.5, label="E[S_k] (analytical)")
    add_month_lines(ax, market)
    ax.set(xlabel="day k", ylabel="spot S_k [EUR/MWh]",
           title=f"Spot price, {market.n_paths} paths ({min(n_show, market.n_paths)} shown)")
    ax.legend(loc="upper left")
    return ax


def plot_spot_vs_forward(market: MarketData, path: int = 0, ax=None):
    """One path: spot and front-month forward (the forward is NaN, i.e. not drawn, in the last month)."""
    ax = ax or plt.gca()
    days = np.arange(market.n_days)
    ax.plot(days, market.spot[path].numpy(), lw=1, label="spot S_k")
    if market.fwd is not None:
        ax.plot(days, market.fwd[path].numpy(), lw=2, label="front-month forward F_k")
    add_month_lines(ax, market)
    ax.set(xlabel="day k", ylabel="EUR/MWh", title=f"Spot vs. forward (path {path})")
    ax.legend(loc="upper left")
    return ax


def plot_market(market: MarketData, n_show: int = 10, expected: np.ndarray | None = None,
                path: int = 0):
    """Overview figure of the data: spot paths (left) and spot vs. forward on one path (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    plot_paths(market, n_show=n_show, expected=expected, ax=axes[0])
    plot_spot_vs_forward(market, path=path, ax=axes[1])
    fig.tight_layout()
    return fig


def plot_levels(result: RolloutResult, spec: StorageSpec, ax=None):
    raise NotImplementedError


def plot_pnl_hist(pnls: dict[str, Tensor], ax=None):
    raise NotImplementedError


def plot_pnl_box(pnls: dict[str, Tensor], ax=None):
    raise NotImplementedError

#Need to add the thing about each experiment plots into its own folder with the config that made it
