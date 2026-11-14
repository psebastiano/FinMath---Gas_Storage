"""Benchmarks: intrinsic LP (lower bound), perfect foresight (upper bound), LSMC (paper's benchmark)."""
import numpy as np

from .market import MarketData
from .storage import StorageSpec


def intrinsic_lp(price_curve: np.ndarray, spec: StorageSpec) -> tuple[np.ndarray, float]:
    """Deterministic plan on a fixed price curve (scipy.optimize.linprog)."""
    raise NotImplementedError


def perfect_foresight(market: MarketData, spec: StorageSpec) -> np.ndarray:
    """LP per path knowing the future: P&L per path (upper bound, reference only)."""
    raise NotImplementedError


def lsmc(market: MarketData, spec: StorageSpec, n_levels: int = 101,
         basis_degree: int = 3) -> np.ndarray:
    """Least-squares Monte Carlo: P&L per path."""
    raise NotImplementedError
