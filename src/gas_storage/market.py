"""Marktseite (Person A): Datencontainer MarketData + Preissimulatoren.

Konventionen (siehe notes/code_structure.md):
    - Preise in EUR/MWh, Tensoren torch.float32, Shape (M, K) = (Pfade, Tage)
    - K = 351 Tage, J = 12 Monate, month_start = [n_0=0, n_1, ..., n_J=K]
    - fwd[:, k] = Preis am Tag k des Kontrakts, der im NAECHSTEN Monat liefert
      (Front-Month); im letzten Monat gibt es keinen -> NaN
    - jede Zufallsfunktion nimmt ein `seed`-Argument
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Protocol

import torch
from torch import Tensor

from .config import DataConfig


# ---------------------------------------------------------------------------
# Kalender
# ---------------------------------------------------------------------------
def make_month_start(K: int = 351, n_months: int = 12) -> Tensor:
    """Teilt K Tage in n_months moeglichst gleich lange Monate.

    K=351, J=12 -> [0, 29, 58, 88, 117, 146, 176, 205, 234, 263, 292, 322, 351]
    """
    return torch.round(torch.linspace(0, K, n_months + 1)).long()


def month_of_day(month_start: Tensor) -> Tensor:
    """(K,) long: Monatsindex jedes Tages."""
    K = int(month_start[-1])
    return torch.bucketize(torch.arange(K), month_start[1:].cpu(), right=True)


# ---------------------------------------------------------------------------
# Datencontainer
# ---------------------------------------------------------------------------
@dataclass
class MarketData:
    spot: Tensor          # (M, K) Day-Ahead-Preis S_k
    fwd: Tensor | None    # (M, K) Front-Month-Forward F_k, NaN im letzten Monat
    month_start: Tensor   # (J+1,) long, n_0 = 0 < ... < n_J = K

    def __post_init__(self) -> None:
        assert self.spot.dim() == 2, "spot muss Shape (M, K) haben"
        K = self.spot.shape[1]
        assert int(self.month_start[0]) == 0 and int(self.month_start[-1]) == K, \
            "month_start muss bei 0 beginnen und bei K enden"
        assert bool(torch.all(self.month_start[1:] > self.month_start[:-1])), \
            "month_start muss streng steigend sein"
        if self.fwd is not None:
            assert self.fwd.shape == self.spot.shape, "fwd muss dieselbe Shape wie spot haben"

    @property
    def n_paths(self) -> int:
        return self.spot.shape[0]

    @property
    def n_days(self) -> int:
        return self.spot.shape[1]

    @property
    def n_months(self) -> int:
        return len(self.month_start) - 1

    def month_of_day(self) -> Tensor:
        return month_of_day(self.month_start)

    def days_in_month(self) -> Tensor:
        """(J,) long: Anzahl Tage je Monat."""
        return self.month_start[1:] - self.month_start[:-1]

    def fwd_mask(self) -> Tensor:
        """(K,) bool: True an Tagen, an denen ein Front-Month-Forward existiert."""
        return self.month_of_day() < self.n_months - 1

    def subset(self, idx: Tensor) -> "MarketData":
        return replace(self, spot=self.spot[idx],
                       fwd=None if self.fwd is None else self.fwd[idx])

    def to(self, device) -> "MarketData":
        return MarketData(self.spot.to(device),
                          None if self.fwd is None else self.fwd.to(device),
                          self.month_start.to(device))


def train_test_split(data: MarketData, n_train: int, seed: int) -> tuple[MarketData, MarketData]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(data.n_paths, generator=g)
    return data.subset(perm[:n_train]), data.subset(perm[n_train:])


class MarketSimulator(Protocol):
    def simulate(self, n_paths: int, seed: int) -> MarketData:
        """Return n_paths simulated price paths, reproducible with seed."""
        ...


def make_data(sim: MarketSimulator, cfg: DataConfig) -> tuple[MarketData, MarketData]:
    """Simulate cfg.n_paths paths and split them into (train, test) with cfg.n_train / cfg.n_test paths.

    Reproducibility:
        - Same DataConfig (same n_paths AND same seed) -> bit-identical tensors, across separate
          runs and independent of the global RNG state (torch.manual_seed, numpy, random), because
          every random function uses its own torch.Generator seeded with cfg.seed.
        - More paths means different paths: the first 1000 paths of a 10_000-path run are NOT the
          same as a 1000-path run, since torch draws large tensors in a different order.
        - Bit-identical numbers are only guaranteed within the same torch version and platform.
          PyTorch does not guarantee identical random streams across versions, so results from a
          different local torch version may differ slightly from the pinned environment
          (Colab 2026.01, torch 2.9.0). The pinned environment is the reference.
        - This covers the data only. Training must also be seeded (network initialization, batch
          shuffling); on the GPU additionally call torch.use_deterministic_algorithms(True).
    """
    data = sim.simulate(cfg.n_paths, seed=cfg.seed)
    return train_test_split(data, cfg.n_train, seed=cfg.seed)


# ---------------------------------------------------------------------------
# Saisonkurve (gemeinsam fuer Dummy und OU)
# ---------------------------------------------------------------------------
# Das Speicherjahr beginnt ~1. April: Einspeichern im Sommer (k <= 200),
# Ausspeichern im Winter. Preis-Maximum etwa Mitte Januar.
PEAK_DAY = 290


def seasonal_level(K: int, mean_level: float, amp: float, peak_day: int = PEAK_DAY) -> Tensor:
    """(K,) float64: saisonaler Preis in EUR/MWh (Kosinus mit Winterpeak)."""
    k = torch.arange(K, dtype=torch.float64)
    return mean_level + amp * torch.cos(2 * math.pi * (k - peak_day) / 365.0)


# ---------------------------------------------------------------------------
# Tag-1-Simulator: nur richtige Shapes
# ---------------------------------------------------------------------------
class DummySimulator:
    """Saisonkurve + unabhaengiges Rauschen. Nur zum Entsperren von Person B."""

    def __init__(self, K: int = 351, n_months: int = 12,
                 mean_level: float = 20.0, amp: float = 5.0, noise: float = 0.1):
        self.K = K
        self.month_start = make_month_start(K, n_months)
        self.curve = seasonal_level(K, mean_level, amp)
        self.noise = noise

    def simulate(self, n_paths: int, seed: int) -> MarketData:
        g = torch.Generator().manual_seed(seed)
        eps = torch.randn(n_paths, self.K, generator=g, dtype=torch.float64)
        spot = self.curve * torch.exp(self.noise * eps)

        # Forward = Mittel der Saisonkurve im naechsten Monat (deterministisch)
        months = month_of_day(self.month_start)
        J = len(self.month_start) - 1
        fwd = torch.full((n_paths, self.K), float("nan"), dtype=torch.float64)
        for j in range(J - 1):
            a, b = int(self.month_start[j + 1]), int(self.month_start[j + 2])
            fwd[:, months == j] = self.curve[a:b].mean()
        return MarketData(spot.float(), fwd.float(), self.month_start.clone())


# ---------------------------------------------------------------------------
# Echtes Modell: log S = saisonaler Mittelwert + Ornstein-Uhlenbeck
# ---------------------------------------------------------------------------
@dataclass
class SeasonalOUConfig:
    K: int = 351
    n_months: int = 12
    mean_level: float = 20.0       # EUR/MWh, mittleres Niveau
    seasonal_amp: float = 5.0      # Winter-Sommer-Amplitude
    mean_reversion: float = 0.05   # theta, pro Tag
    vol: float = 0.03              # taegliche Vola des Log-Preises
    fwd_premium: float = 0.0       # Risikopraemie p.a. im Forward (0 = Martingal, keine Drift)
    peak_day: int = PEAK_DAY


class SeasonalOUSimulator:
    """log S_k = m_k + X_k,  X_{k+1} = a X_k + sd Z,  a = exp(-theta)   (exakte OU-Diskretisierung)

    Forward:  F_k = (1/|T|) * sum_{t in naechster Monat} E[S_t | X_k] * exp(premium * tau_k / 365)
    mit tau_k = Tage bis Lieferbeginn. Mit premium = 0 ist F ein Martingal (keine Drift);
    mit premium != 0 laeuft F bis zum Lieferbeginn systematisch auf den erwarteten Spot zu
    (= Drift, fuer das Experiment "mit vs. ohne Drift").
    """

    def __init__(self, cfg: SeasonalOUConfig | None = None):
        self.cfg = cfg or SeasonalOUConfig()
        self.month_start = make_month_start(self.cfg.K, self.cfg.n_months)

    def seasonal_mean(self) -> Tensor:
        """(K,) float64: m_k = log der Saisonkurve."""
        c = self.cfg
        return torch.log(seasonal_level(c.K, c.mean_level, c.seasonal_amp, c.peak_day))

    def _cond_var(self, tau: Tensor) -> Tensor:
        """Var[X_{k+tau} | X_k] fuer den OU-Prozess."""
        th, v = self.cfg.mean_reversion, self.cfg.vol
        return v ** 2 * (1 - torch.exp(-2 * th * tau)) / (2 * th)

    def expected_spot(self) -> Tensor:
        """(K,) float64: E[S_k] aus Sicht von Tag 0 (X_0 = 0), geschlossene Form."""
        k = torch.arange(self.cfg.K, dtype=torch.float64)
        return torch.exp(self.seasonal_mean() + 0.5 * self._cond_var(k))

    def simulate_log_deviation(self, n_paths: int, seed: int) -> Tensor:
        """(M, K) float64: OU-Abweichung X, startet bei X_0 = 0 (heutiger Preis bekannt)."""
        c = self.cfg
        th = c.mean_reversion
        a = math.exp(-th)
        sd = c.vol * math.sqrt((1 - a ** 2) / (2 * th))
        g = torch.Generator().manual_seed(seed)
        Z = torch.randn(n_paths, c.K, generator=g, dtype=torch.float64)
        X = torch.zeros_like(Z)
        for k in range(1, c.K):
            X[:, k] = a * X[:, k - 1] + sd * Z[:, k]
        return X

    def forward_from_X(self, X: Tensor) -> Tensor:
        """(M, K) float64: Front-Month-Forward, NaN im letzten Monat."""
        c, th, ms = self.cfg, self.cfg.mean_reversion, self.month_start
        m = self.seasonal_mean()
        months = month_of_day(ms)
        J = len(ms) - 1
        F = torch.full_like(X, float("nan"))
        for k in range(c.K):
            j = int(months[k])
            if j == J - 1:                       # letzter Monat: kein Front-Month
                continue
            t = torch.arange(int(ms[j + 1]), int(ms[j + 2]))
            tau = (t - k).double()
            ES = torch.exp(m[t] + X[:, k:k + 1] * torch.exp(-th * tau) + 0.5 * self._cond_var(tau))
            premium = math.exp(c.fwd_premium * (int(ms[j + 1]) - k) / 365.0)
            F[:, k] = ES.mean(dim=1) * premium
        return F

    def simulate(self, n_paths: int, seed: int) -> MarketData:
        X = self.simulate_log_deviation(n_paths, seed)
        spot = torch.exp(self.seasonal_mean() + X)
        fwd = self.forward_from_X(X)
        return MarketData(spot.float(), fwd.float(), self.month_start.clone())


# ---------------------------------------------------------------------------
# Kalibrierung (optional): AR(1) auf den deseasonalisierten Log-Preisen
# ---------------------------------------------------------------------------
def fit_ou(log_dev: Tensor) -> tuple[float, float]:
    """Schaetzt (theta, vol) aus einer 1-D-Zeitreihe X der Log-Abweichungen (taeglich).

    Regression X_{k+1} = b X_k + eps  ->  theta = -log b,  vol = std(eps) * sqrt(2 theta / (1 - b^2))
    """
    x0, x1 = log_dev[:-1].double(), log_dev[1:].double()
    b = float((x0 * x1).sum() / (x0 * x0).sum())
    eps = x1 - b * x0
    theta = -math.log(b)
    vol = float(eps.std()) * math.sqrt(2 * theta / (1 - b ** 2))
    return theta, vol
