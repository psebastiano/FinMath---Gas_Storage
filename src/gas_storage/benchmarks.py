"""Benchmarks: intrinsic LP (lower bound), perfect foresight (upper bound), LSMC (paper's benchmark).

Same storage problem as the rollout (storage.py / rollout.py), spot trading only:
    H_0 = 0,  H_{k+1} = H_k + q_k,  l_k <= q_k <= u_k,  0 <= H_k <= c,  H_K = 0
    cash on day k = -q_k S_k - kappa |q_k|          (q > 0 = buy/inject)
    P&L W = sum_k cash_k - fixed_cost, reported in units of pnl_scale (default: millions EUR)

All functions return P&L per path as np.ndarray (M,), directly comparable to RolloutResult.pnl.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, vstack

from .market import MarketData
from .storage import StorageSpec


def _np(x) -> np.ndarray:
    return x.detach().cpu().double().numpy() if hasattr(x, "detach") else np.asarray(x, dtype=float)


def _limits(spec: StorageSpec) -> tuple[np.ndarray, np.ndarray, float]:
    return _np(spec.inj_max), _np(spec.wdr_max), float(spec.capacity)


def max_level(spec: StorageSpec) -> np.ndarray:
    """(K+1,) highest level reachable on day k that can still be emptied by day K.

    Hmax_k = min(c, sum_{j<k} u_j, R_k)  with R_k = sum_{j>=k} |l_j|.
    """
    u, l, c = _limits(spec)
    K = len(u)
    reach = np.concatenate([[0.0], np.cumsum(u)])
    R = np.concatenate([np.cumsum(np.abs(l)[::-1])[::-1], [0.0]])
    return np.minimum(np.minimum(c, reach), R)


def plan_pnl(flow: np.ndarray, spot: np.ndarray, spec: StorageSpec, pnl_scale: float = 1e6) -> np.ndarray:
    """P&L of flows q (K,) or (M, K) on spot paths (M, K) -> (M,)."""
    flow = np.broadcast_to(flow, spot.shape)
    cash = -(flow * spot).sum(1) - spec.kappa * np.abs(flow).sum(1) - spec.fixed_cost
    return cash / pnl_scale


# ---------------------------------------------------------------------------
# LP: optimal plan for ONE known price curve
# ---------------------------------------------------------------------------
def _solve_lp(prices: np.ndarray, spec: StorageSpec) -> np.ndarray:
    """Optimal flow q (K,) for a known price curve.

    Variables: q = qp - qm, qp, qm >= 0 (so kappa |q| is linear).
    Level constraints 0 <= cumsum(q) <= c on days 1..K-1, sum(q) = 0.
    """
    u, l, c = _limits(spec)
    K = len(prices)
    Hmax = max_level(spec)
    cost = np.concatenate([prices + spec.kappa, -prices + spec.kappa])     # minimise -cash
    L = csr_matrix(np.tril(np.ones((K, K))))                                # H_{k+1} = (L q)_k
    A = hstack([L, -L]).tocsr()
    A_ub = vstack([A[:-1], -A[:-1]])
    b_ub = np.concatenate([Hmax[1:K], np.zeros(K - 1)])
    A_eq, b_eq = A[-1:], [0.0]
    bounds = [(0, ui) for ui in u] + [(0, -li) for li in l]
    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"LP failed: {res.message}")
    return res.x[:K] - res.x[K:]


def intrinsic_lp(price_curve: np.ndarray, spec: StorageSpec) -> tuple[np.ndarray, float]:
    """Deterministic plan on a fixed price curve (e.g. E[S_k] or the forward curve at day 0).

    Returns (flow (K,), value in EUR on that curve). Lower-bound benchmark: the plan is fixed on
    day 0 and never reacts to prices; evaluate it on paths with plan_pnl(flow, spot, spec).
    """
    prices = _np(price_curve)
    q = _solve_lp(prices, spec)
    value = float(-(q * prices).sum() - spec.kappa * np.abs(q).sum() - spec.fixed_cost)
    return q, value


def perfect_foresight(market: MarketData, spec: StorageSpec, pnl_scale: float = 1e6) -> np.ndarray:
    """LP per path knowing the whole future: P&L per path (upper bound, reference only)."""
    S = _np(market.spot)
    flows = np.stack([_solve_lp(S[m], spec) for m in range(S.shape[0])])
    return plan_pnl(flows, S, spec, pnl_scale)


# ---------------------------------------------------------------------------
# LSMC (Boogert & de Jong 2008 style)
# ---------------------------------------------------------------------------
@dataclass
class LSMCPolicy:
    """Fitted LSMC strategy: regression coefficients of the continuation value per day."""
    beta: list[np.ndarray]      # beta[k]: (basis_degree+1, n_levels) for C_{k+1}(H, S_k)
    s_mean: np.ndarray          # (K,) standardisation of S_k for the basis
    s_std: np.ndarray
    Hmax: np.ndarray            # (K+1,) grid upper end per day
    n_levels: int
    basis_degree: int


def _basis(s: np.ndarray, degree: int) -> np.ndarray:
    return np.stack([s ** d for d in range(degree + 1)], axis=-1)        # (M, degree+1)


def _interp(values: np.ndarray, H: np.ndarray, hmax: float) -> np.ndarray:
    """Linear interpolation on the uniform grid linspace(0, hmax, L) along axis 1.

    values (M, L), H (M, A) -> (M, A)."""
    L = values.shape[1]
    if hmax <= 0 or L == 1:
        return np.repeat(values[:, :1], H.shape[1], axis=1)
    x = np.clip(H / hmax * (L - 1), 0, L - 1)
    i = np.minimum(np.floor(x).astype(int), L - 2)
    w = x - i
    v0 = np.take_along_axis(values, i, axis=1)
    v1 = np.take_along_axis(values, i + 1, axis=1)
    return (1 - w) * v0 + w * v1


def _candidates(spec: StorageSpec, k: int, H: np.ndarray, Hmax_next: float) -> np.ndarray:
    """Bang-bang actions {max withdraw, idle, max inject}, clipped so that H' in [0, Hmax_{k+1}].

    H (..., ) -> q (..., 3). Bang-bang is optimal for linear payoffs (Boogert & de Jong)."""
    u, l, _ = _limits(spec)
    lo = np.maximum(l[k], -H)
    hi = np.minimum(u[k], Hmax_next - H)
    lo = np.minimum(lo, hi)                                    # (rounding safety)
    zero = np.clip(0.0, lo, hi)
    return np.stack([lo, zero, hi], axis=-1)


def _step_value(q, S_k, C_next, H, Hmax_next, kappa):
    """cash + continuation for candidate flows q (M, L, 3)."""
    M, L, A = q.shape
    Hn = (H[..., None] + q).reshape(M, L * A)
    cont = _interp(C_next, Hn, Hmax_next).reshape(M, L, A)
    return -q * S_k[:, None, None] - kappa * np.abs(q) + cont


def lsmc_fit(market: MarketData, spec: StorageSpec, n_levels: int = 101,
             basis_degree: int = 3) -> LSMCPolicy:
    """Backward induction on a grid of storage levels; continuation value regressed on S_k.

    V_K = 0 (H_K = 0 enforced by Hmax_K = 0). For k = K-1..0, on every grid level H:
        C_{k+1}(H', S_k) ~ basis(S_k) @ beta   (least squares of realised V_{k+1}(H') on basis(S_k))
        q* = argmax_q  -q S_k + C_{k+1}(H + q, S_k)    over bang-bang actions
        V_k(H) = -q* S_k + V_{k+1}(H + q*)            (realised values, Longstaff-Schwartz)
    """
    S = _np(market.spot)
    M, K = S.shape
    Hmax = max_level(spec)
    s_mean, s_std = S.mean(0), S.std(0) + 1e-12
    V = np.zeros((M, n_levels))                                # V_K on grid of day K (Hmax=0)
    beta: list[np.ndarray] = [None] * K
    for k in range(K - 1, -1, -1):
        X = _basis((S[:, k] - s_mean[k]) / s_std[k], basis_degree)
        b, *_ = np.linalg.lstsq(X, V, rcond=None)               # (deg+1, L)
        beta[k] = b
        C = X @ b                                               # (M, L) continuation at grid k+1
        H = np.broadcast_to(np.linspace(0, Hmax[k], n_levels), (M, n_levels))
        q = _candidates(spec, k, H, Hmax[k + 1])                # (M, L, 3)
        a = _step_value(q, S[:, k], C, H, Hmax[k + 1], spec.kappa).argmax(-1)
        qs = np.take_along_axis(q, a[..., None], -1)[..., 0]
        V = -qs * S[:, k:k + 1] - spec.kappa * np.abs(qs) + _interp(V, H + qs, Hmax[k + 1])
    return LSMCPolicy(beta, s_mean, s_std, Hmax, n_levels, basis_degree)


def lsmc_apply(policy: LSMCPolicy, market: MarketData, spec: StorageSpec,
               pnl_scale: float = 1e6) -> tuple[np.ndarray, np.ndarray]:
    """Run a fitted LSMC strategy forward on (new) paths. Returns (P&L (M,), flow (M, K))."""
    S = _np(market.spot)
    M, K = S.shape
    H = np.zeros(M)
    flow = np.zeros((M, K))
    for k in range(K):
        X = _basis((S[:, k] - policy.s_mean[k]) / policy.s_std[k], policy.basis_degree)
        C = X @ policy.beta[k]
        q = _candidates(spec, k, H[:, None], policy.Hmax[k + 1])      # (M, 1, 3)
        a = _step_value(q, S[:, k], C, H[:, None], policy.Hmax[k + 1], spec.kappa).argmax(-1)
        qs = np.take_along_axis(q, a[..., None], -1)[:, 0, 0]
        flow[:, k] = qs
        H = H + qs
    return plan_pnl(flow, S, spec, pnl_scale), flow


def lsmc(market: MarketData, spec: StorageSpec, n_levels: int = 101,
         basis_degree: int = 3, test: MarketData | None = None, pnl_scale: float = 1e6) -> np.ndarray:
    """Least-squares Monte Carlo: P&L per path.

    Fits on `market`; evaluates out-of-sample on `test` if given (recommended: the in-sample
    value is biased upwards because the regression has seen the future of the same paths).
    """
    policy = lsmc_fit(market, spec, n_levels, basis_degree)
    return lsmc_apply(policy, test if test is not None else market, spec, pnl_scale)[0]
