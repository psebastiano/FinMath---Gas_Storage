"""Consistency tests for the benchmarks. Run: pytest tests/  (or: python tests/test_benchmarks.py)"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gas_storage.benchmarks import (intrinsic_lp, lsmc, lsmc_apply, lsmc_fit, max_level,
                                    perfect_foresight, plan_pnl)
from gas_storage.market import SeasonalOUConfig, SeasonalOUSimulator
from gas_storage.storage import StorageSpec

K = 90


def small_spec() -> StorageSpec:
    k = torch.arange(K)
    return StorageSpec(capacity=1000.0,
                       inj_max=torch.where(k < 50, 60.0, 10.0),
                       wdr_max=torch.where(k < 40, -10.0, -80.0),
                       kappa=0.0, fixed_cost=0.0)


def sim(vol: float) -> SeasonalOUSimulator:
    return SeasonalOUSimulator(SeasonalOUConfig(K=K, n_months=3, vol=vol, mean_reversion=0.02,
                                                peak_day=80))


def assert_feasible(flow: np.ndarray, spec: StorageSpec) -> None:
    flow = np.atleast_2d(flow)
    u, l = spec.inj_max.numpy(), spec.wdr_max.numpy()
    H = np.cumsum(flow, axis=1)
    tol = 1e-6
    assert (flow <= u + tol).all() and (flow >= l - tol).all()
    assert (H >= -tol).all() and (H <= spec.capacity + tol).all()
    assert np.abs(H[:, -1]).max() < 1e-6


def test_intrinsic_feasible_and_positive():
    spec = small_spec()
    q, value = intrinsic_lp(sim(0.03).expected_spot().numpy(), spec)
    assert_feasible(q, spec)
    assert value > 0


def test_max_level_ends_empty():
    Hmax = max_level(small_spec())
    assert Hmax[0] == 0 and Hmax[-1] == 0


def test_no_volatility_all_benchmarks_agree():
    spec = small_spec()
    s = sim(1e-9)
    data = s.simulate(50, seed=0)
    q, value = intrinsic_lp(data.spot[0].double().numpy(), spec)
    pf = perfect_foresight(data, spec, pnl_scale=1.0)
    ls = lsmc(data, spec, n_levels=201, pnl_scale=1.0)
    assert np.allclose(pf, value, rtol=1e-4)
    assert abs(ls.mean() - value) / value < 0.01


def test_ordering_intrinsic_lsmc_foresight():
    spec = small_spec()
    s = sim(0.04)
    train, test = s.simulate(2000, seed=0), s.simulate(500, seed=1)
    q, _ = intrinsic_lp(s.expected_spot().numpy(), spec)
    intr = plan_pnl(q, test.spot.double().numpy(), spec, 1.0)
    policy = lsmc_fit(train, spec)
    ls, flow = lsmc_apply(policy, test, spec, 1.0)
    pf = perfect_foresight(test, spec, pnl_scale=1.0)
    assert_feasible(flow, spec)
    assert (ls <= pf + 1e-6).all()                      # nothing beats knowing the future
    assert intr.mean() < ls.mean() < pf.mean()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
