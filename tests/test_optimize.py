import numpy as np
import pandas as pd
import pytest

from rpi_smf.config import load_config
from rpi_smf.optimize import OptimizationError, optimize, shrunk_covariance

CFG = load_config().section("optimizer")


def _setup(n=40, seed=0, groups=4):
    rng = np.random.default_rng(seed)
    f = rng.normal(0, 0.01, (300, 1))
    rets = pd.DataFrame(f * rng.uniform(0.8, 1.5, n) + rng.normal(0, 0.012, (300, n)),
                        columns=[f"S{i}" for i in range(n)])
    mu = pd.Series(rng.normal(0.01, 0.01, n), index=rets.columns)       # monthly
    b = pd.Series(rng.dirichlet(np.ones(n) * 3), index=rets.columns)
    gics = pd.Series([f"G{i % groups}" for i in range(n)], index=rets.columns)
    bench_vol = float(np.sqrt(b @ shrunk_covariance(rets, 0.1, 252) @ b))
    return mu, rets, b, gics, bench_vol


def test_constraints_and_lambda_calibration():
    mu, rets, b, gics, bv = _setup()
    r = optimize(mu, rets, b, b, gics, bv, CFG)
    w = r.weights
    assert w.sum() == pytest.approx(1) and (w >= -1e-9).all()
    assert w.max() <= CFG["single_name_cap"] + 1e-6
    assert gics.groupby(gics).apply(lambda g: w[g.index].sum()).max() <= CFG["gics_cap"] + 1e-6
    assert r.tracking_error <= CFG["te_max"] + 1e-6
    assert r.vol <= CFG["vol_ratio_max"] * bv + 1e-6
    assert r.lambda_status == "calibrated"
    assert r.vol == pytest.approx(bv, rel=1e-3)           # 100% of benchmark vol
    assert r.sharpe == pytest.approx(r.expected_return / r.vol)


def test_turnover_penalty_keeps_close_to_prev_when_alpha_tiny():
    mu, rets, b, gics, bv = _setup(seed=2)
    r = optimize(mu * 1e-4, rets, b, b, gics, bv, CFG)
    assert r.turnover < 0.5      # tiny alpha: little incentive to move off benchmark-like weights


def test_infeasible_reports_min_te():
    mu, rets, b, gics, bv = _setup(seed=1)
    b = pd.Series(0.0, index=b.index); b.iloc[0] = 1.0     # benchmark = one name; cap 8% => TE huge
    bv = float(np.sqrt(b @ shrunk_covariance(rets, 0.1, 252) @ b))
    with pytest.raises(OptimizationError, match="minimum achievable tracking error"):
        optimize(mu, rets, b, b, gics, bv, CFG)


def test_missing_gics_raises():
    mu, rets, b, gics, bv = _setup()
    with pytest.raises(OptimizationError, match="GICS"):
        optimize(mu, rets, b, b, gics.drop("S0"), bv, CFG)


def test_min_positions_tightens_cap():
    mu, rets, b, gics, bv = _setup(n=60, seed=5)
    mu = mu * 3                                              # strong views -> concentrated solution
    plain = optimize(mu, rets, b, b, gics, bv, CFG)
    r = optimize(mu, rets, b, b, gics, bv, CFG, min_positions=30)
    assert r.n_positions >= 30
    assert r.cap_used <= CFG["single_name_cap"]
    assert r.weights.max() <= r.cap_used + 1e-6
    if plain.n_positions < 30:
        assert r.cap_used < CFG["single_name_cap"]
    assert r.vol <= CFG["vol_ratio_max"] * bv + 1e-6 and r.tracking_error <= CFG["te_max"] + 1e-6
