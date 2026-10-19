import numpy as np
import pandas as pd
import pytest

from rpi_smf.config import load_config
from rpi_smf.factors import Context, compute_factors, zscore


def _prices(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-01-01", periods=n)
    r = rng.normal(0.0005, 0.02, size=(n, 4))
    return pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=idx, columns=list("ABCD"))


def test_factor_values():
    p = _prices()
    mkt = (1 + p.pct_change().mean(axis=1).fillna(0)).cumprod()
    ctx = Context(p.index[-1], p, market=mkt)
    f = compute_factors(ctx, load_config().factors)
    exp_vol = p["A"].pct_change().iloc[-252:].std()
    assert f.loc["A", "volatility"] == pytest.approx(exp_vol)
    end = p["A"].loc[:p.index[-1] - pd.DateOffset(months=1)].iloc[-1]
    start = p["A"].loc[:p.index[-1] - pd.DateOffset(years=1)].iloc[-1]
    assert f.loc["A", "momentum"] == pytest.approx(end / start - 1)


def test_no_lookahead_and_young_names():
    p = _prices()
    p["E"] = p["A"].where(p.index > p.index[-100])                  # listed ~100 days ago
    asof = p.index[-50]
    ctx = Context(asof, p, market=p["A"])
    assert ctx.prices.index.max() == asof
    f = compute_factors(ctx, load_config().factors)
    assert np.isnan(f.loc["E", "momentum"]) and np.isnan(f.loc["E", "volatility"])


def test_zscore_cap_and_missing():
    raw = pd.DataFrame({"x": [1.0, 2, 3, 4, 100, np.nan], "c": [5.0] * 6})
    z = zscore(raw, cap=3)
    assert z["x"].max() <= 3 and z.loc[5, "x"] == 0
    assert (z["c"] == 0).all()                                      # zero variance -> 0
    assert abs(z["x"].iloc[:5].mean()) < 1.0


def test_market_beta():
    p = _prices()
    m = p["A"] * 0 + 100 * np.cumprod(1 + np.random.default_rng(9).normal(0, 0.01, len(p)))
    p["B"] = 100 * np.cumprod(1 + 2 * m.pct_change().fillna(0))        # exactly 2x market
    ctx = Context(p.index[-1], p, market=m)
    spec = [s for s in load_config().factors if s.name == "market_beta"]
    assert compute_factors(ctx, spec).loc["B", "market_beta"] == pytest.approx(2.0, abs=1e-6)


def test_market_beta_no_lookahead():
    p = _prices()
    m = p["A"]
    asof = p.index[-50]
    spec = [s for s in load_config().factors if s.name == "market_beta"]
    a = compute_factors(Context(asof, p, market=m), spec)
    q, mq = p.copy(), m.copy()
    q.loc[q.index > asof] *= 7.0
    mq.loc[mq.index > asof] *= 3.0
    b = compute_factors(Context(asof, q, market=mq), spec)
    pd.testing.assert_frame_equal(a, b)


def test_market_beta_requires_market():
    p = _prices()
    spec = [s for s in load_config().factors if s.name == "market_beta"]
    with pytest.raises(ValueError, match="market"):
        compute_factors(Context(p.index[-1], p), spec)
