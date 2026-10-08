import numpy as np
import pandas as pd
import pytest

from rpi_smf.config import load_config
from rpi_smf.data.fundamentals import EST_COLUMNS
from rpi_smf.factors import Context, compute_factors, zscore


def _prices(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-01-01", periods=n)
    r = rng.normal(0.0005, 0.02, size=(n, 4))
    return pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=idx, columns=list("ABCD"))


def _est(prices):
    d1, d2 = prices.index[-1], prices.index[-1] - pd.DateOffset(months=3)
    rows = []
    for d, eps in ((d2, 1.0), (d1, 1.5)):
        for t in "ABC":
            rows.append({"date": d, "ticker": t, "ntm_eps": eps, "ntm_revenue_growth": 0.1,
                         "ntm_eps_growth": 0.2})
    return pd.DataFrame(rows).set_index(["date", "ticker"])[EST_COLUMNS]


def test_factor_values():
    p = _prices()
    ctx = Context(p.index[-1], p, _est(p))
    f = compute_factors(ctx, load_config().factors)
    assert f.loc["A", "forward_earnings_yield"] == pytest.approx(1.5 / p["A"].iloc[-1])
    assert f.loc["A", "estimate_revisions"] == pytest.approx(0.5)
    assert np.isnan(f.loc["D", "estimate_revisions"])               # no estimates for D
    exp_vol = p["A"].pct_change().iloc[-252:].std()
    assert f.loc["A", "volatility"] == pytest.approx(exp_vol)
    end = p["A"].loc[:p.index[-1] - pd.DateOffset(months=1)].iloc[-1]
    start = p["A"].loc[:p.index[-1] - pd.DateOffset(years=1)].iloc[-1]
    assert f.loc["A", "momentum"] == pytest.approx(end / start - 1)


def test_no_lookahead_and_young_names():
    p = _prices()
    p["E"] = p["A"].where(p.index > p.index[-100])                  # listed ~100 days ago
    asof = p.index[-50]
    ctx = Context(asof, p, _est(p))
    assert ctx.prices.index.max() == asof
    assert ctx.estimates.empty or ctx.estimates.index.get_level_values("date").max() <= asof
    f = compute_factors(ctx, load_config().factors)
    assert np.isnan(f.loc["E", "momentum"]) and np.isnan(f.loc["E", "volatility"])


def test_zscore_cap_and_missing():
    raw = pd.DataFrame({"x": [1.0, 2, 3, 4, 100, np.nan], "c": [5.0] * 6})
    z = zscore(raw, cap=3)
    assert z["x"].max() <= 3 and z.loc[5, "x"] == 0
    assert (z["c"] == 0).all()                                      # zero variance -> 0
    assert abs(z["x"].iloc[:5].mean()) < 1.0
