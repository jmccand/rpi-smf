import numpy as np
import pandas as pd
import pytest

from rpi_smf.config import FactorSpec, load_config
from rpi_smf.model import fit_model, month_ends, monthly_regression


def test_monthly_regression_recovers_coefficients():
    rng = np.random.default_rng(1)
    z = pd.DataFrame(rng.normal(size=(200, 2)), columns=["a", "b"])
    y = pd.Series(0.01 + 0.03 * z["a"] - 0.02 * z["b"] + rng.normal(0, 0.001, 200))
    alpha, beta = monthly_regression(z, y, 20)
    assert alpha == pytest.approx(0.01, abs=1e-3)
    assert beta["a"] == pytest.approx(0.03, abs=1e-3) and beta["b"] == pytest.approx(-0.02, abs=1e-3)
    assert monthly_regression(z.iloc[:5], y.iloc[:5], 20) is None


def test_month_ends_excludes_incomplete_month():
    idx = pd.bdate_range("2025-01-01", "2025-04-15")
    ends = month_ends(idx, pd.Timestamp("2025-04-15"))
    assert ends[-1] == pd.Timestamp("2025-03-31")
    ends2 = month_ends(idx, pd.Timestamp("2025-03-31"))   # asof is the last bday of March
    assert ends2[-1] == pd.Timestamp("2025-03-31")


def _panel(n_days=900, n=40, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n_days)
    vol = rng.uniform(0.01, 0.04, n)
    r = rng.normal(0.0004, 1, (n_days, n)) * vol
    return pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=idx,
                        columns=[f"S{i}" for i in range(n)])


SPECS = [FactorSpec("momentum", "momentum", 1, params={"lookback_months": 12, "skip_months": 1}),
         FactorSpec("volatility", "volatility", -1, params={"window_days": 252}),
         FactorSpec("market_beta", "market_beta", 1, params={"window_days": 252})]


def test_fit_model_end_to_end_and_no_lookahead():
    p = _panel()
    rf = pd.Series(0.04, index=p.index)
    cfg = load_config().section("model")
    asof = p.index[-60]
    mkt = (1 + p.pct_change().mean(axis=1).fillna(0)).cumprod()      # synthetic market series
    res = fit_model(p, rf, SPECS, asof, cfg, market=mkt)
    assert (res.months_used > 0).all()
    assert (res.beta * pd.Series({"momentum": 1, "volatility": -1, "market_beta": 1}) >= 0).all()   # sign rule
    assert len(res.history) <= cfg["regression_window_months"]
    assert res.expected_return.notna().all()
    # shrinkage identity
    phi = cfg["phi"]
    both = res.er_ma.notna()
    assert np.allclose(res.expected_return[both],
                       phi * res.er_ma[both] + (1 - phi) * res.er_factor[both])
    # corrupting data after asof changes nothing
    p2 = p.copy()
    p2.iloc[-59:] *= 5
    res2 = fit_model(p2, rf, SPECS, asof, cfg, market=mkt)
    pd.testing.assert_series_equal(res.expected_return, res2.expected_return)
