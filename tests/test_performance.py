import numpy as np
import pandas as pd
import pytest

from rpi_smf.config import load_config
from rpi_smf.performance import max_drawdown, performance_metrics

CFG = load_config().section("performance")


def test_drawdown_and_metrics():
    assert max_drawdown(pd.Series([1.0, 1.2, 0.9, 1.0])) == pytest.approx(-0.25)
    idx = pd.bdate_range("2026-01-01", periods=120)
    rng = np.random.default_rng(0)
    rb = rng.normal(0.0004, 0.01, 120)
    bench = pd.Series(100 * np.cumprod(1 + rb), index=idx)
    nav = pd.Series(1e6 * np.cumprod(1 + rb * 1.2), index=idx)
    rf = pd.Series(0.04, index=idx)
    m = performance_metrics(nav, bench, rf, CFG, 1e6)
    assert m["beta"] == pytest.approx(1.2, abs=1e-6)
    assert m["return_portfolio"] == pytest.approx(nav.iloc[-1] / nav.iloc[0] - 1)
    assert m["volatility_portfolio"] == pytest.approx(
        nav.pct_change().std() * np.sqrt(252))
    assert m["var_10d_99_usd"] == pytest.approx(m["var_10d_99_pct"] * 1e6)
    assert m["max_drawdown_portfolio"] <= 0



def test_fund_and_benchmark_statistics_and_no_lookahead():
    idx = pd.bdate_range("2026-01-01", periods=30)
    rng = np.random.default_rng(1)
    rb = rng.normal(0.0004, 0.01, 30)
    bench = pd.Series(100 * np.cumprod(1 + rb), index=idx)
    nav = pd.Series(1e6 * np.cumprod(1 + rb * 0.8 + 0.001), index=idx)
    rf = pd.Series(0.04, index=idx)
    m = performance_metrics(nav, bench, rf, CFG, 1e6)
    # short history: rolling Sharpe uses every return and reports its window
    assert m["sharpe_rolling_window_days"] == 29
    assert m["daily_vol_10d_benchmark"] == pytest.approx(bench.pct_change().iloc[-10:].std())
    assert m["daily_vol_since_inception_portfolio"] == pytest.approx(nav.pct_change().std())
    assert m["return_5d_benchmark"] == pytest.approx(bench.iloc[-1] / bench.iloc[-6] - 1)
    assert m["var_10d_95_pct_benchmark"] < m["var_10d_99_pct_benchmark"]
    # nothing after the as-of date may change the as-of statistics
    cut = idx[19]
    m2 = performance_metrics(nav.loc[:cut], bench.loc[:cut], rf, CFG, 1e6)
    m3 = performance_metrics(nav, bench, rf, CFG, 1e6)
    assert m2["as_of"] == cut.date().isoformat() and m3["as_of"] != m2["as_of"]
    m4 = performance_metrics(nav.loc[:cut], bench.loc[:cut], rf.loc[:cut], CFG, 1e6)
    assert m2 == m4
