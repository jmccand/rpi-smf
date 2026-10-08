import numpy as np
import pandas as pd
import pytest

from rpi_smf.config import load_config
from rpi_smf.performance import (brinson_period, link_attribution, max_drawdown,
                                 performance_metrics)

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


def test_brinson_sums_to_active_and_links():
    groups = pd.Series({"A": "semi", "B": "semi", "C": "soft", "D": "soft"})
    wb = pd.Series({"A": .3, "B": .2, "C": .3, "D": .2})
    r1 = pd.Series({"A": .05, "B": -.01, "C": .02, "D": .03})
    wp1 = pd.Series({"A": .4, "B": .1, "C": .25, "D": .25})
    r2 = pd.Series({"A": -.02, "B": .04, "C": .01, "D": -.03})
    wp2 = pd.Series({"A": .2, "B": .3, "C": .3, "D": .2})
    effs, rps, rbs = [], [], []
    for wp, r in ((wp1, r1), (wp2, r2)):
        e = brinson_period(wp, wb, r, groups)
        assert e.to_numpy().sum() == pytest.approx((wp * r).sum() - (wb * r).sum())
        effs.append(e); rps.append((wp * r).sum()); rbs.append((wb * r).sum())
    linked = link_attribution(effs, rps, rbs)
    cum_active = np.prod(1 + np.array(rps)) - np.prod(1 + np.array(rbs))
    assert linked.to_numpy().sum() == pytest.approx(cum_active)
