import numpy as np
import pandas as pd
import pytest

from rpi_smf.compliance import check_compliance
from rpi_smf.config import load_config
from rpi_smf.optimize import shrunk_covariance
from rpi_smf.trades import propose_trades

IPS = load_config().raw


def _data(n=40):
    rng = np.random.default_rng(0)
    rets = pd.DataFrame(rng.normal(0, 0.015, (300, n)), columns=[f"S{i}" for i in range(n)])
    gics = pd.Series([f"G{i % 5}" for i in range(n)], index=rets.columns)
    return rets, gics


def test_compliance_pass_and_fail():
    rets, gics = _data()
    eq = pd.Series(1 / 40, index=rets.columns)
    bv = float(np.sqrt(eq @ shrunk_covariance(rets, 0.1, 252) @ eq))
    ok = check_compliance(eq, gics, rets, eq, bv, IPS)
    assert ok.passed, ok.table
    bad = eq.copy(); bad.iloc[0] = 0.20; bad = bad / bad.sum()
    rep = check_compliance(bad, gics, rets, eq, bv, IPS)
    assert "single-company cap" in set(rep.failures["check"])
    concentrated = pd.Series(0.0, index=rets.columns)
    concentrated.iloc[[0, 5, 10, 15, 20, 25, 30, 35]] = 1 / 8     # one GICS group, 8 names
    f = check_compliance(concentrated, gics, rets, eq, bv, IPS).failures["check"].tolist()
    assert any("GICS" in c for c in f) and "minimum positions" in f


def test_trades_threshold_and_budget():
    cur = pd.Series({"A": 0.30, "B": 0.30, "C": 0.20, "D": 0.20})
    tgt = pd.Series({"A": 0.25, "B": 0.285, "C": 0.265, "D": 0.20})    # B within 1.5pp: no trade
    px = pd.Series({"A": 10.0, "B": 20.0, "C": 50.0, "D": 5.0})
    plan = propose_trades(cur, tgt, 1_000_000, px, {"threshold": 0.015, "cost_bps": 15}, cap=0.40)
    assert set(plan.trades.index) == {"A", "C"}                        # B and D untouched
    assert plan.post_weights.sum() == pytest.approx(1.0)
    assert plan.post_weights["B"] == 0.30 and plan.post_weights["D"] == 0.20
    assert plan.trades.loc["A", "shares"] == pytest.approx(-(0.30 - plan.post_weights["A"]) * 1e6 / 10)
    assert plan.total_cost == pytest.approx(plan.trades["dollars"].abs().sum() * 0.0015)
    assert plan.trades.loc["A", "side"] == "SELL"
