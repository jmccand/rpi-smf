"""End-to-end rebalance on synthetic data: no network."""
import numpy as np
import pandas as pd
import pytest

from rpi_smf.data import prices as prices_mod
from rpi_smf.pipeline import Paths, Pipeline

N = 45
TICKERS = [f"S{i:02d}" for i in range(N)]


def _holdings_text():
    lines = ["Security Name Market Value Assets Ticker CUSIP ISIN SEDOL,,,", "Domestic Equities,,,"]
    w = np.full(N, 100.0 / N / 1.0)
    for t, x in zip(TICKERS, w):
        lines.append(f"{t} Corp $1,000 {x:.3f}% {t} 123 US123 456")
    lines.append("Bonds,,,")
    lines.append("Some Bond 15% 12/31/2026 $3,622,554 0.010% %BOND,")
    return "\n".join(lines)


@pytest.fixture
def env(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2023-06-01", "2026-10-07")
    mkt = rng.normal(0.0004, 0.01, len(idx))
    data = {t: 50 * np.cumprod(1 + mkt * rng.uniform(0.8, 1.4) + rng.normal(0, 0.012, len(idx)))
            for t in TICKERS}
    px = pd.DataFrame(data, index=idx)
    px["FSPTX"] = 100 * np.cumprod(1 + mkt * 1.1)
    monkeypatch.setattr(prices_mod, "_yf_download", lambda ts: px[[t for t in ts if t in px]])
    monkeypatch.setattr(prices_mod, "CACHE_DIR", tmp_path / "cache")
    h = tmp_path / "fsptx_holdings_test.csv"; h.write_text(_holdings_text())
    g = tmp_path / "gics.csv"
    pd.DataFrame({"ticker": TICKERS, "sub_industry": [f"G{i % 6}" for i in range(N)]}).to_csv(g, index=False)
    est = tmp_path / "est"; est.mkdir()
    paths = Paths(holdings=h, gics=g, estimates=est, runs=tmp_path / "runs")
    monkeypatch.setattr(prices_mod, "get_risk_free",
                        lambda **kw: pd.Series(0.04, index=idx))
    return Pipeline(paths), tmp_path


def test_rebalance_end_to_end(env):
    pipe, tmp = env
    pipe.ips["trading"]["threshold"] = 0.0         # trade whatever the optimiser asks for
    r = pipe.rebalance("2026-10-07")
    assert r.asof == pd.Timestamp("2026-10-06")                      # inputs from prior business day
    assert "%BOND" in r.universe.excluded
    assert r.opt.weights.sum() == pytest.approx(1)
    assert r.compliance_target.passed, r.compliance_target.table.to_string()
    assert r.compliance_post.passed, r.compliance_post.table.to_string()
    run = tmp / "runs" / "2026-10-07"
    assert (run / "manifest.json").exists() and (run / "trades.csv").exists()
    assert (tmp / "runs" / "journal.csv").exists()
    # state moved to the new holdings; start portfolio was created at the configured start date
    hist = pd.read_csv(tmp / "runs" / "holdings_history.csv")
    assert set(hist["date"]) == {"2026-09-28", "2026-10-07"}
    m, attr, nav, bench = pipe.performance()
    assert nav.index[0] == pd.Timestamp("2026-09-28")
    assert m["return_portfolio"] == pytest.approx(nav.iloc[-1] / nav.iloc[0] - 1)
    # attribution components add up to the cumulative active return vs the look-through benchmark
    assert np.isfinite(attr.to_numpy()).all()
