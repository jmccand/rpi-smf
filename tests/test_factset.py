from datetime import date

import pandas as pd
import pytest

from rpi_smf.data.factset import (consensus_to_estimates, fetch_consensus, load_factset_config,
                                  resolve_credentials_path)

CFG = load_factset_config()


def _fake_fetch(ids, metrics, periodicity, rel, start, end):
    rows = []
    for i in ids:
        for d, eps_scale in (("2026-09-18", 1.0), ("2026-09-25", 1.1)):
            for m in metrics:
                base = {"EPS": 4.0, "SALES": 100.0}[m] * eps_scale
                if periodicity == "NTMA" and rel == 1:
                    mean = base * 1.5
                elif periodicity == "LTMA" and rel == 0:
                    mean = base
                else:
                    continue
                rows.append({"request_id": i, "estimate_date": d, "metric": m,
                             "periodicity": periodicity, "relative_period": rel, "mean": mean})
    return rows


def test_fetch_and_transform():
    raw = fetch_consensus(["NVDA", "AAPL"], date(2026, 9, 1), date(2026, 9, 30), _fake_fetch, CFG)
    est = consensus_to_estimates(raw, CFG)
    assert list(est.columns) == ["ntm_eps", "ntm_revenue_growth", "ntm_eps_growth"]
    row = est.loc[(pd.Timestamp("2026-09-25"), "NVDA")]
    assert row["ntm_eps"] == pytest.approx(4.4 * 1.5)
    assert row["ntm_eps_growth"] == pytest.approx(0.5) and row["ntm_revenue_growth"] == pytest.approx(0.5)
    assert set(est.index.get_level_values("ticker")) == {"NVDA", "AAPL"}   # '-US' stripped


def test_non_positive_base_gives_nan_growth():
    def fetch(ids, metrics, per, rel, s, e):
        rows = _fake_fetch(ids, metrics, per, rel, s, e)
        for r in rows:
            if per == "LTMA" and r["metric"] == "EPS":
                r["mean"] = -1.0
        return rows
    est = consensus_to_estimates(fetch_consensus(["X"], date(2026, 9, 1), date(2026, 9, 30), fetch, CFG), CFG)
    assert est["ntm_eps_growth"].isna().all() and est["ntm_revenue_growth"].notna().all()


def test_batching_and_empty():
    calls = []
    def fetch(ids, *a):
        calls.append(len(ids)); return []
    cfg = dict(CFG, batch_size=2)
    raw = fetch_consensus(["A", "B", "C"], date(2026, 9, 1), date(2026, 9, 30), fetch, cfg)
    assert calls == [2, 2, 1, 1] and raw.empty
    assert consensus_to_estimates(raw, CFG).empty


def test_credentials_path_resolution(tmp_path, monkeypatch):
    monkeypatch.delenv("FACTSET_CONFIG", raising=False)
    with pytest.raises(ValueError):
        resolve_credentials_path(None)
    f = tmp_path / "app-config.json"; f.write_text("{}")
    assert resolve_credentials_path(str(f)) == str(f)
    monkeypatch.setenv("FACTSET_CONFIG", str(f))
    assert resolve_credentials_path(None) == str(f)
    with pytest.raises(FileNotFoundError):
        resolve_credentials_path(str(tmp_path / "missing.json"))


def test_key_file(tmp_path, monkeypatch):
    from rpi_smf.data.factset import read_key_file, resolve_key_file
    monkeypatch.delenv("FACTSET_KEY_FILE", raising=False)
    assert resolve_key_file(None) is None
    f = tmp_path / "k.txt"; f.write_text("user-123\nsecret\n")
    assert read_key_file(str(f)) == ("user-123", "secret")
    assert resolve_key_file(str(f)) == str(f)
    f.write_text("only-one-line\n")
    with pytest.raises(ValueError):
        read_key_file(str(f))
    with pytest.raises(FileNotFoundError):
        resolve_key_file(str(tmp_path / "nope.txt"))
