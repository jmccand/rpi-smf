import pandas as pd

from rpi_smf.data.fundamentals import (estimates_asof, load_estimates, save_snapshot,
                                       snapshot_yahoo)


def test_load_and_asof(tmp_path):
    (tmp_path / "bbg.csv").write_text(
        "date,ticker,ntm_eps,ntm_eps_growth\n"
        "2026-09-01,AAA,1.0,0.1\n2026-09-20,AAA,1.2,0.2\n2026-09-20,BBB,2.0,\n2026-10-01,AAA,9,9\n")
    est = load_estimates(tmp_path / "bbg.csv")
    assert "ntm_revenue_growth" in est.columns and est["ntm_revenue_growth"].isna().all()
    a = estimates_asof(est, "2026-09-27")
    assert a.loc["AAA", "ntm_eps"] == 1.2          # later 10-01 row is not visible
    assert pd.isna(a.loc["BBB", "ntm_eps_growth"])
    assert estimates_asof(est, "2026-09-27", max_age_days=3).empty


def test_snapshot_roundtrip(tmp_path):
    fake = lambda t: {"ntm_eps": 1.0, "ntm_revenue_growth": 0.1, "ntm_eps_growth": 0.2}
    snap = snapshot_yahoo(["X", "Y"], "2026-10-07", fetch=fake)
    path = save_snapshot(snap, tmp_path)
    assert path.name == "yahoo_2026-10-07.csv"
    assert len(load_estimates(tmp_path)) == 2


def test_proxy_snapshots_never_override_real_estimates(tmp_path):
    (tmp_path / "factset_2026-10-07.csv").write_text("date,ticker,ntm_eps\n2026-10-07,AAA,5.0\n")
    (tmp_path / "yahoo_2026-10-07.csv").write_text("date,ticker,ntm_eps\n2026-10-07,AAA,1.0\n")
    assert load_estimates(tmp_path).loc[(pd.Timestamp("2026-10-07"), "AAA"), "ntm_eps"] == 5.0
