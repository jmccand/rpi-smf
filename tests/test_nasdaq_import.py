import pandas as pd
import pytest

from rpi_smf.data import nasdaq

HOLD = '''﻿Ticker,Company,Share/ Par,% TNA,Class of shares,CUSIP,Market value
"AAA","A Corp","1","60.00%","Common Stock","x","$1"
"GOOGL","Alpha","1","20.00%","Common Stock","x","$1"
"GOOG","Alpha","1","10.00%","Common Stock","x","$1"
"--","CASH","1","0.01%","Currency Collateral","USDMTM","$1"
"--","FUT","-1","-0.12%","Synthetic Cash","NQZ6","-$1"
"USDPDV","Pending","0","--","Currency","--","$1"

# as of 2026-10-16'''


def _xlsx_df(tickers):
    return pd.DataFrame({"sector": "S", "industry_group": "G", "industry": "I", "sub_industry": "SI"},
                        index=pd.Index(tickers, name="ticker"))


def test_holdings_parse_and_merge(tmp_path):
    f = tmp_path / "h.csv"
    f.write_text(HOLD, encoding="utf-8")
    w, asof = nasdaq.load_holdings(f, {"GOOG": "GOOGL"})
    assert asof == "2026-10-16"
    assert list(w.index) == ["AAA", "GOOGL"] or set(w.index) == {"AAA", "GOOGL"}
    assert w["AAA"] == pytest.approx(60 / 90) and w["GOOGL"] == pytest.approx(30 / 90)


def test_import_restricts_universe_to_holdings(tmp_path, monkeypatch):
    f = tmp_path / "h.csv"
    f.write_text(HOLD, encoding="utf-8")
    monkeypatch.setattr(nasdaq, "load_nq100_xlsx", lambda p: _xlsx_df(["AAA", "GOOGL", "ZZZ"]))
    out = nasdaq.import_nq100("x.xlsx", tmp_path / "c.csv", tmp_path / "g.csv",
                              holdings=f, merge={"GOOG": "GOOGL"})
    assert set(out.index) == {"AAA", "GOOGL"}          # ZZZ is not held -> dropped
    assert out["weight"].sum() == pytest.approx(1.0)
    assert nasdaq.load_meta(tmp_path / "c.csv")["asof"] == "2026-10-16"


def test_import_requires_classification(tmp_path, monkeypatch):
    f = tmp_path / "h.csv"
    f.write_text(HOLD, encoding="utf-8")
    monkeypatch.setattr(nasdaq, "load_nq100_xlsx", lambda p: _xlsx_df(["AAA"]))
    with pytest.raises(ValueError, match="GOOGL"):
        nasdaq.import_nq100("x.xlsx", tmp_path / "c.csv", tmp_path / "g.csv",
                            holdings=f, merge={"GOOG": "GOOGL"})


def test_closest_holdings_by_asof_date(tmp_path):
    for name, d in (("invesco_a-Complete_Holdings.csv", "2026-10-02"),
                    ("invesco_b-Complete_Holdings.csv", "2026-10-16")):
        (tmp_path / name).write_text(HOLD.replace("2026-10-16", d), encoding="utf-8")
    assert nasdaq.closest_holdings(tmp_path, "2026-10-05").name == "invesco_a-Complete_Holdings.csv"
    assert nasdaq.closest_holdings(tmp_path, "2026-10-18").name == "invesco_b-Complete_Holdings.csv"
    with pytest.raises(FileNotFoundError):
        nasdaq.closest_holdings(tmp_path / "none", "2026-10-18")
