"""Consensus estimates (point-in-time) for the estimate-based factors.

Primary source is a user-supplied Bloomberg (BEst) / FactSet export in long format:

    date,ticker,ntm_eps,ntm_revenue_growth,ntm_eps_growth
    2026-09-28,NVDA,6.12,0.41,0.55

`date` is the date the consensus was observed (point-in-time). Growth fields are
decimals (0.41 = 41%). Missing columns/values are allowed; the affected factors are
then undefined (z = 0) for those names/dates.

When no export exists for recent dates, `snapshot_yahoo` records today's Yahoo
consensus so history accumulates. Yahoo has no history, so these are PROXIES:
`ntm_eps` = Yahoo forwardEps (next fiscal-year EPS), growth fields = next-FY vs
current-FY consensus growth. Prefer the Bloomberg export where available.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd

log = logging.getLogger(__name__)
EST_COLUMNS = ["ntm_eps", "ntm_revenue_growth", "ntm_eps_growth"]


def load_estimates(path) -> pd.DataFrame:
    """Read one CSV or every *.csv in a directory; index (date, ticker), sorted."""
    p = Path(path)
    # later files win on duplicate (date, ticker): Yahoo proxy snapshots sort first (lowest priority)
    files = (sorted(p.glob("*.csv"), key=lambda f: (f.name.startswith("yahoo_") is False, f.name))
             if p.is_dir() else [p])
    if not files:
        return pd.DataFrame(columns=EST_COLUMNS,
                            index=pd.MultiIndex.from_arrays([[], []], names=["date", "ticker"]))
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = [c.strip().lower() for c in df.columns]
        missing = {"date", "ticker"} - set(df.columns)
        if missing:
            raise ValueError(f"{f}: missing columns {sorted(missing)}")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    for c in EST_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else float("nan")
    df = df.drop_duplicates(["date", "ticker"], keep="last")
    return df.set_index(["date", "ticker"])[EST_COLUMNS].sort_index()


def estimates_asof(est: pd.DataFrame, asof, max_age_days: Optional[int] = None) -> pd.DataFrame:
    """Latest observation per ticker on or before `asof` (index = ticker)."""
    asof = pd.Timestamp(asof)
    e = est[est.index.get_level_values("date") <= asof]
    if e.empty:
        return pd.DataFrame(columns=EST_COLUMNS)
    flat = e.reset_index().sort_values("date")
    latest = flat.groupby("ticker").tail(1)
    if max_age_days is not None:
        latest = latest[latest["date"] >= asof - pd.Timedelta(days=max_age_days)]
    return latest.set_index("ticker")[EST_COLUMNS]


def _yahoo_row(ticker: str) -> dict:
    import yfinance as yf
    t = yf.Ticker(ticker)
    row = {"ntm_eps": float("nan"), "ntm_revenue_growth": float("nan"),
           "ntm_eps_growth": float("nan")}
    try:
        row["ntm_eps"] = t.info.get("forwardEps", float("nan"))
        row["ntm_eps_growth"] = t.earnings_estimate.loc["+1y", "growth"]
        row["ntm_revenue_growth"] = t.revenue_estimate.loc["+1y", "growth"]
    except Exception as e:
        log.warning("yahoo estimates incomplete for %s: %s", ticker, e)
    return row


def snapshot_yahoo(tickers: Iterable[str], on_date,
                   fetch: Callable[[str], dict] = _yahoo_row) -> pd.DataFrame:
    """Long-format snapshot rows (date, ticker, estimate columns) for `on_date`."""
    rows = [{"date": pd.Timestamp(on_date), "ticker": t, **fetch(t)} for t in tickers]
    return pd.DataFrame(rows).set_index(["date", "ticker"])[EST_COLUMNS]


def save_snapshot(snapshot: pd.DataFrame, directory) -> Path:
    """Write the snapshot to <directory>/yahoo_<date>.csv (one file per snapshot date)."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    day = snapshot.index.get_level_values("date")[0].date().isoformat()
    path = d / f"yahoo_{day}.csv"
    snapshot.reset_index().to_csv(path, index=False)
    return path
