"""Adjusted-close prices (yfinance) and the risk-free rate (FRED, ^IRX fallback).

Prices are cached per ticker as CSV under data/cache/prices and refreshed once per
day. Adjusted close reinvests dividends/distributions, so returns are total returns.
Functions never look past `asof`; callers pass the prior business day's date.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import pandas as pd

from rpi_smf.config import REPO_ROOT

log = logging.getLogger(__name__)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)   # we log missing tickers ourselves
CACHE_DIR = REPO_ROOT / "data" / "cache"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


def _cache_path(ticker: str, cache_dir: Path) -> Path:
    safe = ticker.replace("^", "_").replace("/", "_")
    return cache_dir / "prices" / f"{safe}.csv"


def _fresh(path: Path) -> bool:
    return path.exists() and date.fromtimestamp(path.stat().st_mtime) == date.today()


def _yf_download(tickers: List[str]) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(tickers, period="max", auto_adjust=True, progress=False,
                     group_by="column", threads=True)
    if df.empty:
        return pd.DataFrame()
    close = df["Close"]
    if isinstance(close, pd.Series):  # single ticker
        close = close.to_frame(tickers[0])
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close


def get_prices(tickers: Iterable[str], asof: Optional[pd.Timestamp] = None,
               refresh: bool = False, cache_dir: Optional[Path] = None,
               downloader: Optional[Callable[[List[str]], pd.DataFrame]] = None
               ) -> pd.DataFrame:
    """Adjusted closes, columns = tickers with any data, truncated at `asof` inclusive."""
    cache_dir = cache_dir or CACHE_DIR
    downloader = downloader or _yf_download
    tickers = list(dict.fromkeys(tickers))
    stale = [t for t in tickers if refresh or not _fresh(_cache_path(t, cache_dir))]
    if stale:
        try:
            got = downloader(stale)
        except Exception as e:  # network failure: fall back to whatever is cached
            log.warning("price download failed (%s); using stale cache where present", e)
            got = pd.DataFrame()
        for t in stale:
            path = _cache_path(t, cache_dir)
            s = got[t].dropna() if t in got.columns else pd.Series(dtype=float)
            if s.empty:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            s.rename("Close").to_frame().to_csv(path)
    out = {}
    for t in tickers:
        path = _cache_path(t, cache_dir)
        if path.exists():
            out[t] = pd.read_csv(path, index_col=0, parse_dates=True)["Close"]
        else:
            log.warning("no price data for %s", t)
    df = pd.DataFrame(out).sort_index()
    if asof is not None:
        df = df.loc[:pd.Timestamp(asof)]
    return df


def get_risk_free(asof: Optional[pd.Timestamp] = None, series: str = "DTB3",
                  refresh: bool = False, cache_dir: Optional[Path] = None,
                  reader: Optional[Callable[..., pd.DataFrame]] = None) -> pd.Series:
    """Annualised 3-month T-bill yield as a decimal (0.045 = 4.5%), daily, forward-filled."""
    cache_dir = cache_dir or CACHE_DIR
    reader = reader or pd.read_csv
    path = cache_dir / f"rf_{series}.csv"
    if refresh or not _fresh(path):
        try:
            df = reader(FRED_URL.format(series=series), index_col=0, parse_dates=True)
            s = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna() / 100.0
            path.parent.mkdir(parents=True, exist_ok=True)
            s.rename("rf").to_frame().to_csv(path)
        except Exception as e:
            log.warning("FRED %s unavailable (%s); trying ^IRX", series, e)
            if not path.exists():
                irx = get_prices(["^IRX"], refresh=refresh, cache_dir=cache_dir)
                if irx.empty:
                    raise RuntimeError("no risk-free rate source available")
                s = (irx["^IRX"] / 100.0).rename("rf")
                path.parent.mkdir(parents=True, exist_ok=True)
                s.to_frame().to_csv(path)
    rf = pd.read_csv(path, index_col=0, parse_dates=True)["rf"]
    if asof is not None:
        rf = rf.loc[:pd.Timestamp(asof)]
    return rf
