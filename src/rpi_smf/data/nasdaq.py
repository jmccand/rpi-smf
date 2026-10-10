"""Nasdaq-100 constituents, GICS classification and benchmark weights.

`NQ100.xlsx` (S&P Capital IQ screen) lists constituents and their classification
but no weights. `import_nq100` writes `data/raw/nq100_constituents.csv`
(ticker, weight, sector, industry_group, industry, sub_industry) and `data/raw/gics.csv`.
Weights are a PROXY: current market capitalisation (Yahoo), renormalised. The index
itself is modified-cap weighted, so these differ from the published NDX weights.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

log = logging.getLogger(__name__)
CLASS_COLS = {"IQ_SECTOR": "sector", "IQ_INDUSTRY_GROUP": "industry_group",
              "IQ_INDUSTRY": "industry", "IQ_PRIMARY_INDUSTRY": "sub_industry"}


def load_nq100_xlsx(path) -> pd.DataFrame:
    """Index = ticker; columns sector, industry_group, industry, sub_industry."""
    raw = pd.read_excel(path, sheet_name=0, header=None)
    hdr = raw.index[raw.iloc[:, 0] == "SP_ENTITY_NAME"]
    if not len(hdr):
        raise ValueError(f"{path}: no SP_ENTITY_NAME header row found")
    df = pd.read_excel(path, sheet_name=0, header=int(hdr[0])).dropna(subset=["SP_TICKER"])
    df["SP_TICKER"] = df["SP_TICKER"].astype(str).str.strip()
    if df["SP_TICKER"].duplicated().any():
        raise ValueError(f"{path}: duplicate tickers")
    return df.set_index("SP_TICKER")[list(CLASS_COLS)].rename(columns=CLASS_COLS).rename_axis("ticker")


def _yahoo_market_caps(tickers) -> pd.Series:
    import yfinance as yf
    out = {}
    for t in tickers:
        try:
            out[t] = float(yf.Ticker(t).fast_info["market_cap"])
        except Exception as e:  # noqa: BLE001 - missing cap just drops the name from the proxy
            log.warning("no market cap for %s (%s)", t, e)
    return pd.Series(out, dtype=float)


def import_nq100(xlsx, constituents_path, gics_path,
                 cap_fn: Optional[Callable[[list], pd.Series]] = None) -> pd.DataFrame:
    df = load_nq100_xlsx(xlsx)
    caps = (cap_fn or _yahoo_market_caps)(list(df.index)).reindex(df.index)
    if caps.isna().any():
        log.warning("no market cap, weight 0 (name stays investable): %s", list(caps.index[caps.isna()]))
    df.insert(0, "weight", caps.fillna(0.0) / caps.sum())
    Path(constituents_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(constituents_path)
    df[["sub_industry", "sector", "industry_group", "industry"]].to_csv(gics_path)
    return df


def load_constituents(path) -> pd.DataFrame:
    """Same shape as `holdings.listed_equities`: index ticker, column weight (fraction)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; run `rpi-smf import-nq100 --xlsx NQ100.xlsx`")
    return pd.read_csv(p, index_col="ticker")
