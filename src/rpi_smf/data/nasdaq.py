"""Nasdaq-100 constituents, GICS classification and benchmark weights.

`NQ100.xlsx` (S&P Capital IQ screen) lists constituents and their classification
but no weights. `import_nq100` writes `data/raw/nq100_constituents.csv`
(ticker, weight, sector, industry_group, industry, sub_industry) and `data/raw/gics.csv`.
Weights come from the issuer's holdings file (Invesco "Complete Holdings" CSV for
QQQM, `% TNA` column) when one is supplied: the universe is then restricted to the names the
fund holds at that date and share classes are merged (GOOG -> GOOGL). Without a holdings file
they are a PROXY: current market capitalisation (Yahoo), renormalised, which differs from
the published (modified-cap) NDX weights.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

log = logging.getLogger(__name__)
CLASS_COLS = {"IQ_SECTOR": "sector", "IQ_INDUSTRY_GROUP": "industry_group",
              "IQ_INDUSTRY": "industry", "IQ_PRIMARY_INDUSTRY": "sub_industry"}


def load_nq100_xlsx(path) -> pd.DataFrame:
    """Index = ticker; columns sector, industry_group, industry, sub_industry."""
    # Capital IQ exports have a preamble; find the real header row first
    raw = pd.read_excel(path, sheet_name=0, header=None)
    hdr = raw.index[raw.iloc[:, 0] == "SP_ENTITY_NAME"]
    if not len(hdr):
        raise ValueError(f"{path}: no SP_ENTITY_NAME header row found")
    # re-read with the right header and drop rows without a ticker
    df = pd.read_excel(path, sheet_name=0, header=int(hdr[0])).dropna(subset=["SP_TICKER"])
    df["SP_TICKER"] = df["SP_TICKER"].astype(str).str.strip()
    if df["SP_TICKER"].duplicated().any():
        raise ValueError(f"{path}: duplicate tickers")
    # keep only the classification columns, renamed to our level names
    return df.set_index("SP_TICKER")[list(CLASS_COLS)].rename(columns=CLASS_COLS).rename_axis("ticker")


def _yahoo_market_caps(tickers) -> pd.Series:
    """Current market cap per ticker from Yahoo; names without one are omitted."""
    # imported lazily so tests never touch the network
    import yfinance as yf
    out = {}
    for t in tickers:
        # one request per ticker; a failure must not abort the whole import
        try:
            out[t] = float(yf.Ticker(t).fast_info["market_cap"])
        except Exception as e:  # noqa: BLE001 - missing cap just drops the name from the proxy
            log.warning("no market cap for %s (%s)", t, e)
    return pd.Series(out, dtype=float)


def load_holdings(path, merge: Optional[dict] = None):
    """Invesco holdings CSV -> (weights Series by ticker summing to 1, as-of date or None).

    Keeps `Common Stock` rows only (drops cash, collateral, futures), parses `% TNA`, and adds
    each share class in `merge` ({"GOOG": "GOOGL"}) to its target ticker.
    """
    # the as-of date is only in the free-text preamble, so search the raw text
    text = Path(path).read_text(encoding="utf-8-sig")
    m = re.search(r"as of (\d{4}-\d{2}-\d{2})", text)
    df = pd.read_csv(path, encoding="utf-8-sig", comment="#", skip_blank_lines=True, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    for col in ("Ticker", "% TNA", "Class of shares"):
        if col not in df.columns:
            raise ValueError(f"{path}: missing column '{col}'")
    # drop non-equity rows, then parse "12.3%" strings into floats indexed by (merged) ticker
    df = df[df["Class of shares"].str.strip() == "Common Stock"]
    w = pd.Series(df["% TNA"].str.rstrip("%").astype(float).to_numpy(),
                  index=df["Ticker"].str.strip().replace(merge or {}), name="weight")
    # merged share classes collapse into one row; renormalise so weights sum to 1
    w = w.groupby(level=0).sum()
    return (w / w.sum()).rename_axis("ticker"), (m.group(1) if m else None)


def import_nq100(xlsx, constituents_path, gics_path,
                 cap_fn: Optional[Callable[[list], pd.Series]] = None,
                 holdings=None, merge: Optional[dict] = None) -> pd.DataFrame:
    """Write constituents (with weights) and GICS files; GICS classes come from the xlsx.

    With `holdings` the universe is the fund's holdings and weights are its `% TNA`; every
    held name must be classified in the xlsx. Otherwise the whole xlsx list is used with
    market-cap proxy weights.
    """
    df = load_nq100_xlsx(xlsx)
    # provenance recorded next to the CSV so runs can report where weights came from
    meta = {"source": "market-cap proxy (Yahoo)", "asof": None}
    if holdings is not None:
        # fund-holdings path: the held names define the universe
        w, asof = load_holdings(holdings, merge)
        unclassified = sorted(set(w.index) - set(df.index))
        if unclassified:
            raise ValueError(f"held names missing from {xlsx} (refresh the export): {unclassified}")
        dropped = sorted(set(df.index) - set(w.index))
        if dropped:
            log.warning("not in the fund's holdings, dropped from the universe: %s", dropped)
        # restrict and order the classification table to the held names
        df = df.loc[w.index]
        df.insert(0, "weight", w)
        meta = {"source": f"fund holdings file {Path(holdings).name} (% TNA)", "asof": asof}
    else:
        # fallback path: market-cap proxy weights (names with no cap get weight 0)
        caps = (cap_fn or _yahoo_market_caps)(list(df.index)).reindex(df.index)
        if caps.isna().any():
            log.warning("no market cap, weight 0 (name stays investable): %s", list(caps.index[caps.isna()]))
        df.insert(0, "weight", caps.fillna(0.0) / caps.sum())
    # write constituents+weights, the GICS file and the provenance sidecar
    Path(constituents_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(constituents_path)
    df[["sub_industry", "sector", "industry_group", "industry"]].to_csv(gics_path)
    meta_path(constituents_path).write_text(json.dumps(meta))
    return df


def meta_path(constituents_path) -> Path:
    """Path of the provenance sidecar (`.meta.json`) for a constituents CSV."""
    return Path(constituents_path).with_suffix(".meta.json")


def load_meta(constituents_path) -> dict:
    """Weight-source metadata written at import time; {} if there is no sidecar."""
    p = meta_path(constituents_path)
    return json.loads(p.read_text()) if p.exists() else {}


def load_constituents(path) -> pd.DataFrame:
    """Index ticker; column `weight` (fraction of the index)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; run `rpi-smf import-nq100 --xlsx NQ100.xlsx`")
    return pd.read_csv(p, index_col="ticker")
