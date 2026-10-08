"""Parse a Fidelity fund-holdings CSV (text dumped from the PDF holdings report).

Each data row is one comma-free text field:
    <name> $<market value> <pct>% <ticker> [<cusip> <isin> <sedol>]
Market values in these dumps are frequently corrupted (dropped digits), so only the
"% of net assets" column is used as the weight.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yaml

_ROW = re.compile(
    r"^(?P<name>.+?)\s+\$(?P<mv>[\d,]+)\s+(?P<pct>[\d.]+)%\s+(?P<ticker>\S+)(?:\s+.*)?$")
_SECTIONS = {
    "domestic equities": "domestic_equity",
    "developed markets": "developed_markets",
    "emerging markets": "emerging_markets",
    "bonds": "bond",
    "cash and other securities": "cash",
}
_PRIVATE = re.compile(r"^[*%]|\+|\.PP$|\.PIPE$|\.CVR$|\.WT$|PP$")


def classify(ticker: str, section: str) -> str:
    """'listed' for tickers that can plausibly be priced, else why not."""
    if section == "bond":
        return "bond"
    if section == "cash":
        return "cash"
    if _PRIVATE.search(ticker):
        return "private"
    return "listed"


def parse_fidelity_csv(path) -> pd.DataFrame:
    """All parsed rows: name, ticker, section, kind, weight (fraction of fund)."""
    rows, section = [], None
    for line in Path(path).read_text().splitlines():
        text = line.strip().rstrip(",").strip().strip('"')
        key = text.lower()
        if key in _SECTIONS:
            section = _SECTIONS[key]
            continue
        m = _ROW.match(text)
        if not m or section is None:
            continue
        ticker = m["ticker"]
        rows.append({
            "name": m["name"], "ticker": ticker, "section": section,
            "kind": classify(ticker, section),
            "weight": float(m["pct"]) / 100.0,
        })
    if not rows:
        raise ValueError(f"no holdings rows parsed from {path}")
    df = pd.DataFrame(rows)
    if df["ticker"].duplicated().any():
        df = df.groupby(["ticker", "section", "kind"], as_index=False).agg(
            name=("name", "first"), weight=("weight", "sum"))
    return df[["ticker", "name", "section", "kind", "weight"]]


def load_universe_config(path) -> Dict:
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    return {"ticker_overrides": cfg.get("ticker_overrides") or {},
            "exclude": set(cfg.get("exclude") or [])}


def listed_equities(holdings: pd.DataFrame, overrides: Optional[Dict[str, str]] = None,
                    exclude=()) -> pd.DataFrame:
    """Listed equity rows with Yahoo tickers; `fidelity_ticker` keeps the source symbol.

    Weights are left as fractions of the whole fund (not renormalised); callers
    renormalise after price-availability filtering.
    """
    overrides = overrides or {}
    h = holdings[holdings["kind"] == "listed"].copy()
    h = h[~h["ticker"].isin(set(exclude))]
    h["fidelity_ticker"] = h["ticker"]
    h["ticker"] = h["ticker"].map(lambda t: overrides.get(t, t))
    return h.set_index("ticker")
