"""GICS classification, supplied by the user as gics.csv.

Required columns: `ticker` plus the level named by `ips.yml: optimizer.gics_level`
(e.g. `sub_industry`). Other columns (sector, industry_group, industry) are optional.
"""
from __future__ import annotations

import pandas as pd


def load_gics(path, level: str) -> pd.Series:
    """Series (index = ticker, name `gics`) holding each ticker's class at `level`."""
    # normalise header case so "Ticker"/"ticker" both work
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ("ticker", level):
        if col not in df.columns:
            raise ValueError(f"gics file {path} is missing column '{col}'")
    df["ticker"] = df["ticker"].str.strip()
    # a ticker with two classes would make the sector caps ambiguous
    if df["ticker"].duplicated().any():
        dups = df.loc[df["ticker"].duplicated(), "ticker"].tolist()
        raise ValueError(f"gics file has duplicate tickers: {dups}")
    return df.set_index("ticker")[level].str.strip().rename("gics")
