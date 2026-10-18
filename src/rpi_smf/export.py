"""Export the post-trade portfolio as a broker basket file (``rebalance_post_trade wk1.csv`` layout).

One headerless row per name held before or after the rebalance, sorted by ticker:
``DES,<ticker>,STK,SMART/AMEX,,,,,,<post-trade weight in percent>``. Names sold to zero are
kept (weight 0) so the basket closes them out.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

EPS = 1e-9


def post_trade_basket(weights: pd.DataFrame) -> pd.DataFrame:
    """`weights` is the run's weights table (index = ticker, columns incl. current, post_trade)."""
    w = weights[(weights["current"] > EPS) | (weights["post_trade"] > EPS)]
    w = w.sort_index()
    rows = pd.DataFrame({
        0: "DES", 1: w.index, 2: "STK", 3: "SMART/AMEX",
        4: "", 5: "", 6: "", 7: "", 8: "",
        9: (w["post_trade"].clip(lower=0) * 100).round(6).to_numpy(),
    })
    return rows


def export_post_trade(run_dir, out_path) -> Path:
    weights = pd.read_csv(Path(run_dir) / "weights.csv", index_col=0)
    out = Path(out_path)
    post_trade_basket(weights).to_csv(out, header=False, index=False, float_format="%.6f")
    return out
