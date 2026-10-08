"""Trade generation (IPS 4 step 6, 4.3 last bullet).

A trade is recorded only when |current weight - optimizer target| exceeds the
threshold (1.5pp). Untouched positions keep their current weight, so the traded
names are scaled slightly (and re-capped) so the post-trade portfolio is still 100%
invested. Trading is simulated: fills are at the supplied prices, with a cost in bps.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TradePlan:
    trades: pd.DataFrame          # one row per executed trade
    post_weights: pd.Series       # weights after the trades (sums to 1)
    residual_before_rescale: float
    total_cost: float             # dollars
    turnover: float               # sum |post - current|


def propose_trades(current: pd.Series, target: pd.Series, nav: float, prices: pd.Series,
                   cfg: dict, cap: float) -> TradePlan:
    """`current`/`target` are weights of NAV (sum to 1); `cfg` is the ips.yml `trading` section."""
    tickers = sorted(set(current.index) | set(target.index))
    cur = current.reindex(tickers).fillna(0.0)
    tgt = target.reindex(tickers).fillna(0.0)
    diff = tgt - cur
    traded = diff.abs() > cfg["threshold"] + 1e-9     # tolerance for float noise at the boundary

    post = cur.copy()
    post[traded] = tgt[traded]
    residual = 1.0 - post.sum()
    if traded.any() and abs(residual) > 1e-12:
        # push the residual into traded names (pro rata), respecting the cap
        free = traded.copy()
        for _ in range(len(tickers)):
            room = (cap - post[free]).clip(lower=0) if residual > 0 else post[free]
            if room.sum() <= 1e-12:
                break
            add = residual * room / room.sum()
            if residual > 0:
                add = np.minimum(add, room)
            else:
                add = np.maximum(add, -room)
            post[free] += add
            residual = 1.0 - post.sum()
            if abs(residual) < 1e-12:
                break
            free = free & (post < cap - 1e-12 if residual > 0 else post > 1e-12)
            if not free.any():
                break

    delta = post - cur
    moved = traded & (delta.abs() > 1e-12)
    px = prices.reindex(tickers)
    dollars = delta[moved] * nav
    trades = pd.DataFrame({
        "side": np.where(delta[moved] > 0, "BUY", "SELL"),
        "current_weight": cur[moved], "target_weight": tgt[moved], "post_weight": post[moved],
        "dollars": dollars, "price": px[moved], "shares": dollars / px[moved],
        "cost": dollars.abs() * cfg["cost_bps"] / 1e4,
    }).sort_values("dollars", key=lambda s: -s.abs())
    return TradePlan(trades, post, float(1.0 - (cur.where(~traded, tgt)).sum()),
                     float(trades["cost"].sum()), float(delta.abs().sum()))
