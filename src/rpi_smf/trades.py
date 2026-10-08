"""Trade generation (IPS 4 step 6, 4.3 last bullet).

A trade is recorded only when |current weight - optimizer target| exceeds the
threshold (1.5pp). Untouched positions keep their current weight, so the traded
names are scaled slightly (and re-capped) so the post-trade portfolio is still 100%
invested. Trading is simulated: fills are at the supplied prices, with a cost in bps.

The threshold exists to limit turnover; it must not leave the portfolio in breach of
a hard IPS limit. Positions above the single-name cap are always traded, and if the
post-trade portfolio is still non-compliant (`is_compliant` callback) the largest
remaining deviations are added one at a time until it passes. Such trades are tagged
reason="compliance" in the trade list.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd


@dataclass
class TradePlan:
    trades: pd.DataFrame          # one row per executed trade
    post_weights: pd.Series       # weights after the trades (sums to 1)
    residual_before_rescale: float
    total_cost: float             # dollars
    turnover: float               # sum |post - current|


def _post_weights(cur: pd.Series, tgt: pd.Series, traded: pd.Series, cap: float) -> pd.Series:
    """Traded names move to target; the budget residual is pushed into them (re-capped)."""
    post = cur.copy()
    post[traded] = tgt[traded]
    residual = 1.0 - post.sum()
    if traded.any() and abs(residual) > 1e-12:
        free = traded.copy()
        for _ in range(len(post)):
            room = (cap - post[free]).clip(lower=0) if residual > 0 else post[free]
            if room.sum() <= 1e-12:
                break
            add = residual * room / room.sum()
            add = np.minimum(add, room) if residual > 0 else np.maximum(add, -room)
            post[free] += add
            residual = 1.0 - post.sum()
            if abs(residual) < 1e-12:
                break
            free = free & (post < cap - 1e-12 if residual > 0 else post > 1e-12)
            if not free.any():
                break
    return post


def propose_trades(current: pd.Series, target: pd.Series, nav: float, prices: pd.Series,
                   cfg: dict, cap: float,
                   is_compliant: Optional[Callable[[pd.Series], bool]] = None) -> TradePlan:
    """`current`/`target` are weights of NAV (sum to 1); `cfg` is the ips.yml `trading` section."""
    tickers = sorted(set(current.index) | set(target.index))
    cur = current.reindex(tickers).fillna(0.0)
    tgt = target.reindex(tickers).fillna(0.0)
    diff = tgt - cur
    thresh = diff.abs() > cfg["threshold"] + 1e-9     # tolerance for float noise at the boundary
    reason = pd.Series(np.where(thresh, "threshold", ""), index=tickers)

    forced = (cur > cap + 1e-9) & (diff.abs() > 1e-12) & ~thresh
    traded = thresh | forced
    reason[forced] = "compliance"
    post = _post_weights(cur, tgt, traded, cap)
    if is_compliant is not None:
        while not is_compliant(post):
            rest = diff.abs().where(~traded & (diff.abs() > 1e-12)).dropna()
            if rest.empty:
                break
            t = rest.idxmax()
            traded[t] = True
            reason[t] = "compliance"
            post = _post_weights(cur, tgt, traded, cap)

    delta = post - cur
    moved = traded & (delta.abs() > 1e-12)
    px = prices.reindex(tickers)
    dollars = delta[moved] * nav
    trades = pd.DataFrame({
        "side": np.where(delta[moved] > 0, "BUY", "SELL"), "reason": reason[moved],
        "current_weight": cur[moved], "target_weight": tgt[moved], "post_weight": post[moved],
        "dollars": dollars, "price": px[moved], "shares": dollars / px[moved],
        "cost": dollars.abs() * cfg["cost_bps"] / 1e4,
    }).sort_values("dollars", key=lambda s: -s.abs())
    return TradePlan(trades, post, float(1.0 - (cur.where(~traded, tgt)).sum()),
                     float(trades["cost"].sum()), float(delta.abs().sum()))
