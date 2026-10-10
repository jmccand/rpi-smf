"""Factor computation (IPS 4.1) and cross-sectional standardisation (IPS 4.2).

Factors are plain functions registered by name; `config/factors.yml` selects which
are used, their expected sign and parameters. Every function sees only data on or
before `ctx.asof` (prices are truncated by the caller as well as here).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from rpi_smf.config import FactorSpec
from rpi_smf.data.fundamentals import EST_COLUMNS, estimates_asof

REGISTRY: Dict[str, Callable[..., pd.Series]] = {}


def register(fn: Callable[..., pd.Series]) -> Callable[..., pd.Series]:
    REGISTRY[fn.__name__] = fn
    return fn


@dataclass
class Context:
    asof: pd.Timestamp
    prices: pd.DataFrame        # adjusted closes, columns = tickers
    estimates: pd.DataFrame     # long (date, ticker) estimates, may be empty
    market: Optional[pd.Series] = None   # market proxy price series (CAPM beta); None -> equal-weight universe

    def __post_init__(self):
        self.asof = pd.Timestamp(self.asof)
        self.prices = self.prices.loc[:self.asof]
        if self.market is not None:
            self.market = self.market.loc[:self.asof]
        if len(self.estimates):
            dates = self.estimates.index.get_level_values("date")
            self.estimates = self.estimates[dates <= self.asof]

    def price_on(self, when: pd.Timestamp) -> pd.Series:
        """Last available price on or before `when`."""
        p = self.prices.loc[:when]
        return p.ffill().iloc[-1] if len(p) else pd.Series(np.nan, index=self.prices.columns)

    def est_asof(self, when: pd.Timestamp) -> pd.DataFrame:
        if self.estimates.empty:
            return pd.DataFrame(columns=EST_COLUMNS, index=self.prices.columns, dtype=float)
        return estimates_asof(self.estimates, when).reindex(self.prices.columns)


def _positive(s: pd.Series) -> pd.Series:
    return s.where(s > 0)


@register
def forward_earnings_yield(ctx: Context) -> pd.Series:
    return ctx.est_asof(ctx.asof)["ntm_eps"] / ctx.price_on(ctx.asof)


@register
def revenue_growth_forecast(ctx: Context) -> pd.Series:
    return ctx.est_asof(ctx.asof)["ntm_revenue_growth"]


@register
def eps_growth_forecast(ctx: Context) -> pd.Series:
    return ctx.est_asof(ctx.asof)["ntm_eps_growth"]


@register
def momentum(ctx: Context, lookback_months: int = 12, skip_months: int = 1) -> pd.Series:
    """Total return from t-lookback to t-skip (skips the most recent month)."""
    end = ctx.price_on(ctx.asof - pd.DateOffset(months=skip_months))
    start = ctx.price_on(ctx.asof - pd.DateOffset(months=lookback_months))
    # a name must already be trading at the start of the window
    first = ctx.prices.apply(lambda c: c.first_valid_index())
    ok = first <= (ctx.asof - pd.DateOffset(months=lookback_months))
    return (end / start - 1).where(ok)


@register
def estimate_revisions(ctx: Context, months: int = 3) -> pd.Series:
    now = ctx.est_asof(ctx.asof)["ntm_eps"]
    then = _positive(ctx.est_asof(ctx.asof - pd.DateOffset(months=months))["ntm_eps"])
    return now / then - 1


@register
def volatility(ctx: Context, window_days: int = 252) -> pd.Series:
    r = ctx.prices.pct_change(fill_method=None).iloc[-window_days:]
    return r.std().where(r.count() >= window_days)


@register
def market_beta(ctx: Context, window_days: int = 252) -> pd.Series:
    """CAPM beta: cov(r_i, r_m) / var(r_m) over the trailing window of daily returns.

    The market is `ctx.market` (e.g. SPY); without it, the equal-weighted universe return.
    """
    r = ctx.prices.pct_change(fill_method=None).iloc[-window_days:]
    if ctx.market is not None:
        m = ctx.market.pct_change(fill_method=None).reindex(r.index)
    else:
        m = r.mean(axis=1)
    mc = m - m.mean()
    cov = r.sub(r.mean()).mul(mc, axis=0).sum() / (mc.notna().sum() - 1)
    return (cov / mc.var()).where((r.count() >= window_days) & m.notna().all())


def compute_factors(ctx: Context, specs: List[FactorSpec]) -> pd.DataFrame:
    """Raw factor values: index = tickers, columns = factor names."""
    cols = {}
    for s in specs:
        if s.function not in REGISTRY:
            raise KeyError(f"factor function '{s.function}' is not registered")
        cols[s.name] = REGISTRY[s.function](ctx, **s.params).reindex(ctx.prices.columns)
    return pd.DataFrame(cols)


def zscore(raw: pd.DataFrame, cap: float = 3.0) -> pd.DataFrame:
    """Cross-sectional z-score, capped at +/-cap; undefined values become 0."""
    mu, sd = raw.mean(), raw.std(ddof=1)
    z = (raw - mu) / sd.replace(0, np.nan)
    return z.clip(-cap, cap).fillna(0.0)
