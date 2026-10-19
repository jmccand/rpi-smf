"""Performance metrics (IPS Section 7).

Daily returns throughout; the same frequency and the 3-month T-bill are used for the
portfolio and the benchmark. Volatility and Sharpe are annualised with
sqrt(trading_days). Metrics carried over from the original fund-metrics script (beta,
rolling Sharpe, parametric VaR, 5-day return) are kept.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import norm


def max_drawdown(nav: pd.Series) -> float:
    """Largest peak-to-trough decline (a negative fraction)."""
    return float((nav / nav.cummax() - 1).min())


def sharpe(r: pd.Series, rf_daily: pd.Series, td: int) -> float:
    """Annualised Sharpe ratio of daily returns `r` over the daily risk-free rate."""
    ex = r - rf_daily.reindex(r.index).ffill().fillna(0.0)
    sd = ex.std()
    return float(ex.mean() / sd * np.sqrt(td)) if sd > 0 else float("nan")


def performance_metrics(nav: pd.Series, bench: pd.Series, rf_annual: pd.Series, cfg: dict,
                        amount: float) -> Dict[str, float]:
    """`nav`/`bench` are levels from inception; they are aligned on common dates."""
    # compare only on dates where both series exist
    idx = nav.index.intersection(bench.index)
    nav, bench = nav.loc[idx], bench.loc[idx]
    if len(idx) < 3:
        raise ValueError("need at least 3 common dates for performance metrics")
    # daily returns, daily risk-free rate and active (portfolio - benchmark) returns
    td = cfg["trading_days"]
    r, rb = nav.pct_change().dropna(), bench.pct_change().dropna()
    rf_d = (rf_annual / td).reindex(r.index, method="ffill").fillna(0.0)
    active = r - rb
    # headline metrics: returns, vol, Sharpe, drawdown, beta, tracking error
    out = {
        "as_of": idx[-1].date().isoformat(), "inception": idx[0].date().isoformat(),
        "n_days": int(len(r)),
        "return_portfolio": float(nav.iloc[-1] / nav.iloc[0] - 1),
        "return_benchmark": float(bench.iloc[-1] / bench.iloc[0] - 1),
        "volatility_portfolio": float(r.std() * np.sqrt(td)),
        "sharpe_portfolio": sharpe(r, rf_d, td),
        "max_drawdown_portfolio": max_drawdown(nav),
        "beta": float(r.cov(rb) / rb.var()) if rb.var() > 0 else float("nan"),
        "tracking_error": float(active.std() * np.sqrt(td)),
    }
    # derived metrics (guard against zero tracking error)
    out["excess_return"] = out["return_portfolio"] - out["return_benchmark"]
    te = out["tracking_error"]
    out["information_ratio"] = float(active.mean() * td / te) if te > 0 else float("nan")
    # fund-only statistics (the benchmark enters only through beta, tracking error, excess return)
    w, h = cfg["sharpe_window"], cfg["var_horizon"]
    # rolling Sharpe over the latest `w` returns (all of them while history is shorter)
    n = min(w, len(r))
    ex = (r - rf_d).iloc[-n:]
    out["sharpe_rolling"] = float(ex.mean() / ex.std() * np.sqrt(td)) if ex.std() > 0 else float("nan")
    out["sharpe_rolling_window_days"] = int(n)
    # realised daily (not annualised) volatility: last 10 returns and since inception
    out["daily_vol_10d"] = float(r.iloc[-10:].std())
    out["daily_vol_since_inception"] = float(r.std())
    # parametric (normal) VaR at 99% and 95%, as a fraction and in dollars
    for c in (0.99, 0.95):
        var = norm.ppf(c) * r.std() * np.sqrt(h)
        out[f"var_{h}d_{int(c * 100)}_pct"] = float(var)
        out[f"var_{h}d_{int(c * 100)}_usd"] = float(var * amount)
    # trailing one-week (5-day) return needs six NAV points
    if len(nav) > 5:
        out["return_5d"] = float(nav.iloc[-1] / nav.iloc[-6] - 1)
    return out
