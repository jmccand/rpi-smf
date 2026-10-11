"""Performance metrics (IPS Section 7) and Brinson attribution.

Daily returns throughout; the same frequency and the 3-month T-bill are used for the
portfolio and the benchmark. Volatility and Sharpe are annualised with
sqrt(trading_days). Metrics carried over from the original fund-metrics script (beta,
rolling Sharpe, parametric VaR, 5-day return) are kept.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import norm


def max_drawdown(nav: pd.Series) -> float:
    return float((nav / nav.cummax() - 1).min())


def sharpe(r: pd.Series, rf_daily: pd.Series, td: int) -> float:
    ex = r - rf_daily.reindex(r.index).ffill().fillna(0.0)
    sd = ex.std()
    return float(ex.mean() / sd * np.sqrt(td)) if sd > 0 else float("nan")


def performance_metrics(nav: pd.Series, bench: pd.Series, rf_annual: pd.Series, cfg: dict,
                        amount: float) -> Dict[str, float]:
    """`nav`/`bench` are levels from inception; they are aligned on common dates."""
    idx = nav.index.intersection(bench.index)
    nav, bench = nav.loc[idx], bench.loc[idx]
    if len(idx) < 3:
        raise ValueError("need at least 3 common dates for performance metrics")
    td = cfg["trading_days"]
    r, rb = nav.pct_change().dropna(), bench.pct_change().dropna()
    rf_d = (rf_annual / td).reindex(r.index, method="ffill").fillna(0.0)
    active = r - rb
    out = {
        "as_of": idx[-1].date().isoformat(), "inception": idx[0].date().isoformat(),
        "n_days": int(len(r)),
        "return_portfolio": float(nav.iloc[-1] / nav.iloc[0] - 1),
        "return_benchmark": float(bench.iloc[-1] / bench.iloc[0] - 1),
        "volatility_portfolio": float(r.std() * np.sqrt(td)),
        "volatility_benchmark": float(rb.std() * np.sqrt(td)),
        "sharpe_portfolio": sharpe(r, rf_d, td),
        "sharpe_benchmark": sharpe(rb, rf_d, td),
        "max_drawdown_portfolio": max_drawdown(nav),
        "max_drawdown_benchmark": max_drawdown(bench),
        "beta": float(r.cov(rb) / rb.var()) if rb.var() > 0 else float("nan"),
        "tracking_error": float(active.std() * np.sqrt(td)),
    }
    out["excess_return"] = out["return_portfolio"] - out["return_benchmark"]
    te = out["tracking_error"]
    out["information_ratio"] = float(active.mean() * td / te) if te > 0 else float("nan")
    w = cfg["sharpe_window"]
    if len(r) >= w:
        ex = (r - rf_d).iloc[-w:]
        out[f"sharpe_{w}d_rolling"] = float(ex.mean() / ex.std() * np.sqrt(td))
    h = cfg["var_horizon"]
    for c in (0.99, 0.95):
        var = norm.ppf(c) * r.std() * np.sqrt(h)
        out[f"var_{h}d_{int(c * 100)}_pct"] = float(var)
        out[f"var_{h}d_{int(c * 100)}_usd"] = float(var * amount)
    if len(nav) > 5:
        out["return_5d"] = float(nav.iloc[-1] / nav.iloc[-6] - 1)
    return out


def brinson_period(wp: pd.Series, wb: pd.Series, r: pd.Series, groups: pd.Series) -> pd.DataFrame:
    """Brinson-Fachler effects for one period by group (weights at period start).

    allocation = (wp_g - wb_g) (Rb_g - Rb);  selection = wb_g (Rp_g - Rb_g);
    interaction = (wp_g - wb_g)(Rp_g - Rb_g). Sum over groups = Rp - Rb.
    """
    names = sorted(set(wp.index) | set(wb.index))
    wp, wb = wp.reindex(names).fillna(0.0), wb.reindex(names).fillna(0.0)
    r = r.reindex(names).fillna(0.0)
    g = groups.reindex(names).fillna("UNCLASSIFIED")
    Wp, Wb = wp.groupby(g).sum(), wb.groupby(g).sum()
    Rp = (wp * r).groupby(g).sum() / Wp.replace(0, np.nan)
    Rb = (wb * r).groupby(g).sum() / Wb.replace(0, np.nan)
    Rb_tot = float((wb * r).sum())
    Rp, Rb = Rp.fillna(Rb).fillna(0.0), Rb.fillna(Rp).fillna(0.0)   # empty side: no selection
    df = pd.DataFrame({
        "allocation": (Wp - Wb) * (Rb - Rb_tot),
        "selection": Wb * (Rp - Rb),
        "interaction": (Wp - Wb) * (Rp - Rb),
    })
    return df


def link_attribution(periods: List[pd.DataFrame], rp: List[float], rb: List[float]) -> pd.DataFrame:
    """Combine per-period effects with Cariño smoothing so they sum to cumulative Rp - Rb."""
    def k(a: float, b: float) -> float:
        return 1 / (1 + a) if np.isclose(a, b) else (np.log1p(a) - np.log1p(b)) / (a - b)

    Rp, Rb = float(np.prod(1 + np.array(rp)) - 1), float(np.prod(1 + np.array(rb)) - 1)
    K = k(Rp, Rb)
    total = None
    for df, a, b in zip(periods, rp, rb):
        scaled = df * k(a, b) / K
        total = scaled if total is None else total.add(scaled, fill_value=0.0)
    return total
