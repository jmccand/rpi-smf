"""Expected-return model (IPS 4.2).

E[R_factor,i] = alpha + sum_k beta_k Z_ik, where alpha/beta are averages of monthly
cross-sectional OLS regressions of next-month excess returns on z-scored factors over
the trailing `regression_window_months`. Betas whose sign contradicts the expected
sign are set to 0. The result is blended with an N-month moving average of excess
returns: E[R] = phi * E[R_MA] + (1 - phi) * E[R_factor]. All returns are monthly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from rpi_smf.config import FactorSpec
from rpi_smf.factors import Context, compute_factors, zscore

log = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Everything the expected-return fit produces, kept for the run journal."""
    asof: pd.Timestamp
    expected_return: pd.Series      # blended monthly excess return, by ticker
    er_factor: pd.Series
    er_ma: pd.Series
    z: pd.DataFrame                 # z-scores at asof
    raw_factors: pd.DataFrame
    alpha: float
    beta_raw: pd.Series             # average monthly betas before sign rule
    beta: pd.Series                 # after sign rule
    months_used: pd.Series          # monthly regressions in which each factor was defined
    history: pd.DataFrame           # per-month alpha/betas, indexed by regression date


def month_ends(index: pd.DatetimeIndex, asof: pd.Timestamp) -> pd.DatetimeIndex:
    """Last trading date of each *completed* calendar month on or before asof."""
    # last trading day within each calendar month (no look-ahead past asof)
    idx = index[index <= asof]
    ends = pd.Series(idx, index=idx).groupby(idx.to_period("M")).last()
    ends = pd.DatetimeIndex(ends.values)
    if len(ends):
        # if the next business day is still in asof's month, that month is incomplete
        nxt = asof + pd.offsets.BDay(1)
        if ends[-1].to_period("M") == asof.to_period("M") and nxt.to_period("M") == asof.to_period("M"):
            ends = ends[:-1]            # asof's month is still in progress
    return ends


def monthly_rf(rf_annual: pd.Series, when: pd.Timestamp) -> float:
    """One-month risk-free return known at `when` (annual T-bill yield, compounded)."""
    # latest yield known on or before `when`, converted to a monthly rate
    y = rf_annual.loc[:when]
    return float((1 + y.iloc[-1]) ** (1 / 12) - 1) if len(y) else 0.0


def monthly_regression(z: pd.DataFrame, y: pd.Series, min_n: int):
    """OLS of y on [1, z]; returns (alpha, betas Series) or None if too few names."""
    # use only names with both a return and z-scores; bail out on a thin cross-section
    ok = y.notna() & z.index.isin(y.index)
    if ok.sum() < min_n:
        return None
    # design matrix: intercept column + z-scores
    X = np.column_stack([np.ones(ok.sum()), z.loc[ok.index[ok]].to_numpy()])
    coef, *_ = np.linalg.lstsq(X, y[ok].to_numpy(), rcond=None)   # min-norm: all-zero column -> 0
    return float(coef[0]), pd.Series(coef[1:], index=z.columns)


def fit_model(prices: pd.DataFrame, rf_annual: pd.Series,
              specs: List[FactorSpec], asof, cfg: dict,
              universe_fn: Optional[Callable[[pd.Timestamp], List[str]]] = None,
              market: Optional[pd.Series] = None) -> ModelResult:
    """Fit on data up to `asof` only. `cfg` is the ips.yml `model` section.

    `universe_fn(date)` may return the tickers that were investable at `date`
    (point-in-time Nasdaq-100 membership); by default all columns of `prices`.
    """
    # truncate prices at asof so nothing after it can leak into the fit
    asof = pd.Timestamp(asof)
    prices = prices.loc[:asof]
    names = [s.name for s in specs]
    ends = month_ends(prices.index, asof)
    window = int(cfg["regression_window_months"])
    if len(ends) < 2:
        raise ValueError("not enough price history for the regression")
    reg_dates = ends[:-1][-window:]                     # month t -> return over (t, next end]
    min_n, cap = int(cfg["min_cross_section"]), float(cfg["zscore_cap"])

    # one cross-sectional regression per month: factors at t vs excess return over t -> t_next
    rows = []
    for t in reg_dates:
        t_next = ends[ends.get_loc(t) + 1]
        cols = list(universe_fn(t)) if universe_fn else list(prices.columns)
        cols = [c for c in cols if c in prices.columns]
        ctx = Context(t, prices[cols], market)
        raw = compute_factors(ctx, specs)
        z = zscore(raw, cap)
        # excess return = price return minus that month's risk-free return
        ret = prices.loc[t_next, cols] / prices.loc[t, cols] - 1 - monthly_rf(rf_annual, t)
        fit = monthly_regression(z, ret, min_n)
        if fit is None:
            continue
        alpha_t, beta_t = fit
        # factors with too few defined names this month are recorded as NaN
        defined = raw.count() >= min_n
        row = {"date": t, "alpha": alpha_t}
        for n in names:
            row[n] = beta_t[n] if defined[n] else np.nan   # undefined factor-months excluded from average
        rows.append(row)
    if not rows:
        raise ValueError("no monthly regression could be estimated")
    hist = pd.DataFrame(rows).set_index("date")

    # average the monthly betas, then zero any whose sign contradicts the expectation
    beta_raw = hist[names].mean().fillna(0.0)
    months_used = hist[names].count()
    signs = pd.Series({s.name: s.expected_sign for s in specs})
    beta = beta_raw.where(beta_raw * signs >= 0, 0.0)
    # log every factor that was dropped or zeroed so the journal explains the tilt
    for n in names:
        if months_used[n] == 0:
            log.warning("factor %s undefined in every regression month; beta=0", n)
        elif beta[n] == 0 and beta_raw[n] != 0:
            log.warning("factor %s beta %.4f contradicts expected sign; set to 0", n, beta_raw[n])
    alpha = float(hist["alpha"].mean())

    # current cross-section: score today's z-scores with the averaged betas
    cols = list(universe_fn(asof)) if universe_fn else list(prices.columns)
    cols = [c for c in cols if c in prices.columns]
    raw_now = compute_factors(Context(asof, prices[cols], market), specs)
    z_now = zscore(raw_now, cap)
    er_factor = alpha + z_now[names] @ beta[names]

    # N-month moving average of excess returns over the last completed months
    n_ma = int(cfg["ma_months"])
    # needs n_ma+1 month-ends to form n_ma monthly returns
    me = ends[-(n_ma + 1):]
    if len(me) == n_ma + 1:
        mret = prices.loc[me, cols].pct_change(fill_method=None).iloc[1:]
        rfm = pd.Series([monthly_rf(rf_annual, d) for d in me[:-1]], index=mret.index)
        er_ma = mret.sub(rfm, axis=0).mean().where(mret.count() == n_ma)
    else:
        er_ma = pd.Series(np.nan, index=cols)
    # blend MA and factor forecasts; fall back to the factor forecast where the MA is missing
    phi = float(cfg["phi"])
    er = (phi * er_ma + (1 - phi) * er_factor).where(er_ma.notna(), er_factor)
    return ModelResult(asof, er.rename("expected_return"), er_factor, er_ma, z_now, raw_now,
                       alpha, beta_raw, beta, months_used, hist)
