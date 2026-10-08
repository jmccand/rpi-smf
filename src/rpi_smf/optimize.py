"""Mean-variance optimisation with turnover penalty (IPS 4.3).

    max_w  w'mu - (lambda/2) w'Cw - tau * ||w - w_prev||_1
    s.t.   w >= 0, sum(w) = 1, w <= single_name_cap,
           sum of w within each GICS group <= gics_cap,
           sqrt(w'Cw) <= vol_ratio_max * sigma_FSPTX,
           sqrt((w-b)'C(w-b)) <= te_max

mu and C are annualised (monthly expected returns x 12; daily covariance x 252 with
diagonal shrinkage). lambda is calibrated by bisection so ex-ante volatility equals
vol_ratio_target x FSPTX's trailing volatility. If the constraints make the problem
infeasible, OptimizationError carries the smallest achievable tracking error.

The IPS (Section 6) also requires >= 30 positions, which the programme in 4.3 does not
enforce (a cardinality constraint is non-convex). If the solution holds fewer, the
single-name cap is tightened (bisection, lambda recalibrated at each cap) to the
largest value that yields enough positions; the cap actually used is reported.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import cvxpy as cp
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class OptimizationError(RuntimeError):
    pass


@dataclass
class OptimizationResult:
    weights: pd.Series
    lam: float
    vol: float                    # ex-ante annualised volatility
    tracking_error: float         # ex-ante annualised, vs benchmark weights
    expected_return: float        # annualised expected excess return
    sharpe: float                 # expected excess return / vol
    turnover: float               # ||w - w_prev||_1
    benchmark_vol: float
    target_vol: float
    lambda_status: str            # 'calibrated' | 'vol_target_unreachable_high' | '..._low'
    cap_used: float = 0.0         # single-name cap in force (below the IPS cap if tightened)
    n_positions: int = 0
    settings: Dict = field(default_factory=dict)


def shrunk_covariance(returns: pd.DataFrame, shrinkage: float, trading_days: int) -> pd.DataFrame:
    """Annualised sample covariance with diagonal shrinkage: (1-d) S + d diag(S)."""
    S = returns.cov().to_numpy()
    C = (1 - shrinkage) * S + shrinkage * np.diag(np.diag(S))
    return pd.DataFrame(C * trading_days, index=returns.columns, columns=returns.columns)


class _Problem:
    def __init__(self, mu, C, w_prev, b, groups, cfg, bench_vol):
        n = len(mu)
        self.n, self.cfg = n, cfg
        L = np.linalg.cholesky(C + 1e-10 * np.eye(n))
        self.L = L
        self.w = cp.Variable(n)
        self.lam = cp.Parameter(nonneg=True)
        w = self.w
        risk = cp.sum_squares(L.T @ w)
        turnover = cp.norm1(w - w_prev)
        self.obj = cp.Maximize(mu @ w - (self.lam / 2) * risk - cfg["tau"] * turnover)
        self.constraints = [
            w >= 0, cp.sum(w) == 1, w <= cfg["single_name_cap"],
            cp.norm(L.T @ w, 2) <= cfg["vol_ratio_max"] * bench_vol,
            cp.norm(L.T @ (w - b), 2) <= cfg["te_max"],
        ]
        for g in groups.unique():
            idx = np.flatnonzero((groups == g).to_numpy())
            self.constraints.append(cp.sum(w[idx]) <= cfg["gics_cap"])
        self.prob = cp.Problem(self.obj, self.constraints)
        self.mu, self.C, self.b, self.w_prev = mu, C, b, w_prev

    def solve(self, lam: float) -> np.ndarray:
        self.lam.value = lam
        self.prob.solve(solver=cp.CLARABEL)
        if self.prob.status not in ("optimal", "optimal_inaccurate"):
            raise OptimizationError(f"solver status {self.prob.status}")
        return np.clip(self.w.value, 0, None)

    def vol(self, w) -> float:
        return float(np.sqrt(w @ self.C @ w))

    def te(self, w) -> float:
        d = w - self.b
        return float(np.sqrt(d @ self.C @ d))

    def min_te(self) -> Optional[float]:
        """Smallest tracking error achievable under every constraint except the TE cap."""
        cons = [c for i, c in enumerate(self.constraints) if i != 4]
        p = cp.Problem(cp.Minimize(cp.norm(self.L.T @ (self.w - self.b), 2)), cons)
        try:
            p.solve(solver=cp.CLARABEL)
        except Exception:
            return None
        return float(p.value) if p.status in ("optimal", "optimal_inaccurate") else None


def _calibrate(pr: "_Problem", target: float):
    """Bisection on lambda so ex-ante vol = target. Returns (w, lambda, status)."""
    lo, hi = 1e-4, 1e4
    w_lo, w_hi = pr.solve(lo), pr.solve(hi)
    if pr.vol(w_lo) <= target:
        return w_lo, lo, "vol_target_unreachable_high"
    if pr.vol(w_hi) >= target:
        return w_hi, hi, "vol_target_unreachable_low"
    a, c = np.log(lo), np.log(hi)
    for _ in range(60):
        mid = (a + c) / 2
        w = pr.solve(float(np.exp(mid)))
        if abs(pr.vol(w) - target) < 1e-5 * target:
            break
        if pr.vol(w) > target:
            a = mid           # too risky -> raise lambda
        else:
            c = mid
    return w, float(np.exp(mid)), "calibrated"


def _clean(w: np.ndarray) -> np.ndarray:
    w = np.where(w < 1e-6, 0.0, w)
    return w / w.sum()


def optimize(mu_monthly: pd.Series, returns: pd.DataFrame, w_prev: pd.Series,
             bench_weights: pd.Series, gics: pd.Series, bench_vol: float, cfg: dict,
             min_positions: int = 0, min_weight: float = 0.001) -> OptimizationResult:
    """`returns`: trailing daily returns (columns = tickers); `bench_vol`: FSPTX annualised vol."""
    tickers = list(mu_monthly.index)
    missing = [t for t in tickers if t not in gics.index or pd.isna(gics[t])]
    if missing:
        raise OptimizationError(f"no GICS classification for: {missing}")
    if returns[tickers].isna().any().any():
        raise OptimizationError("returns contain NaN; filter names with incomplete history")
    C = shrunk_covariance(returns[tickers], cfg["cov_shrinkage"], cfg["trading_days"])
    mu = (mu_monthly * cfg["months_per_year"]).to_numpy()
    b = bench_weights.reindex(tickers).fillna(0.0)
    b = (b / b.sum()).to_numpy()
    wp = w_prev.reindex(tickers).fillna(0.0).to_numpy()
    if cfg["single_name_cap"] * len(tickers) < 1:
        raise OptimizationError("single-name cap infeasible for this universe size")
    target = cfg["vol_ratio_target"] * bench_vol
    groups = gics.loc[tickers]

    def run(cap: float):
        pr = _Problem(mu, C.to_numpy(), wp, b, groups, dict(cfg, single_name_cap=cap), bench_vol)
        w, lam, status = _calibrate(pr, target)
        return pr, _clean(w), lam, status

    def n_pos(w) -> int:
        return int((w >= min_weight).sum())

    cap = cfg["single_name_cap"]
    try:
        pr, w, lam, status = run(cap)
    except OptimizationError as e:
        pr0 = _Problem(mu, C.to_numpy(), wp, b, groups, cfg, bench_vol)
        te = pr0.min_te()
        raise OptimizationError(
            f"constraints infeasible ({e}); minimum achievable tracking error vs benchmark is "
            f"{te:.2%} (limit {cfg['te_max']:.2%})" if te is not None else f"infeasible ({e})")
    if min_positions and n_pos(w) < min_positions:
        floor = 1.0 / min_positions
        try:
            best = run(floor)
        except OptimizationError as e:
            raise OptimizationError(f"cannot hold {min_positions} positions within the other "
                                    f"constraints (cap {floor:.2%}): {e}")
        lo, hi = floor, cap          # lo satisfies the count, hi does not
        for _ in range(14):
            mid = (lo + hi) / 2
            try:
                cand = run(mid)
            except OptimizationError:
                hi = mid
                continue
            if n_pos(cand[1]) >= min_positions:
                lo, best = mid, cand
            else:
                hi = mid
        pr, w, lam, status = best
        cap = lo
        log.warning("30-position minimum enforced by tightening the single-name cap to %.2f%%",
                    cap * 100)
    vol = pr.vol(w)
    er = float(mu @ w)
    settings = {"solver": "CLARABEL", "cvxpy": cp.__version__, "cov_shrinkage": cfg["cov_shrinkage"],
                "tau": cfg["tau"], "single_name_cap_ips": cfg["single_name_cap"],
                "single_name_cap_used": cap, "gics_cap": cfg["gics_cap"],
                "vol_ratio_max": cfg["vol_ratio_max"], "te_max": cfg["te_max"],
                "n_assets": len(tickers)}
    return OptimizationResult(
        weights=pd.Series(w, index=tickers, name="target_weight"), lam=lam, vol=vol,
        tracking_error=pr.te(w), expected_return=er, sharpe=er / vol,
        turnover=float(np.abs(w - wp).sum()), benchmark_vol=bench_vol, target_vol=target,
        lambda_status=status, settings=settings, cap_used=cap, n_positions=n_pos(w))
