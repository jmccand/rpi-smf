"""Orchestration of the weekly rebalance (IPS Section 4) and performance reporting.

All analytics use data up to `asof` = the last trading day before the rebalance date
(IPS 4.4); the only data from the rebalance date itself is the simulated execution
price. Nothing here contains finance logic: it wires data -> factors -> model ->
optimiser -> compliance -> trades -> journal.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from rpi_smf import journal, portfolio as pf_mod
from rpi_smf.compliance import ComplianceReport, check_compliance
from rpi_smf.config import CONFIG_DIR, IPS, REPO_ROOT, load_config
from rpi_smf.data import gics as gics_mod, nasdaq, prices as prices_mod
from rpi_smf.model import ModelResult, fit_model
from rpi_smf.optimize import OptimizationResult, optimize
from rpi_smf.performance import performance_metrics
from rpi_smf.trades import TradePlan, propose_trades

log = logging.getLogger(__name__)
RAW = REPO_ROOT / "data" / "raw"


@dataclass
class Paths:
    """File locations used by the pipeline (overridable from the CLI and tests)."""
    constituents: Path = RAW / "nq100_constituents.csv"
    gics: Path = RAW / "gics.csv"
    runs: Path = REPO_ROOT / "runs"
    ips: Path = CONFIG_DIR / "ips.yml"
    factors: Path = CONFIG_DIR / "factors.yml"



@dataclass
class Universe:
    """The investable universe and its market data as of `asof`."""
    asof: pd.Timestamp
    tickers: List[str]
    bench_weights: pd.Series          # Nasdaq-100 weights over `tickers`, sums to 1
    gics: pd.Series
    prices: pd.DataFrame              # universe adjusted closes, full history (callers truncate)
    bench_prices: pd.Series           # benchmark (QQQM) adjusted-close series
    rf: pd.Series
    excluded: Dict[str, str]


@dataclass
class RebalanceResult:
    """Everything a rebalance produced, for printing and journaling."""
    date: pd.Timestamp
    asof: pd.Timestamp
    model: ModelResult
    opt: OptimizationResult
    plan: TradePlan
    compliance_target: ComplianceReport
    compliance_current: ComplianceReport
    compliance_post: ComplianceReport
    nav: float
    run_dir: Optional[Path]
    universe: Universe


class Pipeline:
    """Wires data, model, optimiser, compliance, trades and journal together."""

    def __init__(self, paths: Paths, refresh: bool = False):
        # load the IPS/factor config once; `ips` is the raw dict used for section lookups
        self.paths, self.refresh = paths, refresh
        self.cfg: IPS = load_config(paths.ips, paths.factors)
        self.ips = self.cfg.raw

    # ---- data -----------------------------------------------------------------
    def _screen(self, asof: Optional[pd.Timestamp] = None):
        """Nasdaq-100 constituents with prices and enough history; no GICS needed."""
        opt = self.ips["optimizer"]
        bench = self.ips["run"]["benchmark"]
        # fetch constituents and the benchmark together, then split the benchmark out
        listed, excluded = nasdaq.load_constituents(self.paths.constituents), {}
        px = prices_mod.get_prices(list(listed.index) + [bench], refresh=self.refresh)
        if bench not in px:
            raise RuntimeError(f"no price data for benchmark {bench}")
        fund = px.pop(bench).dropna()
        asof = pd.Timestamp(asof) if asof is not None else px.index[-1]
        # a name needs min_history_days of returns (+1 price) with no gaps up to asof
        need = opt["min_history_days"] + 1
        keep = []
        for t in listed.index:
            if t not in px.columns:
                excluded[t] = "no price data"
            elif px[t].loc[:asof].iloc[-need:].isna().any() or px[t].loc[:asof].count() < need:
                excluded[t] = f"fewer than {opt['min_history_days']} days of price history"
            else:
                keep.append(t)
        return listed, px, fund, asof, keep, excluded

    def tickers(self) -> List[str]:
        """Investable tickers (priced, enough history); does not need gics.csv."""
        return self._screen()[4]

    def universe(self, asof: Optional[pd.Timestamp] = None) -> Universe:
        """Investable universe = Nasdaq-100 constituents with prices, history and GICS.

        `asof` defaults to the latest price date. Names are dropped (and recorded in
        `excluded`) when they lack price data or fewer than `min_history_days` of history.
        """
        opt = self.ips["optimizer"]
        listed, px, fund, asof, keep, excluded = self._screen(asof)
        # every investable name must have a GICS class for the sector caps
        if not Path(self.paths.gics).exists():
            raise FileNotFoundError(f"GICS file not found: {self.paths.gics} "
                                    "(columns: ticker,sub_industry[,sector,industry_group,industry])")
        gics = gics_mod.load_gics(self.paths.gics, opt["gics_level"])
        missing = [t for t in keep if t not in gics.index]
        if missing:
            raise ValueError(f"gics.csv has no classification for investable names: {missing}")
        # renormalise benchmark weights over the surviving names
        w = listed.loc[keep, "weight"]
        rf = prices_mod.get_risk_free(
            series=self.ips["performance"]["risk_free"], refresh=self.refresh)
        return Universe(asof, keep, w / w.sum(), gics.reindex(keep), px[keep], fund, rf, excluded)

    @staticmethod
    def prior_trading_day(index: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp:
        """Last date in `index` strictly before `date` (the as-of date for a rebalance)."""
        before = index[index < date]
        if not len(before):
            raise ValueError(f"no price data before {date.date()}")
        return before[-1]

    # ---- state ----------------------------------------------------------------
    def ensure_portfolio(self, u: Universe, save: bool = True) -> pf_mod.Portfolio:
        """Load the simulated portfolio, creating it at the configured start date if absent
        (persisted only when `save`, so dry runs leave no state behind)."""
        state = pf_mod.load_state(self.paths.runs)
        if state is not None:
            return state
        # first run: buy the benchmark weights at the last price on or before the start date
        start = pd.Timestamp(self.cfg.start_date)
        px = u.prices.loc[:start].iloc[-1]
        if u.prices.loc[:start].index[-1] != start:
            log.warning("no price on start date %s; using %s", start.date(),
                        u.prices.loc[:start].index[-1].date())
        pf = pf_mod.initial_portfolio(u.bench_weights, px, self.cfg.initial_capital, start)
        if save:
            pf_mod.save_state(self.paths.runs, pf)
        log.info("initialised portfolio at benchmark weights on %s", start.date())
        return pf

    # ---- steps ----------------------------------------------------------------
    def market(self) -> pd.Series:
        """Market proxy prices for CAPM beta (`model.market_proxy`); raises if unavailable."""
        tkr = self.ips["model"].get("market_proxy")
        if not tkr:
            raise ValueError("ips.yml model.market_proxy is not set")
        px = prices_mod.get_prices([tkr], refresh=self.refresh)
        if tkr not in px or px[tkr].dropna().empty:
            raise ValueError(f"no price data for market proxy {tkr}")
        return px[tkr].dropna()

    def fit(self, u: Universe) -> ModelResult:
        """Fit the expected-return model on the universe as of `u.asof`."""
        return fit_model(u.prices, u.rf, self.cfg.factors, u.asof,
                         self.ips["model"], market=self.market())

    def bench_vol(self, u: Universe) -> float:
        """Annualised benchmark volatility over the trailing covariance window."""
        n = self.ips["optimizer"]["cov_window_days"]
        r = u.bench_prices.loc[:u.asof].pct_change().dropna().iloc[-n:]
        return float(r.std() * np.sqrt(self.ips["optimizer"]["trading_days"]))

    def window_returns(self, u: Universe) -> pd.DataFrame:
        """Trailing daily returns of the universe, used for the covariance matrix."""
        n = self.ips["optimizer"]["cov_window_days"]
        return u.prices.loc[:u.asof].pct_change(fill_method=None).iloc[-n:]

    def rebalance(self, date, commit: bool = True) -> RebalanceResult:
        """Run one rebalance; with `commit` also journal it and update the portfolio."""
        date = pd.Timestamp(date)
        # as-of is the trading day before the rebalance date; rebuild the universe at that date
        u0 = self.universe()
        asof = self.prior_trading_day(u0.prices.index, date)
        u = self.universe(asof)
        pf = self.ensure_portfolio(u, save=commit)
        if date < pf.date:
            raise ValueError(f"rebalance date {date.date()} precedes portfolio date {pf.date.date()}")
        # execution prices: the rebalance date's close (or the last close before it)
        held = prices_mod.get_prices(list(pf.shares.index.union(u.tickers)), refresh=False)
        exec_date = held.loc[:date].index[-1]
        if exec_date != date:
            log.warning("no price on %s; executing at %s close", date.date(), exec_date.date())
        px = held.loc[:exec_date].ffill().iloc[-1]
        nav = pf.value(px)
        cur = pf.weights(px)

        # forecast returns, then optimise toward the target weights
        model = self.fit(u)
        returns = self.window_returns(u)
        bvol = self.bench_vol(u)
        opt_cfg = self.ips["optimizer"]
        comp_cfg = self.ips["compliance"]
        res = optimize(model.expected_return, returns, cur, u.bench_weights, u.gics, bvol, opt_cfg,
                       min_positions=comp_cfg["min_positions"],
                       min_weight=comp_cfg["min_position_weight"])
        targets = res.weights

        def comp(w: pd.Series) -> ComplianceReport:
            """IPS compliance of a weight vector over the universe."""
            w = w.reindex(u.tickers).fillna(0.0)
            return check_compliance(w / w.sum(), u.gics, returns, u.bench_weights, bvol, self.ips)

        # threshold-based trades, forced to stay compliant
        plan = propose_trades(cur, targets, nav, px, self.ips["trading"], opt_cfg["single_name_cap"],
                              is_compliant=lambda w: comp(w).passed)
        c_target, c_current, c_post = comp(targets), comp(cur), comp(plan.post_weights)
        # only a committed run writes the journal and moves the simulated portfolio
        run_dir = None
        if commit:
            run_dir = self._journal(date, asof, u, model, res, plan, c_target, c_current, c_post,
                                    nav, cur)
            new = pf.apply(plan.trades, exec_date, nav, plan.total_cost)
            pf_mod.save_state(self.paths.runs, new)
        return RebalanceResult(date, asof, model, res, plan, c_target, c_current, c_post, nav,
                               run_dir, u)

    def _journal(self, date, asof, u, model, res, plan, c_target, c_current, c_post, nav, cur):
        """Write the run's tables, manifest and journal line; returns the run directory."""
        # tables saved as CSV for audit
        tables = {
            "factors_raw": model.raw_factors, "factors_z": model.z,
            "betas": pd.DataFrame({"beta_raw": model.beta_raw, "beta": model.beta,
                                   "months_used": model.months_used}),
            "beta_history": model.history,
            "expected_returns": pd.DataFrame({"expected_return": model.expected_return,
                                              "er_factor": model.er_factor,
                                              "er_ma": model.er_ma}),
            "weights": pd.DataFrame({"current": cur.reindex(u.tickers).fillna(0.0),
                                     "target": res.weights, "post_trade": plan.post_weights,
                                     "benchmark": u.bench_weights, "gics": u.gics}),
            "trades": plan.trades,
            "compliance_target": c_target.table, "compliance_current": c_current.table,
            "compliance_post_trade": c_post.table,
        }
        # manifest: headline numbers, solver/model settings and known caveats
        manifest = {
            "rebalance_date": date.date().isoformat(), "asof": asof.date().isoformat(),
            "nav_before_trades": nav, "lambda": res.lam, "lambda_status": res.lambda_status,
            "target_vol": res.target_vol, "benchmark_vol": res.benchmark_vol,
            "ex_ante": {"vol": res.vol, "tracking_error": res.tracking_error,
                        "expected_return_annual": res.expected_return, "sharpe": res.sharpe,
                        "turnover": res.turnover, "single_name_cap_used": res.cap_used,
                        "n_positions": res.n_positions},
            "solver": res.settings, "model": {"alpha": model.alpha,
                                               "beta": model.beta.to_dict(),
                                               "months_used": model.months_used.to_dict()},
            "trades": {"count": len(plan.trades), "cost_usd": plan.total_cost},
            "compliance_target_passed": c_target.passed,
            "compliance_post_trade_passed": c_post.passed,
            "universe_size": len(u.tickers), "excluded": u.excluded,
            "benchmark_weights": nasdaq.load_meta(self.paths.constituents),
            "caveats": ["universe is today's Nasdaq-100 constituents applied historically "
                        "(survivorship/look-ahead); benchmark weights per benchmark_weights.source"],
        }
        # hash inputs and configs so the run can be reproduced, then add the summary line
        run = journal.write_run(
            self.paths.runs, date.date().isoformat(), tables, manifest,
            input_files=[self.paths.constituents, self.paths.gics],
            config_files=[self.paths.ips, self.paths.factors])
        journal.append_journal(self.paths.runs, {
            "rebalance_date": date.date().isoformat(), "asof": asof.date().isoformat(),
            "nav": round(nav, 2), "lambda": res.lam, "vol": res.vol, "tracking_error": res.tracking_error,
            "sharpe": res.sharpe, "n_trades": len(plan.trades), "cost_usd": plan.total_cost,
            "compliant": c_post.passed, "git_commit": journal.git_commit()})
        return run

    def monitor(self, date=None) -> ComplianceReport:
        """Compliance of the current simulated holdings (Risk Analyst - Portfolio Monitoring)."""
        # as-of is the last trading day on or before `date`
        u0 = self.universe()
        date = pd.Timestamp(date) if date is not None else u0.prices.index[-1]
        asof = self.prior_trading_day(u0.prices.index, date + pd.Timedelta(days=1))
        u = self.universe(asof)
        pf = self.ensure_portfolio(u)
        # current weights at as-of prices, laid out over the universe
        cur = pf.weights(u.prices.loc[:asof].iloc[-1].reindex(pf.shares.index)).reindex(u.tickers).fillna(0.0)
        return check_compliance(cur / cur.sum(), u.gics, self.window_returns(u), u.bench_weights,
                                self.bench_vol(u), self.ips)

    def performance(self, end=None):
        """Metrics versus the benchmark (QQQM), with the NAV and benchmark series."""
        u = self.universe()
        pf = self.ensure_portfolio(u)
        # rebuild daily NAV from the holdings history and compare with the benchmark
        hist = pf_mod.load_history(self.paths.runs)
        end = pd.Timestamp(end) if end is not None else u.prices.index[-1]
        all_px = prices_mod.get_prices(list(hist.columns.union(u.tickers))).loc[:end]
        nav = pf_mod.nav_series(hist, all_px)
        bench = u.bench_prices.loc[nav.index[0]:end]
        m = performance_metrics(nav, bench, u.rf, self.ips["performance"], self.cfg.initial_capital)
        return m, nav, bench
