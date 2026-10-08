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
from rpi_smf.data import fundamentals, gics as gics_mod, holdings as holdings_mod, prices as prices_mod
from rpi_smf.model import ModelResult, fit_model
from rpi_smf.optimize import OptimizationResult, optimize
from rpi_smf.performance import brinson_period, link_attribution, performance_metrics
from rpi_smf.trades import TradePlan, propose_trades

log = logging.getLogger(__name__)
RAW = REPO_ROOT / "data" / "raw"


@dataclass
class Paths:
    holdings: Path
    gics: Path = RAW / "gics.csv"
    estimates: Path = RAW / "estimates"
    universe: Path = CONFIG_DIR / "universe.yml"
    runs: Path = REPO_ROOT / "runs"
    ips: Path = CONFIG_DIR / "ips.yml"
    factors: Path = CONFIG_DIR / "factors.yml"

    @classmethod
    def default(cls, holdings: Optional[Path] = None) -> "Paths":
        if holdings is None:
            found = sorted(RAW.glob("fsptx_holdings_*.csv"))
            if not found:
                raise FileNotFoundError(f"no fsptx_holdings_*.csv in {RAW}")
            holdings = found[-1]
        return cls(holdings=Path(holdings))


@dataclass
class Universe:
    asof: pd.Timestamp
    tickers: List[str]
    bench_weights: pd.Series          # FSPTX look-through weights over `tickers`, sums to 1
    gics: pd.Series
    prices: pd.DataFrame              # universe adjusted closes, full history (callers truncate)
    fund_prices: pd.Series            # actual FSPTX NAV series
    rf: pd.Series
    excluded: Dict[str, str]


@dataclass
class RebalanceResult:
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
    def __init__(self, paths: Paths, refresh: bool = False):
        self.paths, self.refresh = paths, refresh
        self.cfg: IPS = load_config(paths.ips, paths.factors)
        self.ips = self.cfg.raw

    # ---- data -----------------------------------------------------------------
    def estimates(self) -> pd.DataFrame:
        return fundamentals.load_estimates(self.paths.estimates)

    def universe(self, asof: Optional[pd.Timestamp] = None) -> Universe:
        """Investable universe = listed FSPTX equities with prices, history and GICS.

        `asof` defaults to the latest price date. Names are dropped (and recorded in
        `excluded`) when they lack price data or fewer than `min_history_days` of history.
        """
        opt = self.ips["optimizer"]
        ucfg = holdings_mod.load_universe_config(self.paths.universe)
        allh = holdings_mod.parse_fidelity_csv(self.paths.holdings)
        listed = holdings_mod.listed_equities(allh, ucfg["ticker_overrides"], ucfg["exclude"])
        excluded = {r.ticker: r.kind for r in allh.itertuples() if r.kind != "listed"}
        px = prices_mod.get_prices(list(listed.index) + ["FSPTX"], refresh=self.refresh)
        if "FSPTX" not in px:
            raise RuntimeError("no price data for FSPTX")
        fund = px.pop("FSPTX").dropna()
        asof = pd.Timestamp(asof) if asof is not None else px.index[-1]
        need = opt["min_history_days"] + 1
        keep = []
        for t in listed.index:
            if t not in px.columns:
                excluded[t] = "no price data"
            elif px[t].loc[:asof].iloc[-need:].isna().any() or px[t].loc[:asof].count() < need:
                excluded[t] = f"fewer than {opt['min_history_days']} days of price history"
            else:
                keep.append(t)
        if not Path(self.paths.gics).exists():
            raise FileNotFoundError(f"GICS file not found: {self.paths.gics} "
                                    "(columns: ticker,sub_industry[,sector,industry_group,industry])")
        gics = gics_mod.load_gics(self.paths.gics, opt["gics_level"])
        missing = [t for t in keep if t not in gics.index]
        if missing:
            raise ValueError(f"gics.csv has no classification for investable names: {missing}")
        w = listed.loc[keep, "weight"]
        rf = prices_mod.get_risk_free(
            series=self.ips["performance"]["risk_free"], refresh=self.refresh)
        return Universe(asof, keep, w / w.sum(), gics.reindex(keep), px[keep], fund, rf, excluded)

    @staticmethod
    def prior_trading_day(index: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp:
        before = index[index < date]
        if not len(before):
            raise ValueError(f"no price data before {date.date()}")
        return before[-1]

    # ---- state ----------------------------------------------------------------
    def ensure_portfolio(self, u: Universe) -> pf_mod.Portfolio:
        """Load the simulated portfolio, creating it at the configured start date if absent."""
        state = pf_mod.load_state(self.paths.runs)
        if state is not None:
            return state
        start = pd.Timestamp(self.cfg.start_date)
        px = u.prices.loc[:start].iloc[-1]
        if u.prices.loc[:start].index[-1] != start:
            log.warning("no price on start date %s; using %s", start.date(),
                        u.prices.loc[:start].index[-1].date())
        pf = pf_mod.initial_portfolio(u.bench_weights, px, self.cfg.initial_capital, start)
        pf_mod.save_state(self.paths.runs, pf)
        log.info("initialised portfolio at FSPTX weights on %s", start.date())
        return pf

    # ---- steps ----------------------------------------------------------------
    def fit(self, u: Universe) -> ModelResult:
        return fit_model(u.prices, self.estimates(), u.rf, self.cfg.factors, u.asof,
                         self.ips["model"])

    def bench_vol(self, u: Universe) -> float:
        n = self.ips["optimizer"]["cov_window_days"]
        r = u.fund_prices.loc[:u.asof].pct_change().dropna().iloc[-n:]
        return float(r.std() * np.sqrt(self.ips["optimizer"]["trading_days"]))

    def window_returns(self, u: Universe) -> pd.DataFrame:
        n = self.ips["optimizer"]["cov_window_days"]
        return u.prices.loc[:u.asof].pct_change(fill_method=None).iloc[-n:]

    def rebalance(self, date, commit: bool = True) -> RebalanceResult:
        date = pd.Timestamp(date)
        u0 = self.universe()
        asof = self.prior_trading_day(u0.prices.index, date)
        u = self.universe(asof)
        pf = self.ensure_portfolio(u)
        if date < pf.date:
            raise ValueError(f"rebalance date {date.date()} precedes portfolio date {pf.date.date()}")
        held = prices_mod.get_prices(list(pf.shares.index.union(u.tickers)), refresh=False)
        exec_date = held.loc[:date].index[-1]
        if exec_date != date:
            log.warning("no price on %s; executing at %s close", date.date(), exec_date.date())
        px = held.loc[:exec_date].ffill().iloc[-1]
        nav = pf.value(px)
        cur = pf.weights(px)

        model = self.fit(u)
        returns = self.window_returns(u)
        bvol = self.bench_vol(u)
        opt_cfg = self.ips["optimizer"]
        res = optimize(model.expected_return, returns, cur, u.bench_weights, u.gics, bvol, opt_cfg)
        targets = res.weights

        def comp(w: pd.Series) -> ComplianceReport:
            w = w.reindex(u.tickers).fillna(0.0)
            return check_compliance(w / w.sum(), u.gics, returns, u.bench_weights, bvol, self.ips)

        plan = propose_trades(cur, targets, nav, px, self.ips["trading"], opt_cfg["single_name_cap"])
        c_target, c_current, c_post = comp(targets), comp(cur), comp(plan.post_weights)
        run_dir = None
        if commit:
            run_dir = self._journal(date, asof, u, model, res, plan, c_target, c_current, c_post,
                                    nav, cur)
            new = pf.apply(plan.trades, exec_date, nav, plan.total_cost)
            pf_mod.save_state(self.paths.runs, new)
        return RebalanceResult(date, asof, model, res, plan, c_target, c_current, c_post, nav,
                               run_dir, u)

    def _journal(self, date, asof, u, model, res, plan, c_target, c_current, c_post, nav, cur):
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
                                     "fsptx": u.bench_weights, "gics": u.gics}),
            "trades": plan.trades,
            "compliance_target": c_target.table, "compliance_current": c_current.table,
            "compliance_post_trade": c_post.table,
        }
        manifest = {
            "rebalance_date": date.date().isoformat(), "asof": asof.date().isoformat(),
            "nav_before_trades": nav, "lambda": res.lam, "lambda_status": res.lambda_status,
            "target_vol": res.target_vol, "benchmark_vol": res.benchmark_vol,
            "ex_ante": {"vol": res.vol, "tracking_error": res.tracking_error,
                        "expected_return_annual": res.expected_return, "sharpe": res.sharpe,
                        "turnover": res.turnover},
            "solver": res.settings, "model": {"alpha": model.alpha,
                                               "beta": model.beta.to_dict(),
                                               "months_used": model.months_used.to_dict()},
            "trades": {"count": len(plan.trades), "cost_usd": plan.total_cost},
            "compliance_target_passed": c_target.passed,
            "compliance_post_trade_passed": c_post.passed,
            "universe_size": len(u.tickers), "excluded": u.excluded,
            "estimates_source": "see data/raw/estimates (Bloomberg/FactSet export and/or Yahoo proxy snapshots)",
            "caveats": ["universe is today's FSPTX holdings applied historically (survivorship/look-ahead); "
                        "point-in-time N-PORT membership not yet wired in"],
        }
        run = journal.write_run(
            self.paths.runs, date.date().isoformat(), tables, manifest,
            input_files=[self.paths.holdings, self.paths.gics, self.paths.estimates],
            config_files=[self.paths.ips, self.paths.factors, self.paths.universe])
        journal.append_journal(self.paths.runs, {
            "rebalance_date": date.date().isoformat(), "asof": asof.date().isoformat(),
            "nav": round(nav, 2), "lambda": res.lam, "vol": res.vol, "tracking_error": res.tracking_error,
            "sharpe": res.sharpe, "n_trades": len(plan.trades), "cost_usd": plan.total_cost,
            "compliant": c_post.passed, "git_commit": journal.git_commit()})
        return run

    def monitor(self, date=None) -> ComplianceReport:
        """Compliance of the current simulated holdings (Risk Analyst - Portfolio Monitoring)."""
        u0 = self.universe()
        date = pd.Timestamp(date) if date is not None else u0.prices.index[-1]
        asof = self.prior_trading_day(u0.prices.index, date + pd.Timedelta(days=1))
        u = self.universe(asof)
        pf = self.ensure_portfolio(u)
        cur = pf.weights(u.prices.loc[:asof].iloc[-1].reindex(pf.shares.index)).reindex(u.tickers).fillna(0.0)
        return check_compliance(cur / cur.sum(), u.gics, self.window_returns(u), u.bench_weights,
                                self.bench_vol(u), self.ips)

    def performance(self, end=None):
        """Metrics versus FSPTX (actual NAV) plus Brinson attribution by GICS group."""
        u = self.universe()
        pf = self.ensure_portfolio(u)
        hist = pf_mod.load_history(self.paths.runs)
        end = pd.Timestamp(end) if end is not None else u.prices.index[-1]
        all_px = prices_mod.get_prices(list(hist.columns.union(u.tickers))).loc[:end]
        nav = pf_mod.nav_series(hist, all_px)
        bench = u.fund_prices.loc[nav.index[0]:end]
        m = performance_metrics(nav, bench, u.rf, self.ips["performance"], self.cfg.initial_capital)
        # attribution: periods between holdings changes, benchmark = FSPTX look-through snapshot
        dates = list(hist.index) + ([nav.index[-1]] if nav.index[-1] > hist.index[-1] else [])
        effects, rps, rbs = [], [], []
        for d0, d1 in zip(dates[:-1], dates[1:]):
            p0, p1 = all_px.loc[:d0].iloc[-1], all_px.loc[:d1].iloc[-1]
            sh = hist.loc[d0].reindex(all_px.columns).fillna(0.0)
            wp = sh * p0 / (sh * p0).sum()
            r = p1 / p0 - 1
            effects.append(brinson_period(wp, u.bench_weights, r, u.gics))
            rps.append(float((wp * r).sum())); rbs.append(float((u.bench_weights * r).sum()))
        attribution = link_attribution(effects, rps, rbs) if effects else pd.DataFrame()
        return m, attribution, nav, bench
