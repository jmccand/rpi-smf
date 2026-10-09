"""Command line: one entry point for each pipeline step and for the whole rebalance.

    rpi-smf fetch                      refresh prices, snapshot Yahoo estimates
    rpi-smf fit      [--date D]        fit the factor model, print betas / expected returns
    rpi-smf optimize [--date D]        solve the optimisation, print target weights (dry run)
    rpi-smf check    [--date D]        IPS compliance of the current simulated holdings
    rpi-smf trades   [--date D]        trades required to reach the targets (dry run)
    rpi-smf run      [--date D]        full rebalance: writes runs/<D>/ and updates the portfolio
    rpi-smf perf     [--end D]         performance and attribution versus FSPTX
    rpi-smf pull-factset --factset-config F [--start D] [--end D] [--probe]
                                       point-in-time consensus from the FactSet Estimates API
"""
from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from rpi_smf.data import fundamentals
from rpi_smf.pipeline import Paths, Pipeline

pd.set_option("display.width", 160, "display.max_rows", 200, "display.float_format", "{:,.4f}".format)


def _build(a) -> Pipeline:
    paths = Paths.default(a.holdings)
    if a.gics: paths.gics = Path(a.gics)
    if a.estimates: paths.estimates = Path(a.estimates)
    if a.runs: paths.runs = Path(a.runs)
    return Pipeline(paths, refresh=a.refresh)


def _print_compliance(title, rep):
    print(f"\n{title}: {'PASS' if rep.passed else 'FAIL'}")
    print(rep.table.to_string(index=False))


def _print_rebalance(r, what):
    o = r.opt
    print(f"Rebalance {r.date.date()} (inputs as of {r.asof.date()}), NAV ${r.nav:,.0f}, "
          f"{len(r.universe.tickers)} names")
    if r.universe.excluded:
        print(f"Excluded: {len(r.universe.excluded)} holdings (private/bond/cash/no data)")
    if what in ("fit", "all"):
        print("\nFactor betas (monthly, avg of cross-sectional regressions)")
        print(pd.DataFrame({"raw": r.model.beta_raw, "used": r.model.beta,
                            "months": r.model.months_used}).to_string())
        print(f"alpha {r.model.alpha:.4f}")
    if what in ("opt", "all"):
        print(f"\nlambda {o.lam:.4f} ({o.lambda_status}); ex-ante vol {o.vol:.2%} "
              f"(FSPTX {o.benchmark_vol:.2%}), TE {o.tracking_error:.2%}, "
              f"exp. excess return {o.expected_return:.2%}, Sharpe {o.sharpe:.2f}, "
              f"turnover {o.turnover:.2%}")
        if o.cap_used + 1e-9 < o.settings["single_name_cap_ips"]:
            print(f"NOTE: single-name cap tightened to {o.cap_used:.2%} (IPS {o.settings['single_name_cap_ips']:.0%}) "
                  f"to hold >= {o.n_positions} positions (IPS minimum 30)")
        t = pd.DataFrame({"target": o.weights, "fsptx": r.universe.bench_weights,
                          "gics": r.universe.gics}).sort_values("target", ascending=False)
        print(t[t.target > 1e-4].to_string())
        _print_compliance("Compliance of targets", r.compliance_target)
    if what in ("trades", "all"):
        print(f"\n{len(r.plan.trades)} trades, cost ${r.plan.total_cost:,.2f}, "
              f"turnover {r.plan.turnover:.2%}")
        print(r.plan.trades.to_string())
        _print_compliance("Compliance after trades", r.compliance_post)
    if r.run_dir:
        print(f"\nJournal: {r.run_dir}")


def _pull_factset(a, factset, tickers, creds, pipe):
    if a.probe:
        raw = factset.probe(tickers, creds)
        print(raw.to_string() if len(raw) else "authenticated, but no rows returned")
        print("\ndistinct (metric, periodicity, relative_period):")
        print(raw.groupby(["metric", "periodicity", "relative_period"]).size().to_string()
              if len(raw) else "none")
        return
    est = factset.pull_estimates(tickers, date.fromisoformat(a.start),
                                 date.fromisoformat(a.end), creds)
    out = pipe.paths.estimates
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"factset_{a.end}.csv"
    est.reset_index().to_csv(path, index=False)
    cov = est.groupby(level="ticker").count().reindex(tickers).fillna(0)
    print(f"{len(est)} rows ({est.index.get_level_values('ticker').nunique()}/{len(tickers)} "
          f"names) saved to {path}")
    print("non-missing observations per field:\n" + est.count().to_string())
    print("names with no EPS data:", [t for t in tickers if cov.loc[t, "ntm_eps"] == 0])


def main(argv=None):
    p = argparse.ArgumentParser(prog="rpi-smf", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--holdings", help="Fidelity holdings CSV (default: newest data/raw/fsptx_holdings_*.csv)")
    p.add_argument("--gics", help="gics.csv (default data/raw/gics.csv)")
    p.add_argument("--estimates", help="estimates CSV/dir (default data/raw/estimates)")
    p.add_argument("--runs", help="output directory (default runs/)")
    p.add_argument("--refresh", action="store_true", help="ignore the daily price cache")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    for name in ("fit", "optimize", "check", "trades", "run"):
        s = sub.add_parser(name)
        s.add_argument("--date", default=date.today().isoformat(), help="rebalance date")
    s = sub.add_parser("pull-factset")
    s.add_argument("--factset-config", help="FactSet Developer Portal app-config JSON "
                   "(or set $FACTSET_CONFIG); only its path is used, contents are never read here")
    s.add_argument("--start", default=(date.today() - timedelta(days=30 * 28)).isoformat(),
                   help="first perspective date (default ~28 months back: 24-month regression + revisions)")
    s.add_argument("--end", default=date.today().isoformat())
    s.add_argument("--probe", action="store_true",
                   help="tiny request for 3 tickers; prints raw rows to verify access and field mapping")
    s = sub.add_parser("perf")
    s.add_argument("--end", default=None)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    pipe = _build(a)
    if a.cmd == "fetch":
        listed, px, _, asof, tickers, _ = pipe._screen()
        snap = fundamentals.snapshot_yahoo(tickers, pd.Timestamp(date.today()))
        path = fundamentals.save_snapshot(snap, pipe.paths.estimates)
        print(f"{len(tickers)} names priced through {px.index[-1].date()}; "
              f"Yahoo estimate snapshot (proxy) saved to {path}")
    elif a.cmd == "pull-factset":
        from rpi_smf.data import factset
        try:
            creds = factset.resolve_credentials_path(a.factset_config)
        except (ValueError, FileNotFoundError) as e:
            p.error(str(e))
        tickers = pipe.tickers()
        try:
            _pull_factset(a, factset, tickers, creds, pipe)
        except (RuntimeError, ImportError) as e:
            raise SystemExit(f"error: {e}")
    elif a.cmd == "check":
        _print_compliance("Compliance of current holdings", pipe.monitor(a.date))
    elif a.cmd == "perf":
        m, attr, nav, bench = pipe.performance(a.end)
        for k, v in m.items():
            print(f"{k:32s} {v:,.4f}" if isinstance(v, float) else f"{k:32s} {v}")
        if m["n_days"] < 63:
            print(f"NOTE: only {m['n_days']} daily returns since inception; annualised "
                  "volatility/Sharpe/IR are not meaningful yet")
        if len(attr):
            print("\nAttribution by group (cumulative active return, Carino-linked)")
            print(attr.assign(total=attr.sum(axis=1)).sort_values("total").to_string())
            print(f"TOTAL {attr.to_numpy().sum():.4%}")
    else:
        r = pipe.rebalance(a.date, commit=(a.cmd == "run"))
        _print_rebalance(r, {"fit": "fit", "optimize": "opt", "trades": "trades", "run": "all"}[a.cmd])


if __name__ == "__main__":
    main()
