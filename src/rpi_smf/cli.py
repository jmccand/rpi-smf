"""Command line: one entry point for each pipeline step and for the whole rebalance.

    rpi-smf fetch                      refresh prices, snapshot Yahoo estimates
    rpi-smf fit      [--date D]        fit the factor model, print betas / expected returns
    rpi-smf optimize [--date D]        solve the optimisation, print target weights (dry run)
    rpi-smf check    [--date D]        IPS compliance of the current simulated holdings
    rpi-smf trades   [--date D]        trades required to reach the targets (dry run)
    rpi-smf run      [--date D]        full rebalance: writes runs/<D>/ and updates the portfolio
    rpi-smf perf     [--end D]         performance and attribution versus FSPTX
"""
from __future__ import annotations

import argparse
import logging
from datetime import date
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
    s = sub.add_parser("perf")
    s.add_argument("--end", default=None)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    pipe = _build(a)
    if a.cmd == "fetch":
        u = pipe.universe()
        snap = fundamentals.snapshot_yahoo(u.tickers, pd.Timestamp(date.today()))
        path = fundamentals.save_snapshot(snap, pipe.paths.estimates)
        print(f"{len(u.tickers)} names priced through {u.prices.index[-1].date()}; "
              f"Yahoo estimate snapshot (proxy) saved to {path}")
    elif a.cmd == "check":
        _print_compliance("Compliance of current holdings", pipe.monitor(a.date))
    elif a.cmd == "perf":
        m, attr, nav, bench = pipe.performance(a.end)
        for k, v in m.items():
            print(f"{k:32s} {v:,.4f}" if isinstance(v, float) else f"{k:32s} {v}")
        if len(attr):
            print("\nAttribution by group (cumulative active return, Carino-linked)")
            print(attr.assign(total=attr.sum(axis=1)).sort_values("total").to_string())
            print(f"TOTAL {attr.to_numpy().sum():.4%}")
    else:
        r = pipe.rebalance(a.date, commit=(a.cmd == "run"))
        _print_rebalance(r, {"fit": "fit", "optimize": "opt", "trades": "trades", "run": "all"}[a.cmd])


if __name__ == "__main__":
    main()
