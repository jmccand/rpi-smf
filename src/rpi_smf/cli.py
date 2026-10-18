"""Command line: one entry point for each pipeline step and for the whole rebalance.

    rpi-smf import-nq100 [--xlsx F]    build constituents/GICS/weights from the Capital IQ export
    rpi-smf fetch                      refresh prices
    rpi-smf fit      [--date D]        fit the factor model, print betas / expected returns
    rpi-smf optimize [--date D]        solve the optimisation, print target weights (dry run)
    rpi-smf check    [--date D]        IPS compliance of the current simulated holdings
    rpi-smf trades   [--date D]        trades required to reach the targets (dry run)
    rpi-smf run      [--date D]        full rebalance: writes runs/<D>/ and updates the portfolio
    rpi-smf perf     [--end D]         performance and attribution versus the benchmark
    rpi-smf export   [--date D] [--out F]  post-trade basket CSV from runs/<D>/ (rebalance_post_trade layout)
"""
from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from rpi_smf.data import nasdaq
from rpi_smf.export import export_post_trade
from rpi_smf.pipeline import Paths, Pipeline

pd.set_option("display.width", 160, "display.max_rows", 200, "display.float_format", "{:,.4f}".format)


def _build(a) -> Pipeline:
    paths = Paths()
    if a.gics: paths.gics = Path(a.gics)
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
        print(f"Excluded: {len(r.universe.excluded)} names ({sorted(r.universe.excluded)})")
    if what in ("fit", "all"):
        print("\nFactor betas (monthly, avg of cross-sectional regressions)")
        print(pd.DataFrame({"raw": r.model.beta_raw, "used": r.model.beta,
                            "months": r.model.months_used}).to_string())
        print(f"alpha {r.model.alpha:.4f}")
    if what in ("opt", "all"):
        print(f"\nlambda {o.lam:.4f} ({o.lambda_status}); ex-ante vol {o.vol:.2%} "
              f"(benchmark {o.benchmark_vol:.2%}), TE {o.tracking_error:.2%}, "
              f"exp. excess return {o.expected_return:.2%}, Sharpe {o.sharpe:.2f}, "
              f"turnover {o.turnover:.2%}")
        if o.cap_used + 1e-9 < o.settings["single_name_cap_ips"]:
            print(f"NOTE: single-name cap tightened to {o.cap_used:.2%} (IPS {o.settings['single_name_cap_ips']:.0%}) "
                  f"to hold >= {o.n_positions} positions (IPS minimum 30)")
        t = pd.DataFrame({"target": o.weights, "benchmark": r.universe.bench_weights,
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
    p.add_argument("--gics", help="gics.csv (default data/raw/gics.csv)")
    p.add_argument("--runs", help="output directory (default runs/)")
    p.add_argument("--refresh", action="store_true", help="ignore the daily price cache")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    for name in ("fit", "optimize", "check", "trades", "run"):
        s = sub.add_parser(name)
        s.add_argument("--date", default=date.today().isoformat(), help="rebalance date")
    s = sub.add_parser("import-nq100")
    s.add_argument("--xlsx", default="NQ100.xlsx", help="Capital IQ constituents/GICS export")
    s = sub.add_parser("export")
    s.add_argument("--date", default=date.today().isoformat(), help="rebalance date (runs/<date>/)")
    s.add_argument("--out", default=None, help="output CSV (default 'rebalance_post_trade <date>.csv')")
    s = sub.add_parser("perf")
    s.add_argument("--end", default=None)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    pipe = _build(a)
    if a.cmd == "import-nq100":
        df = nasdaq.import_nq100(a.xlsx, pipe.paths.constituents, pipe.paths.gics)
        print(f"{len(df)} constituents -> {pipe.paths.constituents}, GICS -> {pipe.paths.gics}")
    elif a.cmd == "export":
        out = export_post_trade(pipe.paths.runs / a.date, a.out or f"rebalance_post_trade {a.date}.csv")
        print(f"Post-trade basket -> {out}")
    elif a.cmd == "fetch":
        _, px, _, asof, tickers, _ = pipe._screen()
        print(f"{len(tickers)} names priced through {px.index[-1].date()}")
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
