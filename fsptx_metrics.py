"""Risk/return metrics for FSPTX, rebuilt from its underlying stock holdings.

The fund's holdings are pulled, each stock's prices are fetched, and the stocks
are combined at the fund's current weights (constant weights, rebalanced daily)
into a synthetic portfolio. Metrics are benchmarked against the S&P 500.

Data source: Yahoo Finance via the `yfinance` package (free, no API key, covers
mutual funds and indices). Prices are cached locally as CSV so repeat runs only
hit the network when the cache is stale.

    pip install yfinance pandas numpy
    python fsptx_metrics.py --start 2025-01-02 --amount 100000

Caveat: Yahoo only exposes a fund's top 10 holdings (~65% of FSPTX). Those are
renormalised to 100%. For the full list, pass --holdings file.csv with columns
`symbol,weight` (e.g. built from the fund's latest SEC N-PORT filing).
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

CACHE_DIR = Path(__file__).parent / ".price_cache"
TRADING_DAYS = 252
SHARPE_WINDOW = 63  # ~3 months of trading days
VAR_HORIZON = 10
MIN_OBS = 20         # min post-start returns before "since inception" stats are used
RISK_WINDOW = 252    # trailing trading days used instead when inception is too recent


def get_prices(ticker: str, force_refresh: bool = False) -> pd.Series:
    """Adjusted close (dividends/distributions reinvested), cached per day."""
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{ticker.replace('^', '_')}.csv"
    fresh = path.exists() and date.fromtimestamp(path.stat().st_mtime) == date.today()
    if fresh and not force_refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)["Close"]

    df = yf.download(ticker, period="max", auto_adjust=True, progress=False)
    if df.empty:
        if path.exists():  # network failure: fall back to stale cache
            return pd.read_csv(path, index_col=0, parse_dates=True)["Close"]
        raise RuntimeError(f"No data returned for {ticker}")
    close = df["Close"].squeeze().dropna().rename("Close")
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.to_frame().to_csv(path)
    return close


def get_holdings(fund: str, csv_path: str | None = None,
                 force_refresh: bool = False) -> pd.Series:
    """Current fund weights (sum to 1), indexed by stock ticker."""
    if csv_path:
        h = pd.read_csv(csv_path).set_index("symbol")["weight"]
    else:
        CACHE_DIR.mkdir(exist_ok=True)
        path = CACHE_DIR / f"{fund}_holdings.csv"
        fresh = path.exists() and date.fromtimestamp(path.stat().st_mtime) == date.today()
        if fresh and not force_refresh:
            h = pd.read_csv(path, index_col=0)["weight"]
        else:
            top = yf.Ticker(fund).funds_data.top_holdings
            if top is None or top.empty:
                if not path.exists():
                    raise RuntimeError(f"No holdings returned for {fund}")
                h = pd.read_csv(path, index_col=0)["weight"]
            else:
                h = top["Holding Percent"].rename("weight")
                h.index.name = "symbol"
                h.to_frame().to_csv(path)
    return h / h.sum()


def build_portfolio(weights: pd.Series, history_start: str, end: str,
                    force_refresh: bool = False) -> tuple[pd.Series, pd.Series]:
    """Synthetic NAV (starts at 1.0) from daily-rebalanced constant weights.

    Holdings without price history back to `history_start` are dropped and the rest
    renormalised. Returns (nav, weights actually used).
    """
    prices = {}
    for t in weights.index:
        try:
            prices[t] = get_prices(t, force_refresh).loc[history_start:end]
        except Exception as e:
            print(f"warning: skipping {t}: {e}")
    df = pd.DataFrame(prices)
    full = df.columns[df.iloc[0].notna() & (df.isna().sum() == 0)]
    dropped = [t for t in weights.index if t not in full]
    if dropped:
        print(f"warning: dropped (incomplete history since {history_start}): {dropped}")
    w = weights.loc[full] / weights.loc[full].sum()
    rets = df[full].pct_change().dropna()
    port = (rets * w).sum(axis=1)
    nav = (1 + port).cumprod()
    nav.loc[df.index[0]] = 1.0
    return nav.sort_index(), w


def get_risk_free_daily() -> pd.Series:
    """Daily risk-free rate from the 13-week T-bill yield (^IRX, quoted in %)."""
    irx = get_prices("^IRX")
    return (irx / 100 / TRADING_DAYS).rename("rf")


def compute_metrics(fund: pd.Series, bench: pd.Series, rf: pd.Series,
                    start: str, amount: float) -> dict:
    """`fund`/`bench` may include history before `start`; see MIN_OBS."""
    idx = fund.index.intersection(bench.index)
    fund, bench = fund.loc[idx], bench.loc[idx]
    start_ts = pd.Timestamp(start)
    if start_ts > fund.index[-1]:
        raise ValueError(f"Start date {start} is after the latest data "
                         f"({fund.index[-1].date()}).")
    if len(idx) < SHARPE_WINDOW + 1:
        raise ValueError("Not enough price history.")

    r_all, rb_all = fund.pct_change().dropna(), bench.pct_change().dropna()
    rf_d = rf.reindex(r_all.index).ffill().fillna(0.0)

    # Inception-based stats use returns after the start date when there are
    # enough of them; otherwise fall back to the trailing RISK_WINDOW days.
    r, rb = r_all[r_all.index > start_ts], rb_all[rb_all.index > start_ts]
    if len(r) >= MIN_OBS:
        window = f"since {start} ({len(r)} days)"
    else:
        r, rb = r_all.iloc[-RISK_WINDOW:], rb_all.iloc[-RISK_WINDOW:]
        window = (f"trailing {len(r)} days (only {len(r_all[r_all.index > start_ts])} "
                  f"days since {start})")

    # Beta: cov(fund, bench) / var(bench), daily returns
    beta = r.cov(rb) / rb.var()

    # Sharpe, 3-month rolling window, annualised; report latest value
    excess = r_all - rf_d
    rolling = excess.rolling(SHARPE_WINDOW)
    sharpe = (rolling.mean() / rolling.std() * np.sqrt(TRADING_DAYS)).iloc[-1]

    # Parametric (normal) 10-day VaR from daily vol over the risk window, zero-drift
    sd_all = r.std()
    var = {c: norm.ppf(c) * sd_all * np.sqrt(VAR_HORIZON) for c in (0.99, 0.95)}

    # Return since initial investment: last price on/before the start date as base
    fund_base, bench_base = fund.asof(start_ts), bench.asof(start_ts)

    return {
        "As of": fund.index[-1].date().isoformat(),
        "Window for beta/VaR/vol": window,
        "Beta (vs benchmark)": beta,
        f"Sharpe ({SHARPE_WINDOW}d rolling, ann.)": sharpe,
        "10-day 99% VaR (%)": var[0.99],
        "10-day 99% VaR ($)": var[0.99] * amount,
        "10-day 95% VaR (%)": var[0.95],
        "10-day 95% VaR ($)": var[0.95] * amount,
        "One week return (5d)": fund.iloc[-1] / fund.iloc[-6] - 1,
        "Return since initial investment": fund.iloc[-1] / fund_base - 1,
        "Realised daily vol (10d)": r_all.iloc[-10:].std(),
        "Realised daily vol (since inception)": sd_all,
        "Benchmark return (since inception)": bench.iloc[-1] / bench_base - 1,
        "Benchmark daily vol (since inception)": rb.std(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ticker", default="FSPTX")
    p.add_argument("--benchmark", default="^GSPC", help="default: S&P 500")
    p.add_argument("--start", default="2025-01-02",
                   help="initial investment date (YYYY-MM-DD)")
    p.add_argument("--end", default=date.today().isoformat(),
                   help="compute metrics as of this date (YYYY-MM-DD), default today")
    p.add_argument("--amount", type=float, default=100_000,
                   help="initial investment, used for $ VaR")
    p.add_argument("--holdings", help="CSV (symbol,weight) overriding Yahoo's top 10")
    p.add_argument("--refresh", action="store_true", help="ignore cache")
    a = p.parse_args()

    weights = get_holdings(a.ticker, a.holdings, a.refresh)
    end = pd.Timestamp(a.end)
    if end < pd.Timestamp(a.start):
        p.error("--end must not be before --start")
    bench = get_prices(a.benchmark, a.refresh).loc[:a.end]
    # Pull enough history before the start date to fill the trailing RISK_WINDOW
    window_start = bench.index[max(0, len(bench) - RISK_WINDOW - 1)]
    history_start = min(pd.Timestamp(a.start), window_start)
    fund, used = build_portfolio(weights, history_start.date().isoformat(),
                                 a.end, a.refresh)
    rf = get_risk_free_daily().loc[:a.end]
    m = compute_metrics(fund, bench, rf, a.start, a.amount)

    print(f"{a.ticker} look-through portfolio vs {a.benchmark}, "
          f"invested {a.start}, as of {a.end} (${a.amount:,.0f})")
    print("Weights: " + ", ".join(f"{t} {x:.1%}" for t, x in used.items()) + "\n")
    for k, v in m.items():
        if isinstance(v, str):
            print(f"{k:42s} {v}")
        elif "($)" in k:
            print(f"{k:42s} ${v:>12,.2f}")
        elif k.startswith(("Beta", "Sharpe")):
            print(f"{k:42s} {v:>12.3f}")
        else:
            print(f"{k:42s} {v:>11.2%}")


if __name__ == "__main__":
    main()
