"""Simulated portfolio ledger.

State = shares per ticker (no cash: the sleeve is fully invested; trading costs are
paid by scaling all positions pro rata). Persisted as:
  runs/portfolio_state.json   latest shares + date
  runs/holdings_history.csv   (date, ticker, shares) after every change, so daily NAV
                              can be rebuilt between rebalances.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class Portfolio:
    """Holdings (shares by ticker) as of a date."""
    date: pd.Timestamp
    shares: pd.Series

    def value(self, prices: pd.Series) -> float:
        """Market value of the holdings at `prices`."""
        return float((self.shares * prices.reindex(self.shares.index)).sum())

    def weights(self, prices: pd.Series) -> pd.Series:
        """Each position's share of total value at `prices`."""
        v = self.shares * prices.reindex(self.shares.index)
        return v / v.sum()

    def apply(self, trades: pd.DataFrame, date, nav: float, total_cost: float) -> "Portfolio":
        """Execute trades (shares column), then pay `total_cost` by scaling all positions."""
        # add each trade's shares (negative for sells), starting a new line if needed
        sh = self.shares.copy()
        for t, s in trades["shares"].items():
            sh[t] = sh.get(t, 0.0) + s
        # drop closed-out positions and refuse to go short
        sh = sh[sh.abs() > 1e-9]
        if (sh < -1e-9).any():
            raise ValueError("trade would create a short position")
        # no cash balance, so costs shrink every position pro rata
        sh = sh * (1 - total_cost / nav)
        return Portfolio(pd.Timestamp(date), sh)


def initial_portfolio(bench_weights: pd.Series, prices: pd.Series, capital: float,
                      date) -> Portfolio:
    """Starting portfolio: `capital` invested at benchmark weights, at `prices`."""
    w = bench_weights / bench_weights.sum()
    return Portfolio(pd.Timestamp(date), capital * w / prices.reindex(w.index))


def save_state(runs_dir, pf: Portfolio) -> None:
    """Overwrite the latest-state JSON and append this date's holdings to the history."""
    runs = Path(runs_dir)
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "portfolio_state.json").write_text(json.dumps(
        {"date": pf.date.date().isoformat(), "shares": pf.shares.to_dict()}, indent=2))
    # append-only history lets daily NAV be rebuilt between rebalances
    hist = runs / "holdings_history.csv"
    rows = pd.DataFrame({"date": pf.date.date().isoformat(), "ticker": pf.shares.index,
                         "shares": pf.shares.values})
    rows.to_csv(hist, mode="a", header=not hist.exists(), index=False)


def load_state(runs_dir):
    """Latest saved Portfolio, or None before the first run."""
    p = Path(runs_dir) / "portfolio_state.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return Portfolio(pd.Timestamp(d["date"]), pd.Series(d["shares"], dtype=float))


def load_history(runs_dir) -> pd.DataFrame:
    """Holdings after each change: index = date, columns = tickers (shares, NaN->0)."""
    df = pd.read_csv(Path(runs_dir) / "holdings_history.csv", parse_dates=["date"])
    return df.pivot_table(index="date", columns="ticker", values="shares", aggfunc="last").fillna(0.0)


def nav_series(history: pd.DataFrame, prices: pd.DataFrame, end=None) -> pd.Series:
    """Daily NAV from the holdings in force on each date (changes take effect at that date's close)."""
    # forward-fill so holidays/missing prints carry the last price
    px = prices.ffill()
    start = history.index[0]
    px = px.loc[start:end] if end is not None else px.loc[start:]
    # holdings in force each day, then price x shares summed over names
    shares = history.reindex(px.index, method="ffill")
    cols = [c for c in shares.columns if c in px.columns]
    return (shares[cols] * px[cols]).sum(axis=1).rename("nav")
