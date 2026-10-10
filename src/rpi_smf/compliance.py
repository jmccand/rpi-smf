"""IPS compliance checks (Sections 3, 4.3 and 6) on a set of sleeve weights.

Works on proposed targets (before trading) or on current holdings (monitoring).
Weights are fractions of the technology sleeve; `sleeve_weight` is the sleeve's share
of total portfolio value (1.0 here: no algorithmic sleeve).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from rpi_smf.optimize import shrunk_covariance

TOL = 1e-6


@dataclass
class ComplianceReport:
    table: pd.DataFrame

    @property
    def passed(self) -> bool:
        return bool((self.table["status"] == "PASS").all())

    @property
    def failures(self) -> pd.DataFrame:
        return self.table[self.table["status"] != "PASS"]


def _row(check: str, value: float, limit: float, ok: bool, detail: str = "") -> dict:
    return {"check": check, "value": value, "limit": limit,
            "status": "PASS" if ok else "FAIL", "detail": detail}


def check_compliance(weights: pd.Series, gics: pd.Series, returns: pd.DataFrame,
                     bench_weights: pd.Series, bench_vol: float, ips: dict,
                     sleeve_weight: float = 1.0) -> ComplianceReport:
    """`ips` is the full config dict (allocation / optimizer / compliance sections)."""
    opt, comp, alloc = ips["optimizer"], ips["compliance"], ips["allocation"]["technology"]
    w = weights[weights.abs() > 0]
    tickers = list(weights.index)
    rows: List[dict] = []

    rows.append(_row("technology sleeve share of portfolio", sleeve_weight, alloc["max"],
                     alloc["min"] - TOL <= sleeve_weight <= alloc["max"] + TOL,
                     f"range {alloc['min']:.0%}-{alloc['max']:.0%}"))
    rows.append(_row("weights sum to 100% of sleeve", float(weights.sum()), 1.0,
                     abs(weights.sum() - 1) < 1e-4))
    rows.append(_row("long-only (min weight)", float(weights.min()), 0.0, weights.min() >= -TOL))
    rows.append(_row("no leverage (gross exposure)", float(weights.abs().sum() * sleeve_weight),
                     1.0, weights.abs().sum() * sleeve_weight <= 1 + 1e-4))
    top = weights.idxmax()
    rows.append(_row("single-company cap", float(weights.max()), opt["single_name_cap"],
                     weights.max() <= opt["single_name_cap"] + TOL, f"largest: {top}"))

    g = weights.groupby(gics.reindex(tickers).fillna("UNCLASSIFIED")).sum()
    rows.append(_row(f"GICS {opt['gics_level']} cap", float(g.max()), opt["gics_cap"],
                     g.max() <= opt["gics_cap"] + TOL, f"largest: {g.idxmax()}"))

    n_pos = int((weights * sleeve_weight >= comp["min_position_weight"]).sum())
    rows.append(_row("minimum positions", float(n_pos), float(comp["min_positions"]),
                     n_pos >= comp["min_positions"]))

    C = shrunk_covariance(returns[tickers], opt["cov_shrinkage"], opt["trading_days"]).to_numpy()
    wv = weights.to_numpy()
    b = bench_weights.reindex(tickers).fillna(0.0)
    b = (b / b.sum()).to_numpy()
    vol, te = float(np.sqrt(wv @ C @ wv)), float(np.sqrt((wv - b) @ C @ (wv - b)))
    rows.append(_row("ex-ante volatility / benchmark volatility", vol / bench_vol,
                     opt["vol_ratio_max"], vol / bench_vol <= opt["vol_ratio_max"] + TOL,
                     f"{vol:.2%} vs benchmark {bench_vol:.2%}"))
    rows.append(_row("ex-ante tracking error", te, opt["te_max"], te <= opt["te_max"] + TOL))
    return ComplianceReport(pd.DataFrame(rows))
