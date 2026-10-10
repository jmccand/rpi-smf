"""FactSet Estimates API -> point-in-time consensus CSV (see fundamentals.py for the format).

Auth (two paths): (a) API key file -- first line username-serial, second line API key, sent
as HTTP basic auth; read only inside this module, never logged. (b) OAuth 2.0 confidential app, machine authorisation with a private-key JWT, via the
official `fds.sdk.utils` ConfidentialClient. The app-config JSON (Developer Portal
download) is only ever passed by *path* to the SDK; this module never reads, logs or
copies its contents. Keep it out of git (`*-config.json` is ignored).

Data: rolling consensus (/factset-estimates/v2/rolling-consensus) at each perspective date
(`estimate_date`), so history is point-in-time:
  ntm_eps             = EPS   mean, NTMA, relative period 1
  ntm_eps_growth      = EPS   NTMA mean / LTMA mean - 1   (base must be > 0)
  ntm_revenue_growth  = SALES NTMA mean / LTMA mean - 1   (base must be > 0)
Estimate revisions are derived later from the ntm_eps history (factors.py).

Requires `pip install -e '.[factset]'` (Python >= 3.10) and an account entitled to the
FactSet Estimates API.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import date
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import pandas as pd
import yaml

from rpi_smf.config import CONFIG_DIR
from rpi_smf.data.fundamentals import EST_COLUMNS

log = logging.getLogger(__name__)
RowFetcher = Callable[[List[str], List[str], str, int, date, date], List[dict]]


def load_factset_config(path: Optional[Path] = None) -> dict:
    return yaml.safe_load(Path(path or CONFIG_DIR / "factset.yml").read_text())


def resolve_credentials_path(cli_value: Optional[str]) -> str:
    path = cli_value or os.environ.get("FACTSET_CONFIG")
    if not path:
        raise ValueError("FactSet credentials not specified: pass --factset-config <app-config.json> "
                         "or set $FACTSET_CONFIG")
    if not Path(path).is_file():
        raise FileNotFoundError(f"FactSet app-config not found: {path}")
    return path


def resolve_key_file(cli_value: Optional[str]) -> Optional[str]:
    """API-key file path from the CLI or $FACTSET_KEY_FILE; None if neither is given."""
    path = cli_value or os.environ.get("FACTSET_KEY_FILE")
    if path and not Path(path).is_file():
        raise FileNotFoundError(f"FactSet API key file not found: {path}")
    return path


def read_key_file(path: str) -> tuple:
    """(username, api_key) from a two-line file; values are never logged."""
    lines = [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]
    if len(lines) < 2:
        raise ValueError("FactSet API key file needs the username on line 1 and the API key on line 2")
    return lines[0], lines[1]


def sdk_fetcher(credentials_path: Optional[str], cfg: dict, key_file: Optional[str] = None) -> RowFetcher:
    """Row fetcher backed by the FactSet SDK (imported lazily; needs the `factset` extra)."""
    try:
        from fds.sdk.utils.authentication import ConfidentialClient
        import fds.sdk.FactSetEstimates as fse
        from fds.sdk.FactSetEstimates.api import consensus_api
        from fds.sdk.FactSetEstimates.models import (Frequency, Ids, Metrics, Periodicity,
                                                      RelativeFiscalEnd, RelativeFiscalStart,
                                                      RollingConsensusRequest)
    except ImportError as e:
        raise ImportError("FactSet SDKs missing: pip install -e '.[factset]' (Python >= 3.10)") from e

    if key_file:
        user, key = read_key_file(key_file)
        configuration = fse.Configuration(username=user, password=key)
    else:
        configuration = fse.Configuration(fds_oauth_client=ConfidentialClient(credentials_path))

    def one_request(ids, metric, periodicity, relative_period, start, end) -> List[dict]:
        req = RollingConsensusRequest(
            ids=Ids(ids), metrics=Metrics([metric]), start_date=start, end_date=end,
            frequency=Frequency(cfg["frequency"]),
            relative_fiscal_start=RelativeFiscalStart(relative_period),
            relative_fiscal_end=RelativeFiscalEnd(relative_period),
            periodicity=Periodicity(periodicity), currency=cfg["currency"])
        for attempt in range(4):
            try:
                with fse.ApiClient(configuration) as api_client:
                    resp = consensus_api.ConsensusApi(api_client).get_rolling_consensus_for_list(req)
                return [item.to_dict() for item in (resp.data or [])]
            except fse.ApiException as e:
                if e.status == 429 and attempt < 3:       # rate limited: back off and retry
                    time.sleep(2 ** attempt * 2)
                    continue
                raise RuntimeError(f"FactSet request failed (HTTP {e.status}, {e.reason}); check that "
                                   "the account is entitled to the Estimates API") from None
        return []

    def fetch(ids, metrics, periodicity, relative_period, start, end) -> List[dict]:
        # the API accepts a single metric per request
        rows: List[dict] = []
        for m in metrics:
            rows.extend(one_request(ids, m, periodicity, relative_period, start, end))
        return rows

    return fetch


def _batches(items: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(items), n):
        yield items[i:i + n]


def fetch_consensus(tickers: List[str], start: date, end: date, fetch: RowFetcher,
                    cfg: dict) -> pd.DataFrame:
    """Raw consensus rows (one per id/date/metric/periodicity) for every requested ticker."""
    suffix = cfg["region_suffix"]
    metrics = list(cfg["metrics"].values())
    rows: List[dict] = []
    for batch in _batches([t + suffix for t in tickers], int(cfg["batch_size"])):
        for key in ("ntm", "ltm"):
            spec = cfg[key]
            got = fetch(batch, metrics, spec["periodicity"], int(spec["relative_period"]), start, end)
            log.info("%s %s: %d rows for %d ids", key, spec["periodicity"], len(got), len(batch))
            rows.extend(got)
    cols = ["request_id", "estimate_date", "metric", "periodicity", "relative_period", "mean"]
    df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
    return df


def consensus_to_estimates(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Raw rows -> long (date, ticker) frame with EST_COLUMNS."""
    if raw.empty:
        return pd.DataFrame(columns=EST_COLUMNS, index=pd.MultiIndex.from_arrays([[], []],
                                                                                   names=["date", "ticker"]))
    df = raw.copy()
    df["date"] = pd.to_datetime(df["estimate_date"])
    df["ticker"] = df["request_id"].str.replace(cfg["region_suffix"] + "$", "", regex=True)
    df["mean"] = pd.to_numeric(df["mean"], errors="coerce")
    ntm, ltm = cfg["ntm"], cfg["ltm"]
    is_ntm = (df["periodicity"] == ntm["periodicity"]) & (df["relative_period"] == ntm["relative_period"])
    is_ltm = (df["periodicity"] == ltm["periodicity"]) & (df["relative_period"] == ltm["relative_period"])

    def series(metric: str, mask) -> pd.Series:
        s = df[mask & (df["metric"] == metric)].drop_duplicates(["date", "ticker"], keep="last")
        return s.set_index(["date", "ticker"])["mean"]

    eps_n, eps_l = series(cfg["metrics"]["eps"], is_ntm), series(cfg["metrics"]["eps"], is_ltm)
    sal_n, sal_l = series(cfg["metrics"]["sales"], is_ntm), series(cfg["metrics"]["sales"], is_ltm)
    if eps_l.empty or sal_l.empty:
        log.warning("no LTMA consensus returned: growth factors will be undefined. Check "
                    "factset.yml `ltm` (run `rpi-smf pull-factset --probe`).")

    def growth(n: pd.Series, l: pd.Series) -> pd.Series:
        l = l.reindex(n.index)
        return (n / l - 1).where(l > 0)

    out = pd.DataFrame({
        "ntm_eps": eps_n,
        "ntm_revenue_growth": growth(sal_n, sal_l),
        "ntm_eps_growth": growth(eps_n, eps_l),
    }).reindex(columns=EST_COLUMNS)
    return out.sort_index()


def pull_estimates(tickers: List[str], start: date, end: date, credentials_path: Optional[str],
                   cfg: Optional[dict] = None, fetch: Optional[RowFetcher] = None,
                   key_file: Optional[str] = None) -> pd.DataFrame:
    cfg = cfg or load_factset_config()
    fetch = fetch or sdk_fetcher(credentials_path, cfg, key_file)
    return consensus_to_estimates(fetch_consensus(tickers, start, end, fetch, cfg), cfg)


def probe(tickers: List[str], credentials_path: Optional[str], cfg: Optional[dict] = None,
          days: int = 14, key_file: Optional[str] = None) -> pd.DataFrame:
    """Small authenticated request that shows exactly what comes back (for first-time setup)."""
    cfg = cfg or load_factset_config()
    end = date.today()
    raw = fetch_consensus(tickers[:3], end - pd.Timedelta(days=days), end,
                          sdk_fetcher(credentials_path, cfg, key_file), cfg)
    return raw
