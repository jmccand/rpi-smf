# rpi-smf

Code for the Student Managed Investment Fund course: a factor-tilted strategy benchmarked
to FSPTX, implementing `IPS_FSPTX.pdf`. The technology-equities sleeve is the whole
portfolio (the algorithmic futures sleeve is out of scope) and starts at FSPTX's own weights.

## Setup

```
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
```

## Weekly workflow

```
rpi-smf fetch                   # refresh prices; snapshot Yahoo estimates (proxy, see below)
rpi-smf fit      --date D       # fit the model, print betas
rpi-smf optimize --date D       # solve for target weights (dry run)
rpi-smf trades   --date D       # trades to place (dry run)
rpi-smf check    --date D       # IPS compliance of current holdings
rpi-smf run      --date D       # full rebalance: writes runs/D/, updates the simulated portfolio
rpi-smf perf                    # return, vol, Sharpe, max drawdown, attribution vs FSPTX
```

`run` is the whole pipeline (IPS Section 4, steps 1-7). Inputs are taken as of the close
of the business day before `D`; fills are simulated at the close of `D`.

## Inputs (all under `data/raw/`, tracked as the audit trail)

| File | What |
|---|---|
| `fsptx_holdings_<date>.csv` | Fidelity fund-holdings dump. The newest one is used. Weights come from the `% of Net Assets` column (market values in these dumps are often corrupted). |
| `gics.csv` | `ticker,sub_industry[,sector,industry_group,industry]` for every investable name. The level used for the 40% cap is `optimizer.gics_level` in `config/ips.yml`. |
| `estimates/*.csv` | Point-in-time consensus, long format: `date,ticker,ntm_eps,ntm_revenue_growth,ntm_eps_growth` (Bloomberg BEst / FactSet export; growth as decimals). `rpi-smf fetch` adds `yahoo_<date>.csv` proxy snapshots. |

Private holdings, bonds and cash are excluded automatically; foreign lines that need a Yahoo
suffix go in `config/universe.yml`. Anything dropped is listed in the run's `manifest.json`.

## Layout

```
config/ips.yml        every IPS parameter (change only via a dated amendment)
config/factors.yml    the six factors: function, expected sign, parameters
src/rpi_smf/
  data/               holdings.py (Fidelity CSV)  prices.py (yfinance, FRED)  fundamentals.py (estimates)  gics.py
  factors.py          factor registry + z-scores              (IPS 4.1)
  model.py            regression, sign rule, MA blend          (IPS 4.2)
  optimize.py         mean-variance QP, lambda calibration     (IPS 4.3)
  compliance.py       limit checks                             (IPS 3, 4.3, 6)
  trades.py           trade list, 1.5pp rule                   (IPS 4 step 6)
  portfolio.py        simulated ledger
  performance.py      metrics + Brinson attribution            (IPS 7)
  journal.py          runs/<date>/ audit trail                 (IPS 4 step 7, 9B)
  pipeline.py cli.py  orchestration
tests/                synthetic data, no network
```

New factor: add a function to `factors.py` decorated with `@register`, then an entry in
`config/factors.yml`. See `docs/methodology.md` for calculations, IPS interpretations and caveats.
