# Methodology, interpretations and caveats

## Data sources
| Need | Source |
|---|---|
| Constituents, GICS | Capital IQ export `NQ100.xlsx`, converted by `rpi-smf import-nq100` |
| Benchmark weights | proxy: current market cap (Yahoo), renormalised over the investable names; the NDX is modified-cap weighted so published weights differ |
| Stock / QQQM / SPY prices | yfinance adjusted close (total return), cached daily in `data/cache/`. QQQM is the benchmark return series, SPY the CAPM market |
| Risk-free | FRED `DTB3` (3-month T-bill, no key), `^IRX` fallback; monthly rf = (1+y)^(1/12)-1 using the yield at month start |

## Calculations
- **Factors** (amended from IPS 4.1; price-only): `market_beta` (CAPM beta of daily returns vs SPY, 252 days), `momentum` (12-1 months), `volatility` (252-day, expected sign -1). Computed from data on/before `asof`; z-scored cross-sectionally, capped at +/-3, undefined -> 0.
- **Model** (4.2): for each of the last 24 completed month-ends, OLS of next-month excess return on the z-scores; alpha and betas are averaged. A factor is averaged only over months it was defined for >= `min_cross_section` names (all-undefined -> beta 0, logged). Betas contradicting the expected sign are set to 0. `E[R_factor] = alpha + sum beta_k Z_k`; `E[R_MA]` = mean excess return of the last 18 completed months; `E[R] = 0.5 E[R_MA] + 0.5 E[R_factor]` (factor-only where a name lacks 18 months).
- **Optimiser** (4.3): mu = monthly E[R] x 12; C = daily covariance of the last 252 days x 252 with diagonal shrinkage `(1-d)S + d diag(S)`, d = 0.10. cvxpy/CLARABEL. lambda by bisection so ex-ante vol = 100% of the benchmark's trailing-252-day vol (QQQM returns). TE uses the benchmark weights renormalised over the investable universe.
- **30-position minimum** (Section 6) is not part of the 4.3 programme (non-convex). If the solution has fewer than 30 positions the single-name cap is tightened (bisection) to the largest value that gives 30; the cap used is logged.
- **Trade rule**: trade only if |current - target| > 1.5pp. Positions above the 8% cap are always traded, and further deviations are added until the post-trade portfolio passes every hard limit (`reason = compliance`). Cost = 15 bp of traded notional, paid by scaling positions pro rata.
- **Performance** (daily returns, 252/yr): total return, volatility, Sharpe (full window, rf = DTB3/252), max drawdown, beta, TE, IR, 63-day rolling Sharpe, 10-day parametric VaR (carried from the original script). Brinson-Fachler attribution by GICS group per holding period, Carino-linked; the benchmark is the constituent-weight snapshot, which differs from QQQM's actual return (weight proxy, fees).

## IPS interpretations (defaults - amend in config)
- Cap of 8% applied to sleeve weight (sleeve = 100% of portfolio; the universe is all Nasdaq-100 sectors).
- Diagonal-shrinkage intensity (0.10) is not specified in the IPS.
- Expected returns are monthly, covariance daily: both annualised in the optimiser.
- alpha = average of the monthly cross-sectional intercepts. It is a constant across names and does not move weights, but it inflates the printed "expected return" (the sample window is a strong bull market).
- "Positions" count weights >= 0.1%.

## Caveats
- **Look-ahead / survivorship**: the universe is today's Nasdaq-100 constituents applied to the past 24 months. `fit_model` accepts a `universe_fn(date)` hook for point-in-time membership. Names with under 252 days of history (recent IPOs) are excluded.
- **Weights**: benchmark weights are a market-cap proxy (Alphabet is represented by GOOGL alone, carrying the company's total cap), so TE and attribution are measured against an approximation of the index.
- **Factors**: with three price-only factors and a strong bull-market window, a factor's fitted sign can contradict the expected one (volatility has); it is then set to 0 and logged.
- A rebalance is feasible only if the 8% cap, 6% TE, vol and GICS limits can all hold together; if not, the optimiser reports the minimum achievable tracking error rather than relaxing anything.
