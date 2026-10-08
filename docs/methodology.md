# Methodology, interpretations and caveats

## Data sources
| Need | Source |
|---|---|
| FSPTX holdings | user-supplied Fidelity CSV (`% of Net Assets`, renormalised over investable listed names) |
| Stock / FSPTX prices | yfinance adjusted close (total return), cached daily in `data/cache/` |
| Risk-free | FRED `DTB3` (3-month T-bill, no key), `^IRX` fallback; monthly rf = (1+y)^(1/12)-1 using the yield at month start |
| Consensus estimates | Bloomberg/FactSet export (point-in-time); Yahoo snapshots are a *proxy* (`forwardEps`, next-FY growth) and only accumulate going forward |
| GICS | user-supplied `gics.csv` |

## Calculations
- **Factors** (IPS 4.1) computed from data on/before `asof`; z-scored cross-sectionally, capped at +/-3, undefined -> 0.
- **Model** (4.2): for each of the last 24 completed month-ends, OLS of next-month excess return on the z-scores; alpha and betas are averaged. A factor is averaged only over months it was defined for >= `min_cross_section` names (all-undefined -> beta 0, logged). Betas contradicting the expected sign are set to 0. `E[R_factor] = alpha + sum beta_k Z_k`; `E[R_MA]` = mean excess return of the last 18 completed months; `E[R] = 0.5 E[R_MA] + 0.5 E[R_factor]` (factor-only where a name lacks 18 months).
- **Optimiser** (4.3): mu = monthly E[R] x 12; C = daily covariance of the last 252 days x 252 with diagonal shrinkage `(1-d)S + d diag(S)`, d = 0.10. cvxpy/CLARABEL. lambda by bisection so ex-ante vol = 100% of FSPTX's trailing-252-day vol (actual FSPTX NAV returns). TE uses FSPTX look-through weights renormalised over the investable universe.
- **30-position minimum** (Section 6) is not part of the 4.3 programme (non-convex). If the solution has fewer than 30 positions the single-name cap is tightened (bisection) to the largest value that gives 30; the cap used is logged.
- **Trade rule**: trade only if |current - target| > 1.5pp. Positions above the 8% cap are always traded, and further deviations are added until the post-trade portfolio passes every hard limit (`reason = compliance`). Cost = 15 bp of traded notional, paid by scaling positions pro rata.
- **Performance** (daily returns, 252/yr): total return, volatility, Sharpe (full window, rf = DTB3/252), max drawdown, beta, TE, IR, 63-day rolling Sharpe, 10-day parametric VaR (carried from the original script). Brinson-Fachler attribution by GICS group per holding period, Carino-linked; the benchmark is the FSPTX look-through snapshot, which differs from the actual FSPTX NAV (private holdings, cash, fees).

## IPS interpretations (defaults - amend in config)
- Cap of 8% applied to sleeve weight (sleeve = 100% of portfolio).
- Diagonal-shrinkage intensity (0.10) is not specified in the IPS.
- Expected returns are monthly, covariance daily: both annualised in the optimiser.
- alpha = average of the monthly cross-sectional intercepts. It is a constant across names and does not move weights, but it inflates the printed "expected return" (the sample window is a strong bull market).
- "Positions" count weights >= 0.1%.

## Caveats
- **Look-ahead / survivorship**: the universe is today's FSPTX holdings applied to the past 24 months. Point-in-time membership (SEC N-PORT filings, public quarterly with ~60-day lag) is not wired in; `fit_model` accepts a `universe_fn(date)` hook for it. Delisted names are missing from Yahoo.
- Four of six factors need consensus estimates. Without a Bloomberg/FactSet history export their betas are 0, so only momentum and volatility drive the model until snapshots accumulate.
- A rebalance is feasible only if the 8% cap, 6% TE, vol and GICS limits can all hold together; if not, the optimiser reports the minimum achievable tracking error rather than relaxing anything.
