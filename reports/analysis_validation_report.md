# Analysis Validation Report

- Scheme coverage: 40/40 schemes in fund master and NAV.
- Historical VaR/CVaR: independent samples match 5th percentile and tail mean formulas.
- Rolling 90-day Sharpe: independent sample matches rolling mean/std x sqrt(252).
- Investor cohort analysis: total invested by first transaction year matches source transactions.
- SIP continuity: 6+ SIP eligibility and >35-day at-risk flag are correct.
- Sector HHI: matches sum of squared sector weights for equity funds.
- Performance scorecard: composite score remains in 0-100 range.