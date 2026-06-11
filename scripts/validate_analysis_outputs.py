"""Validate key analytics outputs against independent recalculations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
TOL = 1e-9


def assert_close(name: str, actual: float, expected: float, tol: float = TOL) -> None:
    """Raise an assertion error if two numeric values differ beyond tolerance."""
    if pd.isna(actual) and pd.isna(expected):
        return
    if abs(actual - expected) > tol:
        raise AssertionError(f"{name}: actual={actual}, expected={expected}")


def main() -> None:
    """Run independent validation checks for core analytics outputs."""
    funds = pd.read_csv(PROCESSED_DIR / "dim_fund.csv")
    nav = pd.read_csv(PROCESSED_DIR / "fact_nav.csv", parse_dates=["nav_date"])
    tx = pd.read_csv(PROCESSED_DIR / "fact_transactions.csv", parse_dates=["transaction_date"])
    holdings = pd.read_csv(PROCESSED_DIR / "fact_holdings.csv")
    var_cvar = pd.read_csv(PROCESSED_DIR / "var_cvar_report.csv")
    rolling = pd.read_csv(PROCESSED_DIR / "rolling_sharpe_90d.csv", parse_dates=["nav_date"])
    cohort = pd.read_csv(PROCESSED_DIR / "cohort_analysis.csv")
    sip_cont = pd.read_csv(PROCESSED_DIR / "sip_continuity.csv")
    hhi = pd.read_csv(PROCESSED_DIR / "sector_hhi.csv")
    score = pd.read_csv(PROCESSED_DIR / "fund_scorecard.csv")

    checks: list[str] = []

    if funds["amfi_code"].nunique() != 40:
        raise AssertionError("Expected 40 schemes in dim_fund.csv")
    if nav["amfi_code"].nunique() != funds["amfi_code"].nunique():
        raise AssertionError("NAV scheme coverage does not match fund master")
    checks.append("Scheme coverage: 40/40 schemes in fund master and NAV.")

    if len(var_cvar) != funds["amfi_code"].nunique():
        raise AssertionError("VaR/CVaR report does not contain all schemes")
    for code in var_cvar["amfi_code"].sample(min(5, len(var_cvar)), random_state=42):
        ret = nav.loc[nav["amfi_code"].eq(code), "daily_return"].dropna()
        expected_var = ret.quantile(0.05) * 100
        expected_cvar = ret[ret <= ret.quantile(0.05)].mean() * 100
        row = var_cvar[var_cvar["amfi_code"].eq(code)].iloc[0]
        assert_close(f"VaR {code}", row["var_95_pct"], expected_var)
        assert_close(f"CVaR {code}", row["cvar_95_pct"], expected_cvar)
    checks.append("Historical VaR/CVaR: independent samples match 5th percentile and tail mean formulas.")

    code = rolling["amfi_code"].iloc[0]
    sample_date = rolling[rolling["amfi_code"].eq(code)]["nav_date"].iloc[10]
    series = nav[nav["amfi_code"].eq(code)].sort_values("nav_date").set_index("nav_date")["daily_return"]
    window = series.loc[:sample_date].tail(90)
    expected_sharpe = window.mean() / window.std() * np.sqrt(252)
    actual_sharpe = rolling[(rolling["amfi_code"].eq(code)) & (rolling["nav_date"].eq(sample_date))]["rolling_sharpe_90d"].iloc[0]
    assert_close("Rolling Sharpe sample", actual_sharpe, expected_sharpe)
    checks.append("Rolling 90-day Sharpe: independent sample matches rolling mean/std x sqrt(252).")

    tx_sorted = tx.sort_values(["investor_id", "transaction_date"]).copy()
    first_year = tx_sorted.groupby("investor_id")["transaction_date"].min().dt.year.rename("first_transaction_year")
    tx_sorted = tx_sorted.merge(first_year, on="investor_id", how="left")
    expected_total = tx_sorted.groupby("first_transaction_year")["amount_inr"].sum().sort_index()
    actual_total = cohort.set_index("first_transaction_year")["total_invested_inr"].sort_index()
    if not expected_total.equals(actual_total):
        raise AssertionError("Cohort total invested values do not match transactions")
    checks.append("Investor cohort analysis: total invested by first transaction year matches source transactions.")

    sip = tx[tx["transaction_type"].str.upper().eq("SIP")].sort_values(["investor_id", "transaction_date"])
    eligible = sip.groupby("investor_id").filter(lambda g: len(g) >= 6)
    if sip_cont["investor_id"].nunique() != eligible["investor_id"].nunique():
        raise AssertionError("SIP continuity eligible investor count mismatch")
    bad_flags = sip_cont[
        ((sip_cont["average_gap_days"] > 35) & sip_cont["continuity_status"].ne("at-risk"))
        | ((sip_cont["average_gap_days"] <= 35) & sip_cont["continuity_status"].ne("regular"))
    ]
    if not bad_flags.empty:
        raise AssertionError("SIP continuity status flags are inconsistent with 35-day rule")
    checks.append("SIP continuity: 6+ SIP eligibility and >35-day at-risk flag are correct.")

    equity_codes = funds[funds["category"].str.upper().eq("EQUITY")]["amfi_code"]
    sector_weights = holdings[holdings["amfi_code"].isin(equity_codes)].groupby(["amfi_code", "sector"])["weight_pct"].sum()
    expected_hhi = ((sector_weights / 100) ** 2).groupby("amfi_code").sum().sort_index()
    actual_hhi = hhi.set_index("amfi_code")["sector_hhi"].sort_index()
    if not np.allclose(expected_hhi.loc[actual_hhi.index], actual_hhi):
        raise AssertionError("Sector HHI does not match sum of squared sector weights")
    checks.append("Sector HHI: matches sum of squared sector weights for equity funds.")

    if score["composite_score"].between(0, 100).all() is False:
        raise AssertionError("Composite score out of 0-100 range")
    checks.append("Performance scorecard: composite score remains in 0-100 range.")

    report = "# Analysis Validation Report\n\n" + "\n".join(f"- {item}" for item in checks)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "analysis_validation_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
