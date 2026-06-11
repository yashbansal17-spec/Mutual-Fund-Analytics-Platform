"""Validate chart files and the statistics feeding dashboard visuals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageStat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"


def assert_image_ok(path: Path, min_width: int = 300, min_height: int = 180) -> None:
    """Validate that an exported chart image exists, is large enough and is nonblank."""
    if not path.exists():
        raise AssertionError(f"Missing chart: {path}")
    if path.stat().st_size < 10_000:
        raise AssertionError(f"Chart file is suspiciously small: {path}")
    with Image.open(path) as img:
        if img.width < min_width or img.height < min_height:
            raise AssertionError(f"Chart dimensions are too small: {path} -> {img.size}")
        extrema = ImageStat.Stat(img.convert("L")).extrema[0]
        if extrema[0] == extrema[1]:
            raise AssertionError(f"Chart appears blank or single-colour: {path}")


def validate_chart_files() -> list[str]:
    """Validate exported chart PNG files across report folders."""
    chart_paths = [
        REPORTS_DIR / "rolling_sharpe_chart.png",
        REPORTS_DIR / "performance_charts" / "benchmark_comparison_top5.png",
        REPORTS_DIR / "performance_charts" / "daily_return_distribution.png",
        REPORTS_DIR / "day5_dashboard_pages" / "page_1_industry_overview.png",
        REPORTS_DIR / "day5_dashboard_pages" / "page_2_fund_performance.png",
        REPORTS_DIR / "day5_dashboard_pages" / "page_3_investor_analytics.png",
        REPORTS_DIR / "day5_dashboard_pages" / "page_4_sip_market_trends.png",
    ]
    chart_paths.extend(sorted((REPORTS_DIR / "eda_charts").glob("*.png")))
    for path in chart_paths:
        assert_image_ok(path)
    return [f"{len(chart_paths)} exported PNG charts exist, have valid dimensions, and are nonblank."]


def validate_industry_chart_data() -> list[str]:
    """Validate source data used by industry overview charts."""
    aum = pd.read_csv(PROCESSED_DIR / "fact_aum.csv", parse_dates=["aum_date"])
    sip = pd.read_csv(PROCESSED_DIR / "fact_sip.csv", parse_dates=["month"])
    folios = pd.read_csv(PROCESSED_DIR / "fact_folios.csv", parse_dates=["month"])
    funds = pd.read_csv(PROCESSED_DIR / "dim_fund.csv")

    latest_aum = aum.sort_values("aum_date").groupby("fund_house").tail(1)["aum_crore"].sum()
    trend = aum.groupby("aum_date", as_index=False)["aum_crore"].sum()
    if trend.empty or latest_aum <= 0:
        raise AssertionError("Industry AUM chart data is empty or invalid")
    if int(funds["amfi_code"].nunique()) != 40:
        raise AssertionError("Dashboard scheme count should be 40 for this project dataset")
    if sip["sip_inflow_crore"].max() <= 0 or folios["total_folios_crore"].max() <= 0:
        raise AssertionError("SIP/Folio chart data has invalid non-positive values")
    return ["Industry charts use valid AUM, SIP, folio, and 40-scheme fund-master data."]


def validate_fund_performance_chart_data() -> list[str]:
    """Validate scorecard and NAV data used by fund performance charts."""
    score = pd.read_csv(PROCESSED_DIR / "fund_scorecard.csv")
    nav = pd.read_csv(PROCESSED_DIR / "fact_nav.csv", parse_dates=["nav_date"])
    needed = ["annualised_return_pct", "annualised_volatility_pct", "aum_crore", "composite_score", "sharpe_ratio"]
    if score[needed].replace([np.inf, -np.inf], np.nan).isna().any().any():
        raise AssertionError("Fund performance chart inputs contain NaN or infinite values")
    if not score["composite_score"].between(0, 100).all():
        raise AssertionError("Composite scores are outside 0-100 range")
    top_code = int(score.sort_values("composite_score", ascending=False)["amfi_code"].iloc[0])
    top_nav = nav[(nav["amfi_code"].eq(top_code)) & (nav["is_observed_nav"].eq(1))]
    if top_nav.empty or top_nav["nav"].le(0).any():
        raise AssertionError("NAV drill-through chart data is empty or contains non-positive NAV")
    return ["Fund performance charts use complete scorecard metrics and positive NAV history."]


def validate_investor_chart_data() -> list[str]:
    """Validate investor chart aggregations against transaction totals."""
    tx = pd.read_csv(PROCESSED_DIR / "fact_transactions.csv", parse_dates=["transaction_date"])
    state_sum = tx.groupby("state")["amount_inr"].sum()
    type_sum = tx.groupby("transaction_type")["amount_inr"].sum()
    monthly = tx.groupby(tx["transaction_date"].dt.to_period("M").dt.to_timestamp()).size()
    if abs(state_sum.sum() - tx["amount_inr"].sum()) > 1e-6:
        raise AssertionError("State transaction chart does not reconcile to total amount")
    if abs(type_sum.sum() - tx["amount_inr"].sum()) > 1e-6:
        raise AssertionError("Transaction split pie does not reconcile to total amount")
    if monthly.sum() != len(tx):
        raise AssertionError("Monthly transaction chart count does not reconcile to transaction rows")
    return ["Investor charts reconcile state, transaction-type, and monthly totals to source transactions."]


def validate_sip_market_chart_data() -> list[str]:
    """Validate SIP, benchmark and category inflow chart inputs."""
    sip = pd.read_csv(PROCESSED_DIR / "fact_sip.csv", parse_dates=["month"])
    bench = pd.read_csv(PROCESSED_DIR / "fact_benchmark.csv", parse_dates=["bench_date"])
    inflows = pd.read_csv(PROCESSED_DIR / "fact_category_inflows.csv", parse_dates=["month"])
    if sip["sip_inflow_crore"].le(0).any():
        raise AssertionError("SIP inflow chart contains non-positive values")
    if not {"NIFTY50", "NIFTY100"}.issubset(set(bench["index_name"])):
        raise AssertionError("Benchmark chart data is missing NIFTY50 or NIFTY100")
    if inflows.groupby(["month", "category"])["net_inflow_crore"].sum().empty:
        raise AssertionError("Category inflow heatmap data is empty")
    return ["SIP and market charts use positive SIP data, benchmark data, and category inflow matrix inputs."]


def validate_performance_chart_data() -> list[str]:
    """Validate benchmark comparison chart inputs."""
    score = pd.read_csv(PROCESSED_DIR / "fund_scorecard.csv")
    nav = pd.read_csv(PROCESSED_DIR / "fact_nav.csv", parse_dates=["nav_date"])
    bench = pd.read_csv(PROCESSED_DIR / "fact_benchmark.csv", parse_dates=["bench_date"])
    top5 = score.sort_values("composite_score", ascending=False)["amfi_code"].head(5)
    end_date = nav["nav_date"].max()
    start_date = end_date - pd.DateOffset(years=3)
    selected = nav[nav["amfi_code"].isin(top5) & nav["nav_date"].between(start_date, end_date)]
    if selected["amfi_code"].nunique() != 5:
        raise AssertionError("Benchmark comparison chart does not have NAV data for all top 5 funds")
    bench_view = bench[bench["index_name"].isin(["NIFTY50", "NIFTY100"]) & bench["bench_date"].between(start_date, end_date)]
    if bench_view["index_name"].nunique() != 2:
        raise AssertionError("Benchmark comparison chart is missing NIFTY50/NIFTY100")
    return ["Performance benchmark chart has top 5 fund NAV series plus NIFTY50 and NIFTY100 data."]


def validate_advanced_chart_data() -> list[str]:
    """Validate rolling Sharpe chart inputs."""
    rolling = pd.read_csv(PROCESSED_DIR / "rolling_sharpe_90d.csv", parse_dates=["nav_date"])
    score = pd.read_csv(PROCESSED_DIR / "fund_scorecard.csv")
    top5_aum = set(score.sort_values("aum_crore", ascending=False)["amfi_code"].head(5))
    chart_codes = set(rolling[rolling["amfi_code"].isin(top5_aum)]["amfi_code"].unique())
    if chart_codes != top5_aum:
        raise AssertionError("Rolling Sharpe chart does not contain all 5 key AUM funds")
    if rolling["rolling_sharpe_90d"].replace([np.inf, -np.inf], np.nan).isna().any():
        raise AssertionError("Rolling Sharpe data contains invalid values")
    return ["Rolling Sharpe chart data contains all 5 key AUM funds and valid Sharpe values."]


def main() -> None:
    """Run all chart file and chart-data validation checks."""
    checks: list[str] = []
    checks.extend(validate_chart_files())
    checks.extend(validate_industry_chart_data())
    checks.extend(validate_fund_performance_chart_data())
    checks.extend(validate_investor_chart_data())
    checks.extend(validate_sip_market_chart_data())
    checks.extend(validate_performance_chart_data())
    checks.extend(validate_advanced_chart_data())

    report = "# Chart Validation Report\n\n" + "\n".join(f"- {item}" for item in checks)
    (REPORTS_DIR / "chart_validation_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
