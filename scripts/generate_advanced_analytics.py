"""Generate advanced analytics deliverables for the Bluestock MF capstone."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

TRADING_DAYS = 252
RISK_MAP = {
    "Low": ["Low"],
    "Moderate": ["Moderate", "Moderately High"],
    "High": ["High", "Very High"],
}


def read_processed(name: str, dates: list[str] | None = None) -> pd.DataFrame:
    """Read a processed CSV with optional date parsing."""
    return pd.read_csv(PROCESSED_DIR / name, parse_dates=dates or [])


def compute_var_cvar(nav: pd.DataFrame, funds: pd.DataFrame) -> pd.DataFrame:
    """Compute historical VaR and CVaR for each scheme."""
    returns = nav.dropna(subset=["daily_return"]).copy()
    rows = []
    for code, group in returns.groupby("amfi_code"):
        r = group["daily_return"].dropna()
        var_threshold = r.quantile(0.05)
        cvar = r[r <= var_threshold].mean()
        rows.append(
            {
                "amfi_code": int(code),
                "var_95_pct": var_threshold * 100,
                "cvar_95_pct": cvar * 100,
                "return_observations": int(r.count()),
            }
        )
    report = pd.DataFrame(rows).merge(
        funds[["amfi_code", "scheme_name", "fund_house", "category", "sub_category", "risk_category"]],
        on="amfi_code",
        how="left",
    )
    cols = [
        "amfi_code",
        "scheme_name",
        "fund_house",
        "category",
        "sub_category",
        "risk_category",
        "var_95_pct",
        "cvar_95_pct",
        "return_observations",
    ]
    return report[cols].sort_values("var_95_pct")


def compute_rolling_sharpe(nav: pd.DataFrame, funds: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling 90-day Sharpe ratio for every scheme."""
    nav = nav.sort_values(["amfi_code", "nav_date"]).copy()
    nav["rolling_sharpe_90d"] = (
        nav.groupby("amfi_code")["daily_return"]
        .transform(lambda s: (s.rolling(90).mean() / s.rolling(90).std()) * np.sqrt(TRADING_DAYS))
    )
    out = nav[["amfi_code", "nav_date", "rolling_sharpe_90d"]].dropna().merge(
        funds[["amfi_code", "scheme_name", "fund_house", "category", "sub_category"]],
        on="amfi_code",
        how="left",
    )
    return out


def plot_rolling_sharpe(rolling: pd.DataFrame, scorecard: pd.DataFrame) -> Path:
    """Export a rolling Sharpe chart for the five largest funds by AUM."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    top_codes = scorecard.sort_values("aum_crore", ascending=False)["amfi_code"].head(5).tolist()
    view = rolling[rolling["amfi_code"].isin(top_codes)].copy()
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#ffffff")
    for name, group in view.groupby("scheme_name"):
        ax.plot(group["nav_date"], group["rolling_sharpe_90d"], linewidth=1.6, label=name[:34])
    ax.axhline(0, color="#111827", linewidth=0.8, alpha=0.6)
    ax.set_title("Rolling 90-Day Sharpe Ratio - 5 Key Funds", fontsize=13, weight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling Sharpe")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=1, loc="best")
    fig.tight_layout()
    output = REPORTS_DIR / "rolling_sharpe_chart.png"
    fig.savefig(output, dpi=160)
    perf_dir = REPORTS_DIR / "performance_charts"
    perf_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(perf_dir / "rolling_sharpe_chart.png", dpi=160)
    plt.close(fig)
    return output


def compute_cohorts(transactions: pd.DataFrame, funds: pd.DataFrame) -> pd.DataFrame:
    """Build investor cohorts using each investor's first transaction year."""
    tx = transactions.sort_values(["investor_id", "transaction_date"]).copy()
    first_year = tx.groupby("investor_id")["transaction_date"].min().dt.year.rename("first_transaction_year")
    tx = tx.merge(first_year, on="investor_id", how="left").merge(
        funds[["amfi_code", "scheme_name", "fund_house", "category", "sub_category"]],
        on="amfi_code",
        how="left",
    )
    sip = tx[tx["transaction_type"].str.upper().eq("SIP")]
    top_pref = (
        tx.groupby(["first_transaction_year", "scheme_name"])["amount_inr"]
        .sum()
        .reset_index()
        .sort_values(["first_transaction_year", "amount_inr"], ascending=[True, False])
        .groupby("first_transaction_year")
        .head(1)
        .rename(columns={"scheme_name": "top_fund_preference", "amount_inr": "top_fund_amount_inr"})
    )
    cohort = (
        tx.groupby("first_transaction_year")
        .agg(
            investors=("investor_id", "nunique"),
            total_invested_inr=("amount_inr", "sum"),
            transaction_count=("amount_inr", "size"),
        )
        .reset_index()
    )
    avg_sip = sip.groupby("first_transaction_year")["amount_inr"].mean().reset_index(name="average_sip_amount_inr")
    cohort = cohort.merge(avg_sip, on="first_transaction_year", how="left").merge(
        top_pref[["first_transaction_year", "top_fund_preference", "top_fund_amount_inr"]],
        on="first_transaction_year",
        how="left",
    )
    return cohort.sort_values("first_transaction_year")


def compute_sip_continuity(transactions: pd.DataFrame) -> pd.DataFrame:
    """Flag SIP investors with six or more SIPs and average gaps above 35 days."""
    sip = transactions[transactions["transaction_type"].str.upper().eq("SIP")].sort_values(["investor_id", "transaction_date"])
    rows = []
    for investor_id, group in sip.groupby("investor_id"):
        if len(group) < 6:
            continue
        gaps = group["transaction_date"].diff().dt.days.dropna()
        avg_gap = gaps.mean()
        rows.append(
            {
                "investor_id": investor_id,
                "sip_transactions": int(len(group)),
                "average_gap_days": avg_gap,
                "total_sip_amount_inr": group["amount_inr"].sum(),
                "last_sip_date": group["transaction_date"].max().date().isoformat(),
                "continuity_status": "at-risk" if avg_gap > 35 else "regular",
            }
        )
    return pd.DataFrame(rows).sort_values(["continuity_status", "average_gap_days"], ascending=[True, False])


def compute_sector_hhi(holdings: pd.DataFrame, funds: pd.DataFrame) -> pd.DataFrame:
    """Compute sector HHI concentration for all equity funds."""
    equity_codes = funds[funds["category"].str.upper().eq("EQUITY")]["amfi_code"]
    h = holdings[holdings["amfi_code"].isin(equity_codes)].copy()
    sector_weights = h.groupby(["amfi_code", "sector"], as_index=False)["weight_pct"].sum()
    sector_weights["weight_fraction_sq"] = (sector_weights["weight_pct"] / 100) ** 2
    hhi = sector_weights.groupby("amfi_code", as_index=False)["weight_fraction_sq"].sum().rename(columns={"weight_fraction_sq": "sector_hhi"})
    hhi["concentration_level"] = pd.cut(
        hhi["sector_hhi"],
        bins=[-np.inf, 0.10, 0.18, np.inf],
        labels=["Diversified", "Moderate", "Concentrated"],
    )
    return hhi.merge(
        funds[["amfi_code", "scheme_name", "fund_house", "category", "sub_category"]],
        on="amfi_code",
        how="left",
    ).sort_values("sector_hhi", ascending=False)


def build_notebook(insights: list[str]) -> None:
    """Create the advanced analytics notebook with method notes and insights."""
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "# Advanced Analytics\n\nThis notebook documents VaR/CVaR, rolling Sharpe, investor cohorts, SIP continuity, fund recommendation, and sector HHI concentration outputs.",
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## Method Summary\n\n- Historical VaR 95% is the 5th percentile of daily returns.\n- CVaR is the mean of daily returns below the VaR threshold.\n- Rolling Sharpe uses 90 trading-day rolling mean divided by rolling standard deviation, annualised by sqrt(252).\n- Cohorts are grouped by each investor's first transaction year.\n- SIP continuity flags investors with 6+ SIPs and average gap above 35 days as at-risk.\n- Sector HHI is the sum of squared sector weights per equity fund.",
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "import pandas as pd\nfrom pathlib import Path\n\nprocessed = Path('../data/processed')\nreports = Path('../reports')\nvar_cvar = pd.read_csv(processed / 'var_cvar_report.csv')\ncohorts = pd.read_csv(processed / 'cohort_analysis.csv')\nsip = pd.read_csv(processed / 'sip_continuity.csv')\nhhi = pd.read_csv(processed / 'sector_hhi.csv')\nvar_cvar.head()",
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## Five Advanced Insights\n\n" + "\n".join(f"{i}. {text}" for i, text in enumerate(insights, start=1)),
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## Rolling Sharpe Chart\n\nThe exported chart is available at `reports/rolling_sharpe_chart.png` and compares 5 key funds by rolling 90-day Sharpe.",
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "from IPython.display import Image\nImage(filename='../reports/rolling_sharpe_chart.png')",
        },
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (NOTEBOOKS_DIR / "Advanced_Analytics.ipynb").write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    (NOTEBOOKS_DIR / "05_advanced_analytics.ipynb").write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def make_insights(var_cvar: pd.DataFrame, cohort: pd.DataFrame, sip: pd.DataFrame, hhi: pd.DataFrame) -> list[str]:
    """Create five narrative insights from advanced analytics outputs."""
    worst_var = var_cvar.iloc[0]
    biggest_cohort = cohort.sort_values("total_invested_inr", ascending=False).iloc[0]
    at_risk_rate = (sip["continuity_status"].eq("at-risk").mean() * 100) if not sip.empty else 0
    most_concentrated = hhi.iloc[0]
    best_regular = sip[sip["continuity_status"].eq("regular")]["average_gap_days"].mean() if not sip.empty else np.nan
    return [
        f"{worst_var['scheme_name']} has the most negative 95% historical VaR at {worst_var['var_95_pct']:.2f}%, indicating the largest one-day downside threshold.",
        f"The {int(biggest_cohort['first_transaction_year'])} investor cohort has the highest total invested amount at Rs {biggest_cohort['total_invested_inr'] / 1e7:.2f} Cr.",
        f"{at_risk_rate:.1f}% of investors with at least 6 SIP transactions are flagged as at-risk using the average-gap > 35 days rule.",
        f"{most_concentrated['scheme_name']} has the highest sector HHI at {most_concentrated['sector_hhi']:.3f}, meaning its equity portfolio is the most sector-concentrated.",
        f"Regular SIP investors show an average SIP gap of {best_regular:.1f} days, supporting monthly investment continuity.",
    ]


def main() -> None:
    """Generate all advanced analytics CSVs, notebook, chart and insight notes."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    funds = read_processed("dim_fund.csv")
    nav = read_processed("fact_nav.csv", ["nav_date"])
    transactions = read_processed("fact_transactions.csv", ["transaction_date"])
    holdings = read_processed("fact_holdings.csv", ["portfolio_date"])
    scorecard = read_processed("fund_scorecard.csv")

    var_cvar = compute_var_cvar(nav, funds)
    rolling = compute_rolling_sharpe(nav, funds)
    cohort = compute_cohorts(transactions, funds)
    sip_continuity = compute_sip_continuity(transactions)
    sector_hhi = compute_sector_hhi(holdings, funds)
    rolling_chart = plot_rolling_sharpe(rolling, scorecard)

    var_cvar.to_csv(PROCESSED_DIR / "var_cvar_report.csv", index=False)
    rolling.to_csv(PROCESSED_DIR / "rolling_sharpe_90d.csv", index=False)
    cohort.to_csv(PROCESSED_DIR / "cohort_analysis.csv", index=False)
    sip_continuity.to_csv(PROCESSED_DIR / "sip_continuity.csv", index=False)
    sector_hhi.to_csv(PROCESSED_DIR / "sector_hhi.csv", index=False)

    insights = make_insights(var_cvar, cohort, sip_continuity, sector_hhi)
    (REPORTS_DIR / "advanced_insights.md").write_text(
        "# Advanced Analytics Insights\n\n" + "\n".join(f"{i}. {insight}" for i, insight in enumerate(insights, start=1)),
        encoding="utf-8",
    )
    build_notebook(insights)

    print("Advanced analytics generated")
    print(f"- {PROCESSED_DIR / 'var_cvar_report.csv'}")
    print(f"- {PROCESSED_DIR / 'rolling_sharpe_90d.csv'}")
    print(f"- {PROCESSED_DIR / 'cohort_analysis.csv'}")
    print(f"- {PROCESSED_DIR / 'sip_continuity.csv'}")
    print(f"- {PROCESSED_DIR / 'sector_hhi.csv'}")
    print(f"- {rolling_chart}")
    print(f"- {NOTEBOOKS_DIR / 'Advanced_Analytics.ipynb'}")


if __name__ == "__main__":
    main()
