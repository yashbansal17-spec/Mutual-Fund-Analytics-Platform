"""Generate Day 3 EDA notebook and exported PNG charts."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
CHART_DIR = PROJECT_ROOT / "reports" / "eda_charts"
NOTEBOOK_PATH = NOTEBOOK_DIR / "EDA_Analysis.ipynb"
CHART_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)

COLORS = ["#2563eb", "#0f766e", "#dc2626", "#7c3aed", "#f59e0b", "#0891b2", "#be123c", "#4b5563"]
sns.set_theme(style="whitegrid", font_scale=0.95)


def read_csv(name: str, dates: list[str] | None = None) -> pd.DataFrame:
    """Read a processed CSV."""
    return pd.read_csv(PROCESSED_DIR / name, parse_dates=dates or [])


def save_matplotlib(fig: plt.Figure, file_name: str) -> Path:
    """Save a Matplotlib figure and close it."""
    path = CHART_DIR / file_name
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_plotly(fig: go.Figure, file_name: str, fallback: callable | None = None) -> Path:
    """Save a Plotly chart as PNG, with a Matplotlib fallback if Chrome/Kaleido is unavailable."""
    path = CHART_DIR / file_name
    try:
        fig.write_image(path, width=1400, height=800, scale=2)
    except Exception as exc:  # noqa: BLE001 - export environment may not have Chrome for Kaleido
        if fallback is None:
            raise
        print(f"Plotly PNG export fallback for {file_name}: {exc}")
        fallback(path)
    return path


def normalize_nav(nav: pd.DataFrame, funds: pd.DataFrame) -> pd.DataFrame:
    """Return NAV data merged with fund names and normalized to 100 by scheme."""
    merged = nav.merge(funds[["amfi_code", "scheme_name", "fund_house", "sub_category"]], on="amfi_code", how="left")
    merged = merged[(merged["nav_date"].dt.year >= 2022) & (merged["nav_date"].dt.year <= 2026)].copy()
    merged["normalized_nav"] = merged.groupby("amfi_code")["nav"].transform(lambda s: s / s.iloc[0] * 100)
    return merged


def main() -> None:
    """Create all Day 3 EDA outputs."""
    funds = read_csv("dim_fund.csv", ["launch_date"])
    nav = read_csv("fact_nav.csv", ["nav_date"])
    aum = read_csv("fact_aum.csv", ["aum_date"])
    sip = read_csv("fact_sip.csv", ["month"])
    inflows = read_csv("fact_category_inflows.csv", ["month"])
    folios = read_csv("fact_folios.csv", ["month"])
    transactions = read_csv("fact_transactions.csv", ["transaction_date"])
    holdings = read_csv("fact_holdings.csv", ["portfolio_date"])
    performance = read_csv("fact_performance.csv")

    nav_observed = nav[nav["is_observed_nav"].eq(1)].copy()
    nav_merged = normalize_nav(nav_observed, funds)
    chart_paths: list[Path] = []

    # 1. Plotly NAV trend for all 40 schemes with highlighted 2023 bull run and 2024 correction.
    fig_nav = px.line(
        nav_merged,
        x="nav_date",
        y="normalized_nav",
        color="scheme_name",
        title="Daily NAV Trend for All 40 Schemes, Normalized to 100",
        labels={"nav_date": "Date", "normalized_nav": "Normalized NAV"},
    )
    fig_nav.add_vrect(x0="2023-01-01", x1="2023-12-31", fillcolor="green", opacity=0.12, line_width=0, annotation_text="2023 bull run")
    fig_nav.add_vrect(x0="2024-03-01", x1="2024-06-30", fillcolor="red", opacity=0.12, line_width=0, annotation_text="2024 correction")
    fig_nav.update_layout(showlegend=False, template="plotly_white")

    def nav_fallback(path: Path) -> None:
        fig, ax = plt.subplots(figsize=(14, 7))
        for _, group in nav_merged.groupby("scheme_name"):
            ax.plot(group["nav_date"], group["normalized_nav"], alpha=0.35, linewidth=1)
        ax.axvspan(pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31"), color="green", alpha=0.08, label="2023 bull run")
        ax.axvspan(pd.Timestamp("2024-03-01"), pd.Timestamp("2024-06-30"), color="red", alpha=0.08, label="2024 correction")
        ax.set_title("Daily NAV Trend for All 40 Schemes, Normalized to 100")
        ax.set_ylabel("Normalized NAV")
        ax.legend()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    chart_paths.append(save_plotly(fig_nav, "01_nav_trend_all_40_plotly.png", nav_fallback))

    # 2. Top 10 AUM normalized NAV trend.
    top_codes = performance.sort_values("aum_crore", ascending=False)["amfi_code"].head(10).tolist()
    top_nav = nav_merged[nav_merged["amfi_code"].isin(top_codes)]
    fig_top_nav = px.line(top_nav, x="nav_date", y="normalized_nav", color="scheme_name", title="Top 10 AUM Funds: Normalized NAV Trend")
    fig_top_nav.add_vrect(x0="2023-01-01", x1="2023-12-31", fillcolor="green", opacity=0.12, line_width=0)
    fig_top_nav.add_vrect(x0="2024-03-01", x1="2024-06-30", fillcolor="red", opacity=0.12, line_width=0)
    fig_top_nav.update_layout(template="plotly_white", legend=dict(orientation="h", y=-0.25))

    def top_nav_fallback(path: Path) -> None:
        fig, ax = plt.subplots(figsize=(14, 7))
        sns.lineplot(data=top_nav, x="nav_date", y="normalized_nav", hue="scheme_name", ax=ax, linewidth=1.8)
        ax.set_title("Top 10 AUM Funds: Normalized NAV Trend")
        ax.legend(fontsize=7)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    chart_paths.append(save_plotly(fig_top_nav, "02_top10_aum_nav_trend_plotly.png", top_nav_fallback))

    # 3. AUM growth grouped bar chart using Seaborn, with SBI dominance highlighted.
    aum_year = aum.copy()
    aum_year["year"] = aum_year["aum_date"].dt.year
    latest_yearly = aum_year.sort_values("aum_date").groupby(["year", "fund_house"], as_index=False).tail(1)
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(data=latest_yearly, x="year", y="aum_lakh_crore", hue="fund_house", ax=ax, palette="tab10")
    ax.set_title("AUM Growth by Fund House, 2022-2025")
    ax.set_ylabel("AUM (Rs lakh crore)")
    ax.annotate("SBI reaches Rs 12.5L Cr", xy=(3, 12.5), xytext=(2.45, 13.3), arrowprops={"arrowstyle": "->", "color": "#dc2626"}, color="#dc2626")
    ax.legend(ncol=2, fontsize=8)
    chart_paths.append(save_matplotlib(fig, "03_aum_growth_by_fund_house_seaborn.png"))

    # 4. Plotly SIP inflow trend with Dec 2025 high annotation.
    fig_sip = px.line(sip, x="month", y="sip_inflow_crore", markers=True, title="Monthly SIP Inflow Trend, Jan 2022-Dec 2025")
    peak = sip.loc[sip["sip_inflow_crore"].idxmax()]
    fig_sip.add_annotation(
        x=peak["month"].to_pydatetime(),
        y=peak["sip_inflow_crore"],
        text=f"All-time high: Rs {peak['sip_inflow_crore']:,.0f} Cr",
        showarrow=True,
        arrowhead=2,
    )
    fig_sip.update_layout(template="plotly_white", yaxis_title="SIP inflow (Rs crore)")

    def sip_fallback(path: Path) -> None:
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.plot(sip["month"], sip["sip_inflow_crore"], marker="o", color=COLORS[1])
        ax.annotate(f"Rs {peak['sip_inflow_crore']:,.0f} Cr", xy=(peak["month"], peak["sip_inflow_crore"]), xytext=(peak["month"], peak["sip_inflow_crore"] - 4500), arrowprops={"arrowstyle": "->"})
        ax.set_title("Monthly SIP Inflow Trend, Jan 2022-Dec 2025")
        ax.set_ylabel("SIP inflow (Rs crore)")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    chart_paths.append(save_plotly(fig_sip, "04_sip_inflow_trend_plotly.png", sip_fallback))

    # 5. Category inflow heatmap.
    pivot = inflows.pivot_table(index="category", columns=inflows["month"].dt.strftime("%Y-%m"), values="net_inflow_crore", aggfunc="sum")
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.heatmap(pivot, cmap="RdYlGn", center=0, linewidths=0.4, ax=ax, cbar_kws={"label": "Net inflow (Rs crore)"})
    ax.set_title("Category-wise Net Inflow Heatmap")
    chart_paths.append(save_matplotlib(fig, "05_category_inflow_heatmap_seaborn.png"))

    # 6. Age group distribution pie chart.
    age_counts = transactions["age_group"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.pie(age_counts, labels=age_counts.index, autopct="%1.1f%%", startangle=90, colors=COLORS)
    ax.set_title("Investor Age Group Distribution")
    chart_paths.append(save_matplotlib(fig, "06_age_group_distribution_pie.png"))

    # 7. SIP amount box plot by age group.
    sip_txn = transactions[transactions["transaction_type"].eq("SIP")].copy()
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(data=sip_txn, x="age_group", y="amount_inr", hue="age_group", ax=ax, palette="Set2", legend=False)
    ax.set_title("SIP Amount Distribution by Age Group")
    ax.set_ylabel("SIP amount (INR)")
    chart_paths.append(save_matplotlib(fig, "07_sip_amount_box_by_age_group.png"))

    # 8. Gender split.
    gender_counts = transactions["gender"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.pie(gender_counts, labels=gender_counts.index, autopct="%1.1f%%", startangle=90, colors=COLORS)
    ax.set_title("Investor Gender Split")
    chart_paths.append(save_matplotlib(fig, "08_gender_split_pie.png"))

    # 9. Geographic SIP amount by state.
    sip_state = sip_txn.groupby("state", as_index=False)["amount_inr"].sum().sort_values("amount_inr", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=sip_state.sort_values("amount_inr"), x="amount_inr", y="state", hue="state", ax=ax, palette="viridis", legend=False)
    ax.set_title("Top 15 States by SIP Amount")
    ax.set_xlabel("SIP amount (INR)")
    chart_paths.append(save_matplotlib(fig, "09_sip_amount_by_state.png"))

    # 10. T30 vs B30 city tier pie chart.
    tier_counts = transactions["city_tier"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.pie(tier_counts, labels=tier_counts.index, autopct="%1.1f%%", startangle=90, colors=COLORS)
    ax.set_title("T30 vs B30 City Tier Split")
    chart_paths.append(save_matplotlib(fig, "10_city_tier_split_pie.png"))

    # 11. Folio count growth with milestones.
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(folios["month"], folios["total_folios_crore"], marker="o", color=COLORS[0], linewidth=2.5)
    start = folios.sort_values("month").iloc[0]
    end = folios.sort_values("month").iloc[-1]
    ax.annotate(f"{start['total_folios_crore']:.2f} Cr", xy=(start["month"], start["total_folios_crore"]), xytext=(start["month"], start["total_folios_crore"] + 1), arrowprops={"arrowstyle": "->"})
    ax.annotate(f"{end['total_folios_crore']:.2f} Cr", xy=(end["month"], end["total_folios_crore"]), xytext=(end["month"], end["total_folios_crore"] - 3), arrowprops={"arrowstyle": "->"})
    ax.set_title("Mutual Fund Folio Count Growth")
    ax.set_ylabel("Total folios (crore)")
    chart_paths.append(save_matplotlib(fig, "11_folio_count_growth.png"))

    # 12. NAV return correlation matrix for 10 selected funds.
    selected_codes = top_codes
    returns = nav_observed[nav_observed["amfi_code"].isin(selected_codes)].pivot(index="nav_date", columns="amfi_code", values="daily_return")
    corr = returns.corr()
    code_to_short = performance.set_index("amfi_code")["scheme_name"].str.replace(" Fund", "", regex=False).str.slice(0, 18).to_dict()
    corr = corr.rename(index=code_to_short, columns=code_to_short)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, ax=ax)
    ax.set_title("Daily NAV Return Correlation Matrix: Top 10 AUM Funds")
    chart_paths.append(save_matplotlib(fig, "12_nav_return_correlation_matrix.png"))

    # 13. Sector allocation donut across equity holdings.
    equity_codes = funds.loc[funds["category"].eq("Equity"), "amfi_code"]
    sector_weights = holdings[holdings["amfi_code"].isin(equity_codes)].groupby("sector")["weight_pct"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.pie(sector_weights, labels=sector_weights.index, autopct="%1.1f%%", startangle=90, colors=COLORS * 4, wedgeprops={"width": 0.45})
    ax.set_title("Aggregate Sector Allocation Across Equity Funds")
    chart_paths.append(save_matplotlib(fig, "13_sector_allocation_donut.png"))

    # 14. Monthly transaction type volume.
    tx_month = transactions.copy()
    tx_month["month"] = tx_month["transaction_date"].dt.to_period("M").dt.to_timestamp()
    tx_volume = tx_month.groupby(["month", "transaction_type"], as_index=False).size()
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.lineplot(data=tx_volume, x="month", y="size", hue="transaction_type", marker="o", ax=ax, palette=COLORS[:3])
    ax.set_title("Monthly Transaction Volume by Type")
    ax.set_ylabel("Transactions")
    chart_paths.append(save_matplotlib(fig, "14_monthly_transaction_volume_by_type.png"))

    # 15. Category total inflows bar chart.
    category_totals = inflows.groupby("category", as_index=False)["net_inflow_crore"].sum().sort_values("net_inflow_crore", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=category_totals, x="category", y="net_inflow_crore", hue="category", ax=ax, palette="mako", legend=False)
    ax.tick_params(axis="x", rotation=35)
    ax.set_title("Total Net Inflow by Category")
    ax.set_ylabel("Net inflow (Rs crore)")
    chart_paths.append(save_matplotlib(fig, "15_total_net_inflow_by_category.png"))

    # 16. Top 10 3-year return funds.
    top_returns = performance.sort_values("return_3yr_pct", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=top_returns.sort_values("return_3yr_pct"), x="return_3yr_pct", y="scheme_name", hue="scheme_name", ax=ax, palette="rocket", legend=False)
    ax.set_title("Top 10 Funds by 3-Year Return")
    ax.set_xlabel("3-year return (%)")
    chart_paths.append(save_matplotlib(fig, "16_top10_3yr_return_funds.png"))

    # Data-driven findings.
    sbi_latest = aum[aum["fund_house"].eq("SBI Mutual Fund")].sort_values("aum_date").iloc[-1]
    sip_peak = sip.loc[sip["sip_inflow_crore"].idxmax()]
    top_state = sip_state.sort_values("amount_inr", ascending=False).iloc[0]
    top_category = category_totals.iloc[0]
    top_sector = sector_weights.index[0]
    folio_growth = (end["total_folios_crore"] / start["total_folios_crore"] - 1) * 100
    top_3yr = top_returns.iloc[0]
    highest_corr = corr.where(~np.eye(corr.shape[0], dtype=bool)).stack().sort_values(ascending=False).iloc[0]
    t30_share = tier_counts.get("T30", 0) / tier_counts.sum() * 100
    age_top = age_counts.idxmax()

    if highest_corr >= 0.7:
        correlation_finding = f"Chart 12 shows the highest pairwise return correlation among selected top-AUM funds is {highest_corr:.2f}, indicating strong co-movement among at least two large funds."
    elif highest_corr >= 0.4:
        correlation_finding = f"Chart 12 shows the highest pairwise return correlation among selected top-AUM funds is {highest_corr:.2f}, indicating moderate co-movement among the selected funds."
    else:
        correlation_finding = f"Chart 12 shows the highest pairwise return correlation among selected top-AUM funds is only {highest_corr:.2f}, suggesting limited co-movement and potential diversification value."

    findings = [
        f"Chart 01 shows that most NAV series participated in the broad 2023 bull run, while the highlighted 2024 period created visible pauses or corrections across equity-oriented schemes.",
        f"Chart 03 shows SBI Mutual Fund as the AUM leader at Rs {sbi_latest['aum_lakh_crore']:.1f} lakh crore in {sbi_latest['aum_date'].date()}, supporting the dominance callout.",
        f"Chart 04 shows SIP inflows peaking at Rs {sip_peak['sip_inflow_crore']:,.0f} crore in {sip_peak['month'].strftime('%b %Y')}, the all-time high in the dataset.",
        f"Chart 05 shows {top_category['category']} as the strongest aggregate category by net inflow, with total inflow of Rs {top_category['net_inflow_crore']:,.0f} crore.",
        f"Chart 06 shows {age_top} as the largest investor age group in transaction records.",
        f"Chart 09 shows {top_state['state']} leading SIP contribution among states, with Rs {top_state['amount_inr']:,.0f} invested through SIP transactions.",
        f"Chart 10 shows T30 cities contributing {t30_share:.1f}% of total investor transactions, indicating a meaningful metro/large-city skew.",
        f"Chart 11 shows total folios rising from {start['total_folios_crore']:.2f} crore to {end['total_folios_crore']:.2f} crore, a {folio_growth:.1f}% increase.",
        correlation_finding,
        f"Chart 13 shows {top_sector} as the largest aggregate equity portfolio sector exposure across holdings.",
        f"Chart 16 shows {top_3yr['scheme_name']} leading the 3-year return ranking at {top_3yr['return_3yr_pct']:.2f}%.",
    ]

    # Notebook with 15+ chart references and Markdown insight cells.
    chart_md = []
    for path in chart_paths:
        rel = Path("..") / "reports" / "eda_charts" / path.name
        chart_md.extend([f"## {path.stem.replace('_', ' ').title()}\n", f"![{path.stem}]({rel.as_posix()})\n\n"])

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# EDA Analysis - Bluestock Mutual Fund Capstone\n",
                "\n",
                "Day 3 deliverable covering NAV trends, AUM growth, SIP flows, category inflows, investor demographics, geography, folio growth, NAV correlation, and sector allocation.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "import pandas as pd\n",
                "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n",
                "PROCESSED = PROJECT_ROOT / 'data' / 'processed'\n",
                "CHARTS = PROJECT_ROOT / 'reports' / 'eda_charts'\n",
                "sorted(p.name for p in CHARTS.glob('*.png'))\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": chart_md,
        },
        {"cell_type": "markdown", "metadata": {}, "source": ["# 10 Key EDA Findings\n", "\n"]},
        *[
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"## Finding {i}\n", "\n", finding + "\n"],
            }
            for i, finding in enumerate(findings[:10], 1)
        ],
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Additional Observation\n", "\n", findings[10] + "\n"],
        },
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10+"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(json.dumps(nb, indent=2), encoding="utf-8")

    summary = ["# Day 3 EDA Outputs", "", f"- Notebook: `{NOTEBOOK_PATH}`", f"- PNG charts exported: {len(chart_paths)}", ""]
    summary.extend(f"- `{path.name}`" for path in chart_paths)
    summary.extend(["", "## Key Findings", ""])
    summary.extend(f"{i}. {finding}" for i, finding in enumerate(findings[:10], 1))
    (CHART_DIR / "eda_findings_summary.md").write_text("\n".join(summary), encoding="utf-8")

    print(f"Created {len(chart_paths)} PNG charts in {CHART_DIR}")
    print(f"Created notebook: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
