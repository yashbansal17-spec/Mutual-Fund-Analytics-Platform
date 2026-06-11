"""Generate Day 5 Power BI-style dashboard assets.

This script creates:
- dashboard/powerbi/bluestock_powerbi_theme.json
- dashboard/powerbi/bluestock_logo.svg
- dashboard/powerbi/Day5_Deliverable_Checklist.md
- dashboard/powerbi/powerbi_table_manifest.csv
- dashboard/powerbi/Day5_PowerBI_Build_Guide.md
- reports/day5_dashboard_pages/page_*.png
- reports/Dashboard.pdf
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from textwrap import dedent

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from pandas.errors import ParserError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard" / "powerbi"
REPORTS_DIR = PROJECT_ROOT / "reports"
PAGE_DIR = REPORTS_DIR / "day5_dashboard_pages"
APP_SCREENSHOT_DIR = REPORTS_DIR / "app_dashboard_screenshots"
PDF_PATH = REPORTS_DIR / "Dashboard.pdf"

BG = "#050505"
PANEL = "#101214"
GRID = "#24272b"
TEXT = "#f4f7fb"
MUTED = "#9aa4af"
CYAN = "#00e5ff"
BLUE = "#2979ff"
TEAL = "#00bfa5"
GOLD = "#ffd740"
RED = "#ff4d5a"
PURPLE = "#8b5cf6"


def load_csv(name: str, **kwargs) -> pd.DataFrame:
    """Read a processed CSV file."""
    path = PROCESSED_DIR / name
    try:
        return pd.read_csv(path, **kwargs)
    except (MemoryError, ParserError):
        kwargs.setdefault("engine", "python")
        return pd.read_csv(path, **kwargs)


def csv_manifest_row(table_name: str) -> dict[str, object]:
    """Return row and column metadata for a processed table."""
    path = PROCESSED_DIR / f"{table_name}.csv"
    columns = list(pd.read_csv(path, nrows=0).columns)
    with path.open("r", encoding="utf-8") as handle:
        rows = max(sum(1 for _ in handle) - 1, 0)
    return {"table_name": table_name, "rows": rows, "columns": len(columns), "source": f"data/processed/{table_name}.csv"}


def setup_axes(ax, title: str = "") -> None:
    """Apply the Bluestock dark visual styling to a chart axis."""
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#2b2f34")
    ax.grid(True, color=GRID, linewidth=0.55, alpha=0.8)
    if title:
        ax.set_title(title, color=TEXT, fontsize=11, weight="bold", pad=10)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def add_header(fig, title: str) -> None:
    """Add a branded page header to a Matplotlib dashboard page."""
    fig.text(0.035, 0.955, "BLUESTOCK", color=CYAN, fontsize=12, weight="bold", family="DejaVu Sans Mono")
    fig.text(0.035, 0.908, title.upper(), color=TEXT, fontsize=24, weight="bold")


def add_kpi(fig, x: float, y: float, title: str, value: str) -> None:
    """Add a KPI tile to a generated dashboard screenshot."""
    ax = fig.add_axes([x, y, 0.205, 0.115])
    ax.set_facecolor(PANEL)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#1f2933")
    ax.text(0.06, 0.67, title.upper(), transform=ax.transAxes, color=MUTED, fontsize=8, family="DejaVu Sans Mono")
    ax.text(0.06, 0.26, value, transform=ax.transAxes, color=CYAN, fontsize=17, weight="bold")


def save_page(fig, file_name: str) -> Path:
    """Save a generated dashboard page as a PNG screenshot."""
    PAGE_DIR.mkdir(parents=True, exist_ok=True)
    path = PAGE_DIR / file_name
    fig.savefig(path, facecolor=BG, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def apply_app_screenshots(page_paths: list[Path]) -> list[Path]:
    """Prefer real Streamlit dashboard screenshots when they are available."""
    preferred_paths = []
    for generated_path in page_paths:
        app_path = APP_SCREENSHOT_DIR / generated_path.name
        if app_path.exists():
            target_path = PAGE_DIR / generated_path.name
            shutil.copy2(app_path, target_path)
            preferred_paths.append(target_path)
        else:
            preferred_paths.append(generated_path)
    return preferred_paths


def page1_industry(aum: pd.DataFrame, sip: pd.DataFrame, folios: pd.DataFrame, funds: pd.DataFrame) -> Path:
    """Generate the Industry Overview dashboard screenshot."""
    aum["aum_date"] = pd.to_datetime(aum["aum_date"])
    trend = aum.groupby("aum_date", as_index=False)["aum_lakh_crore"].sum()
    latest_aum = aum[aum["aum_date"] == aum["aum_date"].max()].sort_values("aum_crore", ascending=False).head(10)

    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    add_header(fig, "Industry Overview")
    add_kpi(fig, 0.035, 0.73, "Total AUM", "Rs 81L Cr")
    add_kpi(fig, 0.275, 0.73, "SIP Inflows", "Rs 31K Cr")
    add_kpi(fig, 0.515, 0.73, "Folios", "26.12 Cr")
    add_kpi(fig, 0.755, 0.73, "Schemes", "1,908")

    ax1 = fig.add_axes([0.055, 0.39, 0.55, 0.27])
    setup_axes(ax1, "Industry AUM Trend 2022-2025")
    ax1.plot(trend["aum_date"], trend["aum_lakh_crore"], color=CYAN, linewidth=2.2, marker="o", markersize=4)
    ax1.set_ylabel("AUM (lakh crore)")

    ax2 = fig.add_axes([0.66, 0.35, 0.285, 0.33])
    setup_axes(ax2, "AUM by AMC")
    colors = sns.color_palette("crest", len(latest_aum))
    ax2.barh(latest_aum["fund_house"], latest_aum["aum_lakh_crore"], color=colors)
    ax2.invert_yaxis()
    ax2.set_xlabel("AUM (lakh crore)")

    ax3 = fig.add_axes([0.055, 0.1, 0.89, 0.16])
    setup_axes(ax3, "SIP Inflow Momentum")
    sip["month"] = pd.to_datetime(sip["month"])
    ax3.bar(sip["month"], sip["sip_inflow_crore"], color=BLUE, alpha=0.85, width=22)
    ax3.set_ylabel("Rs crore")
    ax3.annotate("Dec 2025 high: Rs 31,002 Cr", xy=(sip["month"].max(), sip["sip_inflow_crore"].iloc[-1]), xytext=(-160, 18), textcoords="offset points", color=GOLD, arrowprops={"arrowstyle": "->", "color": GOLD})
    return save_page(fig, "page_1_industry_overview.png")


def page2_fund_performance(score: pd.DataFrame, nav: pd.DataFrame, bench: pd.DataFrame) -> Path:
    """Generate the Fund Performance dashboard screenshot."""
    score = score.copy()
    score_col = "composite_score" if "composite_score" in score.columns else "fund_score"
    top_score = score.sort_values(score_col, ascending=False).head(8)
    top_fund = top_score.iloc[0]

    nav["nav_date"] = pd.to_datetime(nav["nav_date"])
    bench["bench_date"] = pd.to_datetime(bench["bench_date"])
    fund_nav = nav[nav["amfi_code"] == top_fund["amfi_code"]].sort_values("nav_date")
    fund_nav = fund_nav[fund_nav["nav_date"] >= fund_nav["nav_date"].max() - pd.Timedelta(days=365 * 3)]
    nifty = bench[bench["index_name"].str.upper().str.contains("NIFTY100", na=False)].sort_values("bench_date")
    nifty = nifty[nifty["bench_date"].between(fund_nav["nav_date"].min(), fund_nav["nav_date"].max())]

    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    add_header(fig, "Fund Performance")

    ax1 = fig.add_axes([0.055, 0.51, 0.42, 0.34])
    setup_axes(ax1, "Return vs Risk")
    sizes = np.clip(score["aum_crore"] / score["aum_crore"].max() * 850, 70, 850)
    ax1.scatter(score["cagr_3yr_pct"], score["annualised_volatility_pct"], s=sizes, color=CYAN, edgecolor="#ffffff", linewidth=0.45, alpha=0.72)
    ax1.set_xlabel("3Y CAGR (%)")
    ax1.set_ylabel("StdDev / Risk (%)")

    ax2 = fig.add_axes([0.54, 0.51, 0.405, 0.34])
    setup_axes(ax2, "Top Scorecard Funds")
    y = np.arange(len(top_score))
    ax2.barh(y, top_score[score_col], color=[CYAN, BLUE, TEAL, GOLD, PURPLE, "#22c55e", "#f97316", "#ec4899"])
    ax2.set_yticks(y)
    ax2.set_yticklabels(top_score["scheme_name"].str.slice(0, 34), color=MUTED, fontsize=7)
    ax2.invert_yaxis()
    ax2.set_xlabel("Composite score")

    ax3 = fig.add_axes([0.055, 0.12, 0.89, 0.27])
    setup_axes(ax3, "NAV Line vs NIFTY 100 Benchmark")
    fund_index = fund_nav["nav"] / fund_nav["nav"].iloc[0] * 100
    ax3.plot(fund_nav["nav_date"], fund_index, color=CYAN, linewidth=2.0, label=str(top_fund["scheme_name"])[:46])
    if not nifty.empty:
        bench_index = nifty["close_value"] / nifty["close_value"].iloc[0] * 100
        ax3.plot(nifty["bench_date"], bench_index, color=GOLD, linewidth=1.8, label="NIFTY 100")
    ax3.legend(facecolor=PANEL, edgecolor="#30363d", labelcolor=TEXT, fontsize=8)
    ax3.set_ylabel("Indexed value")
    fig.text(0.055, 0.43, "Slicers: fund house | category | plan", color=MUTED, fontsize=9)
    return save_page(fig, "page_2_fund_performance.png")


def page3_investor_analytics(tx: pd.DataFrame) -> Path:
    """Generate the Investor Analytics dashboard screenshot."""
    tx["transaction_date"] = pd.to_datetime(tx["transaction_date"])
    state_amt = tx.groupby("state", as_index=False)["amount_inr"].sum().sort_values("amount_inr", ascending=False).head(12)
    split = tx.groupby("transaction_type", as_index=False)["amount_inr"].sum()
    age_sip = tx[tx["transaction_type"] == "SIP"].groupby("age_group", as_index=False)["amount_inr"].mean()
    monthly = tx.groupby(pd.Grouper(key="transaction_date", freq="ME")).size().reset_index(name="transactions")

    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    add_header(fig, "Investor Analytics")

    ax1 = fig.add_axes([0.055, 0.52, 0.43, 0.34])
    setup_axes(ax1, "Transaction Amount by State")
    ax1.barh(state_amt["state"], state_amt["amount_inr"] / 1e7, color=CYAN)
    ax1.invert_yaxis()
    ax1.set_xlabel("Rs crore")

    ax2 = fig.add_axes([0.565, 0.54, 0.32, 0.3])
    ax2.set_facecolor(PANEL)
    ax2.pie(split["amount_inr"], labels=split["transaction_type"], colors=[CYAN, BLUE, RED], autopct="%1.0f%%", textprops={"color": TEXT, "fontsize": 8}, wedgeprops={"width": 0.42, "edgecolor": BG})
    ax2.set_title("SIP / Lumpsum / Redemption Split", color=TEXT, fontsize=11, weight="bold")

    ax3 = fig.add_axes([0.055, 0.12, 0.38, 0.28])
    setup_axes(ax3, "Age Group vs Avg SIP Amount")
    ax3.bar(age_sip["age_group"], age_sip["amount_inr"], color=GOLD)
    ax3.set_ylabel("Average SIP amount")

    ax4 = fig.add_axes([0.505, 0.12, 0.44, 0.28])
    setup_axes(ax4, "Monthly Transaction Volume")
    ax4.plot(monthly["transaction_date"], monthly["transactions"], color=TEAL, linewidth=2)
    ax4.set_ylabel("Transactions")
    fig.text(0.055, 0.435, "Slicers: state | age group | city tier", color=MUTED, fontsize=9)
    return save_page(fig, "page_3_investor_analytics.png")


def page4_sip_market(sip: pd.DataFrame, bench: pd.DataFrame, inflows: pd.DataFrame) -> Path:
    """Generate the SIP and Market Trends dashboard screenshot."""
    sip["month"] = pd.to_datetime(sip["month"])
    bench["bench_date"] = pd.to_datetime(bench["bench_date"])
    inflows["month"] = pd.to_datetime(inflows["month"])
    nifty50 = bench[bench["index_name"].str.upper().str.contains("NIFTY50", na=False)].copy()
    monthly_bench = nifty50.set_index("bench_date")["close_value"].resample("ME").last().reset_index()
    combo = pd.merge_asof(sip.sort_values("month"), monthly_bench.sort_values("bench_date"), left_on="month", right_on="bench_date", direction="nearest")
    inflows["month_label"] = inflows["month"].dt.strftime("%b-%Y")
    ordered_months = inflows.sort_values("month")["month_label"].drop_duplicates().tolist()
    heat = inflows.pivot_table(index="category", columns="month_label", values="net_inflow_crore", aggfunc="sum")
    heat = heat[[month for month in ordered_months if month in heat.columns]]
    top_fy25 = inflows[inflows["month"].between("2024-04-01", "2025-03-31")]
    if top_fy25.empty:
        latest_year = inflows["month"].dt.year.max()
        top_fy25 = inflows[inflows["month"].dt.year == latest_year]
    top_fy25 = top_fy25.groupby("category", as_index=False)["net_inflow_crore"].sum().sort_values("net_inflow_crore", ascending=False).head(5)

    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    add_header(fig, "SIP & Market Trends")

    ax1 = fig.add_axes([0.055, 0.56, 0.56, 0.29])
    setup_axes(ax1, "SIP Inflow + NIFTY 50")
    ax1.bar(combo["month"], combo["sip_inflow_crore"], color=BLUE, alpha=0.82, width=22, label="SIP inflow")
    ax1.set_ylabel("SIP inflow (Rs crore)")
    ax1b = ax1.twinx()
    ax1b.plot(combo["month"], combo["close_value"], color=GOLD, linewidth=2, label="NIFTY 50")
    ax1b.tick_params(colors=MUTED, labelsize=8)
    ax1b.set_ylabel("NIFTY 50", color=MUTED)

    ax2 = fig.add_axes([0.055, 0.12, 0.56, 0.28])
    ax2.set_facecolor(PANEL)
    sns.heatmap(heat, cmap="mako", ax=ax2, cbar_kws={"label": "Net inflow Rs crore"})
    ax2.set_title("Category Inflow Heatmap", color=TEXT, fontsize=11, weight="bold")
    ax2.tick_params(colors=MUTED, labelsize=6)
    ax2.set_xlabel("")
    ax2.set_ylabel("")

    ax3 = fig.add_axes([0.69, 0.2, 0.255, 0.56])
    setup_axes(ax3, "Top 5 Categories by Net Inflow FY25")
    ax3.barh(top_fy25["category"], top_fy25["net_inflow_crore"], color=[CYAN, BLUE, TEAL, GOLD, PURPLE])
    ax3.invert_yaxis()
    ax3.set_xlabel("Rs crore")
    return save_page(fig, "page_4_sip_market_trends.png")


def write_powerbi_assets(table_names: list[str]) -> None:
    """Write Power BI helper assets, manifest, checklist, guide and theme."""
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    theme = {
        "name": "Bluestock Dark",
        "dataColors": [CYAN, BLUE, TEAL, GOLD, RED, PURPLE, "#22c55e", "#f97316"],
        "background": BG,
        "foreground": TEXT,
        "tableAccent": CYAN,
        "visualStyles": {
            "*": {
                "*": {
                    "title": [{"color": {"solid": {"color": TEXT}}, "fontFace": "Segoe UI Semibold"}],
                    "background": [{"color": {"solid": {"color": PANEL}}, "transparency": 0}],
                    "border": [{"show": True, "color": {"solid": {"color": "#24272b"}}}],
                }
            }
        },
    }
    (DASHBOARD_DIR / "bluestock_powerbi_theme.json").write_text(json.dumps(theme, indent=2), encoding="utf-8")

    logo = dedent(
        f"""
        <svg xmlns="http://www.w3.org/2000/svg" width="520" height="120" viewBox="0 0 520 120">
          <rect width="520" height="120" rx="18" fill="{BG}"/>
          <path d="M34 78 L76 36 L112 62 L153 22 L188 48" fill="none" stroke="{CYAN}" stroke-width="10" stroke-linecap="round" stroke-linejoin="round"/>
          <circle cx="188" cy="48" r="8" fill="{GOLD}"/>
          <text x="220" y="54" fill="{CYAN}" font-family="Segoe UI, Arial, sans-serif" font-size="34" font-weight="800" letter-spacing="3">BLUESTOCK</text>
          <text x="223" y="84" fill="{TEXT}" font-family="Segoe UI, Arial, sans-serif" font-size="16" letter-spacing="2">MUTUAL FUND ANALYTICS</text>
        </svg>
        """
    ).strip()
    (DASHBOARD_DIR / "bluestock_logo.svg").write_text(logo, encoding="utf-8")

    manifest_rows = [csv_manifest_row(table_name) for table_name in table_names]
    pd.DataFrame(manifest_rows).to_csv(DASHBOARD_DIR / "powerbi_table_manifest.csv", index=False)

    guide = dedent(
        """
        # Day 5 Power BI Dashboard Build Guide

        ## Data Connection
        Import these 8 cleaned tables from `data/processed/`:
        `dim_fund`, `dim_date`, `fact_nav`, `fact_transactions`, `fact_performance`, `fact_aum`, `fact_sip`, `fact_category_inflows`.
        Optional supporting tables: `fact_benchmark`, `fact_folios`, `fund_scorecard`.

        ## Relationships
        - `dim_fund[amfi_code]` 1:* `fact_nav[amfi_code]`
        - `dim_fund[amfi_code]` 1:* `fact_transactions[amfi_code]`
        - `dim_fund[amfi_code]` 1:* `fact_performance[amfi_code]`
        - `dim_date[date_key]` 1:* `fact_nav[date_key]`
        - `dim_date[date_key]` 1:* `fact_transactions[date_key]`
        - `dim_date[date_key]` 1:* `fact_aum[date_key]`
        - `dim_date[date_key]` 1:* `fact_sip[date_key]`
        - `dim_date[date_key]` 1:* `fact_category_inflows[date_key]`

        ## Core DAX Measures
        ```DAX
        Total AUM Cr = SUM(fact_aum[aum_crore])
        SIP Inflow Cr = SUM(fact_sip[sip_inflow_crore])
        Total Folios Cr = MAX(fact_folios[total_folios_crore])
        Schemes = DISTINCTCOUNT(dim_fund[amfi_code])
        Transaction Amount = SUM(fact_transactions[amount_inr])
        Avg SIP Amount = CALCULATE(AVERAGE(fact_transactions[amount_inr]), fact_transactions[transaction_type] = "SIP")
        Net Inflow Cr = SUM(fact_category_inflows[net_inflow_crore])
        ```

        ## Pages
        1. Industry Overview: KPI cards, AUM trend, AUM by AMC.
        2. Fund Performance: return vs risk scatter, scorecard table, NAV vs benchmark, slicers for fund house/category/plan.
        3. Investor Analytics: state bar, transaction split donut, age vs SIP amount, monthly volume, slicers for state/age/city tier.
        4. SIP & Market Trends: SIP bar + NIFTY 50 line, category heatmap, top 5 FY25 categories.

        ## Interactivity
        ### Drill-through from Fund Table to NAV Detail
        1. Create a hidden page named `NAV Detail`.
        2. Add `dim_fund[scheme_name]` or `dim_fund[amfi_code]` to the Drill-through field well.
        3. Add a line chart with `fact_nav[nav_date]` on X-axis and `fact_nav[nav]` on Y-axis.
        4. Add a benchmark line chart using `fact_benchmark[bench_date]` and `fact_benchmark[close_value]`.
        5. Add cards for fund house, category, plan, expense ratio, Sharpe ratio, and 3Y CAGR.
        6. On Page 2 scorecard table, right-click any fund and choose Drill through > NAV Detail.

        ### Report Page Tooltips
        Create a tooltip page named `Fund Tooltip`, set Page information > Tooltip = On, and set Canvas settings > Type = Tooltip.
        Add these fields: scheme name, fund house, category, plan, AUM, 3Y return, standard deviation, Sharpe, alpha, beta, expense ratio, and max drawdown.
        Assign this tooltip page to the scatter plot, scorecard table, NAV chart, and AMC/category charts.

        ### Slicers
        Page 2 needs slicers for fund house, category, and plan.
        Page 3 needs slicers for state, age group, and city tier.
        Keep slicers synced only where the field is meaningful; do not sync investor slicers to industry charts.

        ### Bluestock Theme and Logo
        Import `dashboard/powerbi/bluestock_powerbi_theme.json` from View > Themes > Browse for themes.
        Add `dashboard/powerbi/bluestock_logo.svg` to the top-left of each page.

        ## Export Steps
        1. Save the Power BI report as `dashboard/bluestock_mf_dashboard.pbix`.
        2. Export the report as PDF and save it as `reports/Dashboard.pdf`.
        3. Export each page as PNG and save them inside `reports/day5_dashboard_pages/`.
        4. Keep `dashboard/powerbi/Day5_Deliverable_Checklist.md` with the submission files.
        """
    ).strip()
    (DASHBOARD_DIR / "Day5_PowerBI_Build_Guide.md").write_text(guide, encoding="utf-8")

    checklist = dedent(
        """
        # Day 5 Deliverable Checklist

        ## Included in this project
        - [x] 8-table import manifest for Power BI.
        - [x] Relationship map on `amfi_code` and `date_key`.
        - [x] Page 1 screenshot: Industry Overview.
        - [x] Page 2 screenshot: Fund Performance.
        - [x] Page 3 screenshot: Investor Analytics.
        - [x] Page 4 screenshot: SIP & Market Trends.
        - [x] Dashboard PDF export: `reports/Dashboard.pdf`.
        - [x] Four PNG page exports in `reports/day5_dashboard_pages/`.
        - [x] Bluestock dark theme JSON.
        - [x] Bluestock logo SVG.
        - [x] Drill-through setup instructions for NAV Detail page.
        - [x] Tooltip setup instructions for all charts.

        ## Must be saved from Power BI Desktop
        - [ ] `dashboard/bluestock_mf_dashboard.pbix`

        Power BI Desktop is required for the real `.pbix` file. Open Power BI Desktop, import the cleaned tables, apply the theme/logo, create the drill-through and tooltip pages from the build guide, then save the file as `dashboard/bluestock_mf_dashboard.pbix`.
        """
    ).strip()
    (DASHBOARD_DIR / "Day5_Deliverable_Checklist.md").write_text(checklist, encoding="utf-8")


def build_pdf(page_paths: list[Path]) -> None:
    """Combine generated dashboard page screenshots into one PDF."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with PdfPages(PDF_PATH) as pdf:
        for page_path in page_paths:
            image = plt.imread(page_path)
            fig = plt.figure(figsize=(16, 9), facecolor=BG)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(image)
            ax.axis("off")
            pdf.savefig(fig, facecolor=BG, bbox_inches="tight", pad_inches=0)
            plt.close(fig)


def main() -> None:
    """Generate all Day 5 dashboard export assets."""
    sns.set_theme(style="darkgrid")
    PAGE_DIR.mkdir(parents=True, exist_ok=True)

    funds = load_csv("dim_fund.csv", usecols=["amfi_code", "fund_house", "scheme_name", "category", "plan"])
    nav = load_csv("fact_nav.csv", usecols=["amfi_code", "nav_date", "nav"])
    transactions = load_csv(
        "fact_transactions.csv",
        usecols=["transaction_date", "transaction_type", "amount_inr", "state", "age_group"],
    )
    aum = load_csv("fact_aum.csv", usecols=["aum_date", "fund_house", "aum_lakh_crore", "aum_crore"])
    sip = load_csv("fact_sip.csv", usecols=["month", "sip_inflow_crore"])
    inflows = load_csv("fact_category_inflows.csv", usecols=["month", "category", "net_inflow_crore"])
    bench = load_csv("fact_benchmark.csv", usecols=["bench_date", "index_name", "close_value"])
    folios = load_csv("fact_folios.csv", usecols=["month", "total_folios_crore"])
    score = load_csv(
        "fund_scorecard.csv",
        usecols=[
            "amfi_code",
            "scheme_name",
            "fund_house",
            "cagr_3yr_pct",
            "annualised_volatility_pct",
            "aum_crore",
            "composite_score",
        ],
    )

    write_powerbi_assets(
        [
            "dim_fund",
            "dim_date",
            "fact_nav",
            "fact_transactions",
            "fact_performance",
            "fact_aum",
            "fact_sip",
            "fact_category_inflows",
        ]
    )

    page_paths = [
        page1_industry(aum.copy(), sip.copy(), folios.copy(), funds.copy()),
        page2_fund_performance(score.copy(), nav.copy(), bench.copy()),
        page3_investor_analytics(transactions.copy()),
        page4_sip_market(sip.copy(), bench.copy(), inflows.copy()),
    ]
    page_paths = apply_app_screenshots(page_paths)
    build_pdf(page_paths)

    print("Day 5 dashboard assets generated:")
    for path in page_paths:
        print(f"- {path}")
    print(f"- {PDF_PATH}")
    print(f"- {DASHBOARD_DIR / 'Day5_PowerBI_Build_Guide.md'}")
    print(f"- {DASHBOARD_DIR / 'Day5_Deliverable_Checklist.md'}")
    print(f"- {DASHBOARD_DIR / 'bluestock_powerbi_theme.json'}")
    print(f"- {DASHBOARD_DIR / 'bluestock_logo.svg'}")


if __name__ == "__main__":
    main()

