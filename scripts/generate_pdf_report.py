"""Generate the final 15-20 page PDF report for the Bluestock MF capstone."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
EDA_CHART_DIR = REPORTS_DIR / "eda_charts"
PERFORMANCE_CHART_DIR = REPORTS_DIR / "performance_charts"
DASHBOARD_PAGE_DIR = REPORTS_DIR / "day5_dashboard_pages"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_PATH = REPORTS_DIR / "Bluestock_MF_Final_Report.pdf"


def clean_text(text: object) -> str:
    """Return report-safe text for ReportLab paragraphs."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def para(text: object, style: ParagraphStyle) -> Paragraph:
    """Create a safe ReportLab paragraph."""
    return Paragraph(clean_text(text), style)


def bullet(text: object, style: ParagraphStyle) -> Paragraph:
    """Create a bullet-like paragraph without relying on special glyphs."""
    return para(f"- {text}", style)


def page_footer(canvas: Canvas, doc: SimpleDocTemplate) -> None:
    """Draw consistent footer and page number."""
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(0.55 * inch, 0.35 * inch, "Bluestock Mutual Fund Analytics Capstone")
    canvas.drawRightString(7.7 * inch, 0.35 * inch, f"Page {doc.page}")
    canvas.restoreState()


def add_page_title(story: list, title: str, style: ParagraphStyle) -> None:
    """Append a page heading."""
    story.append(para(title, style))
    story.append(Spacer(1, 0.08 * inch))


def add_table(story: list, rows: list[list[object]], col_widths: list[float] | None = None) -> None:
    """Append a styled table."""
    safe_rows = [[clean_text(cell) for cell in row] for row in rows]
    table = Table(safe_rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 8),
                ("FONTSIZE", (0, 1), (-1, -1), 7),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.12 * inch))


def add_image(story: list, path: Path, title: str, caption: str, body: ParagraphStyle, max_height: float = 4.4) -> None:
    """Append an image with a title and caption if it exists."""
    if not path.exists():
        story.append(bullet(f"Missing image: {path.name}", body))
        return
    story.append(para(f"<b>{title}</b>", body))
    image = Image(str(path))
    max_width = 6.65 * inch
    max_height_points = max_height * inch
    scale = min(max_width / image.imageWidth, max_height_points / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    story.append(image)
    story.append(para(caption, body))
    story.append(Spacer(1, 0.08 * inch))


def load_findings() -> list[str]:
    """Read EDA findings generated during Day 3."""
    path = EDA_CHART_DIR / "eda_findings_summary.md"
    if not path.exists():
        return []
    findings: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped[:2] in {f"{i}." for i in range(1, 10)} or stripped.startswith("10."):
            findings.append(stripped.split(".", 1)[-1].strip())
    return findings[:10]


def load_project_metrics() -> dict[str, object]:
    """Load headline metrics for the report."""
    funds = pd.read_csv(PROCESSED_DIR / "dim_fund.csv")
    aum = pd.read_csv(PROCESSED_DIR / "fact_aum.csv", parse_dates=["aum_date"])
    sip = pd.read_csv(PROCESSED_DIR / "fact_sip.csv", parse_dates=["month"])
    folios = pd.read_csv(PROCESSED_DIR / "fact_folios.csv", parse_dates=["month"])
    score = pd.read_csv(PROCESSED_DIR / "fund_scorecard.csv")
    tx = pd.read_csv(PROCESSED_DIR / "fact_transactions.csv")
    latest_aum = aum.sort_values("aum_date").groupby("fund_house").tail(1)["aum_crore"].sum()
    top_fund = score.sort_values("composite_score", ascending=False).iloc[0]
    return {
        "schemes": funds["amfi_code"].nunique(),
        "fund_houses": funds["fund_house"].nunique(),
        "transactions": len(tx),
        "investors": tx["investor_id"].nunique(),
        "latest_aum_crore": latest_aum,
        "latest_sip_crore": sip.sort_values("month")["sip_inflow_crore"].iloc[-1],
        "latest_folios": folios.sort_values("month")["total_folios_crore"].iloc[-1],
        "top_fund": top_fund["scheme_name"],
        "top_score": top_fund["composite_score"],
    }


def build_report() -> Path:
    """Build and export the final PDF report."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = load_project_metrics()
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=24,
        leading=30,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=16,
    )
    subtitle = ParagraphStyle(
        "Subtitle",
        parent=styles["BodyText"],
        alignment=TA_CENTER,
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#475569"),
        spaceAfter=10,
    )
    heading = ParagraphStyle(
        "Heading",
        parent=styles["Heading1"],
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#0f766e"),
        spaceAfter=10,
    )
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=9.2,
        leading=12.5,
        textColor=colors.HexColor("#111827"),
        spaceAfter=6,
    )

    doc = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title="Bluestock Mutual Fund Analytics Final Report",
    )
    story: list = []

    story.append(Spacer(1, 1.0 * inch))
    story.append(para("Bluestock Mutual Fund Analytics", title))
    story.append(para("Final Capstone Report", subtitle))
    story.append(para("Prepared by: Yash Vardhan Bansal", subtitle))
    story.append(Spacer(1, 0.4 * inch))
    story.append(
        para(
            "A 2022-2026 analytics project covering mutual fund data ingestion, ETL, EDA, risk-return analytics, "
            "advanced investor analysis, dashboarding, validation, and recommendations.",
            subtitle,
        )
    )
    story.append(PageBreak())

    add_page_title(story, "1. Executive Summary", heading)
    for item in [
        f"The project analyses {metrics['schemes']} mutual fund schemes across {metrics['fund_houses']} fund houses.",
        f"The latest tracked industry AUM is approximately Rs {metrics['latest_aum_crore'] / 100000:.2f} lakh crore.",
        f"The latest SIP inflow is Rs {metrics['latest_sip_crore']:,.0f} crore and folios reached {metrics['latest_folios']:.2f} crore.",
        f"The transaction dataset covers {metrics['transactions']:,} records across {metrics['investors']:,} investors.",
        f"The top composite score fund is {metrics['top_fund']} with a score of {metrics['top_score']:.2f}.",
    ]:
        story.append(bullet(item, body))
    story.append(Spacer(1, 0.15 * inch))
    story.append(
        para(
            "The final Streamlit app brings together industry overview, fund performance, investor analytics, SIP and market trends, "
            "advanced analytics, ML prediction, Power BI-style exports, and validation reports. The analysis is designed to support "
            "fund comparison, risk assessment, investor behaviour understanding, and recommendation logic.",
            body,
        )
    )
    story.append(PageBreak())

    add_page_title(story, "2. Data Sources", heading)
    rows = [
        ["Dataset", "Purpose"],
        ["01_fund_master", "Scheme metadata, AMC, category, plan, benchmark, expense ratio and risk category."],
        ["02_nav_history", "Daily NAV series and daily return basis for CAGR, Sharpe, Sortino, drawdown and VaR."],
        ["03_aum_by_fund_house", "AMC-level AUM trend and fund-house leadership analysis."],
        ["04_monthly_sip_inflows", "Monthly SIP inflow, SIP account and SIP AUM analysis."],
        ["05_category_inflows", "Category heatmap and net inflow trend analysis."],
        ["06_industry_folio_count", "Folio growth by equity, debt, hybrid and other segments."],
        ["07_scheme_performance", "Scheme performance, AUM, risk grades and expense ratios."],
        ["08_investor_transactions", "Investor geography, cohorts, SIP continuity and transaction mix."],
        ["09_portfolio_holdings", "Sector allocation and HHI concentration analysis."],
        ["10_benchmark_indices", "NIFTY50, NIFTY100 and other benchmark comparisons."],
    ]
    add_table(story, rows, [1.8 * inch, 4.8 * inch])
    story.append(PageBreak())

    add_page_title(story, "3. ETL Design", heading)
    for item in [
        "Raw files are stored in data/raw and processed outputs are written to data/processed.",
        "Pandas is used for parsing dates, standardising enums, validating numeric columns and removing duplicates.",
        "NAV data is sorted by AMFI code and date, reindexed to a daily calendar and forward-filled for holidays/weekends.",
        "SQLite star schema outputs include dim_fund, dim_date and fact tables for NAV, transactions, performance, AUM, SIP, holdings and benchmarks.",
        "Validation outputs confirm row counts and business-rule checks before analytics scripts run.",
    ]:
        story.append(bullet(item, body))
    add_table(
        story,
        [
            ["Layer", "Output"],
            ["Raw ingestion", "10 CSV files + live NAV fetch"],
            ["Cleaning", "10 cleaned CSVs in data/processed"],
            ["Storage", "SQLite DB plus schema.sql and queries.sql"],
            ["Analytics", "Scorecards, VaR/CVaR, rolling Sharpe, cohorts, HHI and ML predictions"],
            ["Presentation", "Streamlit app, PDF report, dashboard screenshots and notebooks"],
        ],
        [1.7 * inch, 4.9 * inch],
    )
    story.append(PageBreak())

    add_page_title(story, "4. EDA Findings", heading)
    findings = load_findings()
    for idx, finding in enumerate(findings[:8], start=1):
        story.append(para(f"{idx}. {finding}", body))
    add_image(
        story,
        EDA_CHART_DIR / "01_nav_trend_all_40_plotly.png",
        "NAV Trend Across All Schemes",
        "Daily NAV trends show how the full scheme universe moved over the analysis period.",
        body,
    )
    story.append(PageBreak())

    add_page_title(story, "5. EDA Visual Evidence", heading)
    add_image(story, EDA_CHART_DIR / "04_sip_inflow_trend_plotly.png", "SIP Inflow Trend", "SIP inflows rose steadily and reached the project high in Dec 2025.", body, 2.9)
    add_image(story, EDA_CHART_DIR / "05_category_inflow_heatmap_seaborn.png", "Category Inflow Heatmap", "The heatmap highlights category-level inflow concentration by month.", body, 2.9)
    story.append(PageBreak())

    add_page_title(story, "6. Investor EDA", heading)
    add_image(story, EDA_CHART_DIR / "09_sip_amount_by_state.png", "SIP Amount by State", "Geographic analysis identifies the leading states by SIP contribution.", body, 2.9)
    add_image(story, EDA_CHART_DIR / "11_folio_count_growth.png", "Folio Count Growth", "Folio count expansion shows broadening investor participation.", body, 2.9)
    story.append(PageBreak())

    add_page_title(story, "7. Performance Analysis", heading)
    score = pd.read_csv(PROCESSED_DIR / "fund_scorecard.csv").sort_values("composite_score", ascending=False)
    rows = [["Rank", "Scheme", "Fund House", "3Y CAGR", "Sharpe", "Max DD", "Score"]]
    for rank, row in enumerate(score.head(8).itertuples(index=False), start=1):
        rows.append(
            [
                rank,
                row.scheme_name[:42],
                row.fund_house,
                f"{row.cagr_3yr_pct:.2f}%",
                f"{row.sharpe_ratio:.2f}",
                f"{row.max_drawdown_pct:.2f}%",
                f"{row.composite_score:.2f}",
            ]
        )
    add_table(story, rows, [0.35 * inch, 2.35 * inch, 1.25 * inch, 0.65 * inch, 0.55 * inch, 0.6 * inch, 0.55 * inch])
    for item in [
        "CAGR is annualised using observed trading-day history.",
        "Sharpe and Sortino apply a 6.5% annual risk-free-rate proxy.",
        "Alpha and beta are estimated with OLS regression against NIFTY100 daily returns.",
        "Composite score combines 3-year return, Sharpe, alpha, expense ratio and max drawdown ranks.",
    ]:
        story.append(bullet(item, body))
    story.append(PageBreak())

    add_page_title(story, "8. Benchmark and Risk Analysis", heading)
    add_image(story, PERFORMANCE_CHART_DIR / "benchmark_comparison_top5.png", "Top 5 Funds vs NIFTY50/NIFTY100", "Top scorecard funds are compared with benchmark indices on a normalized 3-year basis.", body, 3.1)
    add_image(story, PERFORMANCE_CHART_DIR / "daily_return_distribution.png", "Daily Return Distribution", "Daily return distribution validates reasonableness of fund return behaviour.", body, 2.4)
    story.append(PageBreak())

    add_page_title(story, "9. Advanced Analytics", heading)
    advanced = (REPORTS_DIR / "advanced_insights.md").read_text(encoding="utf-8") if (REPORTS_DIR / "advanced_insights.md").exists() else ""
    for line in advanced.splitlines():
        if line.strip() and line[0].isdigit():
            story.append(para(line, body))
    add_image(story, REPORTS_DIR / "rolling_sharpe_chart.png", "Rolling 90-Day Sharpe", "Rolling Sharpe compares the stability of risk-adjusted returns for five key funds.", body, 3.0)
    story.append(PageBreak())

    add_page_title(story, "10. Dashboard Screenshots - Industry and Fund Performance", heading)
    add_image(story, DASHBOARD_PAGE_DIR / "page_1_industry_overview.png", "Dashboard Page 1: Industry Overview", "Industry KPIs, AUM trend, AMC ranking and SIP momentum.", body, 3.0)
    add_image(story, DASHBOARD_PAGE_DIR / "page_2_fund_performance.png", "Dashboard Page 2: Fund Performance", "Risk-return scatter, scorecard and NAV drill-through view.", body, 3.0)
    story.append(PageBreak())

    add_page_title(story, "11. Dashboard Screenshots - Investor and Market Trends", heading)
    add_image(story, DASHBOARD_PAGE_DIR / "page_3_investor_analytics.png", "Dashboard Page 3: Investor Analytics", "State, transaction mix, age-group and monthly volume views.", body, 3.0)
    add_image(story, DASHBOARD_PAGE_DIR / "page_4_sip_market_trends.png", "Dashboard Page 4: SIP & Market Trends", "SIP inflow, NIFTY50, category heatmap and FY25 category leaders.", body, 3.0)
    story.append(PageBreak())

    add_page_title(story, "12. Validation and Quality Controls", heading)
    for path in [REPORTS_DIR / "analysis_validation_report.md", REPORTS_DIR / "chart_validation_report.md"]:
        if path.exists():
            story.append(para(path.stem.replace("_", " ").title(), body))
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith("- "):
                    story.append(para(line, body))
    story.append(PageBreak())

    add_page_title(story, "13. Limitations", heading)
    for item in [
        "The project uses the supplied capstone dataset and should not be interpreted as live investment advice.",
        "Historical NAV behaviour does not guarantee future fund returns.",
        "AUM data is quarterly, so short date windows may show limited AUM observations.",
        "ML predictions are directional and depend on the engineered metrics available in the sample.",
        "Power BI .pbix export requires Power BI Desktop; this project provides Streamlit and exported dashboard assets.",
    ]:
        story.append(bullet(item, body))
    story.append(PageBreak())

    add_page_title(story, "14. Recommendations", heading)
    for item in [
        "Use composite score with Sharpe, alpha and drawdown rather than ranking funds only by raw return.",
        "Monitor SIP continuity flags to identify investors likely to pause or stop SIPs.",
        "Use sector HHI to identify concentrated funds before recommending them to conservative investors.",
        "Keep NAV and benchmark data refreshed before presenting recent performance.",
        "Use dashboard slicers for fund house, category, plan and risk grade before drawing fund-level conclusions.",
    ]:
        story.append(bullet(item, body))
    story.append(PageBreak())

    add_page_title(story, "15. Conclusion", heading)
    story.append(
        para(
            "The Bluestock Mutual Fund Analytics capstone delivers a full analytics pipeline: raw data ingestion, robust cleaning, "
            "SQLite modelling, EDA, performance analytics, advanced investor analysis, ML comparison, validation reports, and an "
            "interactive Streamlit dashboard. The project is ready for mentor review and can be extended with scheduled ETL, email "
            "reports and hosted dashboard deployment.",
            body,
        )
    )
    story.append(Spacer(1, 0.4 * inch))
    story.append(para("Prepared by Yash Vardhan Bansal", subtitle))

    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    return OUTPUT_PATH


def main() -> None:
    """CLI entrypoint for final PDF report generation."""
    path = build_report()
    print(f"PDF report generated: {path}")


if __name__ == "__main__":
    main()
