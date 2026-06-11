"""Insert real Streamlit dashboard screenshots inline into the enhanced PDF report.

The original enhanced report has dashboard description pages at pages 12 and 13.
This script replaces those two pages with polished versions where each dashboard
screenshot appears immediately after its matching description.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph



PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPORT = Path.home() / "Downloads" / "Bluestock_MF_Enhanced_Report.pdf"
REPORTS_DIR = PROJECT_ROOT / "reports"
SCREENSHOT_DIR = REPORTS_DIR / "app_dashboard_screenshots"
INLINE_PATH = REPORTS_DIR / "_dashboard_inline_pages.pdf"
OUTPUT_REPORT = REPORTS_DIR / "Bluestock_MF_Enhanced_Report.pdf"
REPLACE_PAGE_START = 12  # 1-based page number in the source report.

BLUE = colors.HexColor("#0f4c99")
TEXT = colors.HexColor("#000000")
MUTED = colors.HexColor("#4b5563")

DASHBOARD_INLINE_PAGES = [
    {
        "section": "10.  Dashboard Screenshots ? Industry and Fund Performance",
        "page_no": 12,
        "items": [
            {
                "heading": "Dashboard Page 1: Industry Overview",
                "body": "The Industry Overview dashboard presents headline KPIs (Total AUM, SIP Inflows, Folios, and Scheme Count), industry AUM trend from 2022 to 2025, AMC-wise AUM ranking, and the SIP inflow momentum chart. The dashboard is designed for executive-level consumption and provides at-a-glance industry context before any fund-level drill-down.",
                "image": "page_1_industry_overview.png",
            },
            {
                "heading": "Dashboard Page 2: Fund Performance",
                "body": "The Fund Performance page features a risk-return scatter plot, the composite scorecard leaderboard, and a NAV drill-through view with benchmark overlay. The scatter plot axes ? 3-Year CAGR against standard deviation ? allow rapid identification of funds with superior risk-adjusted profiles.",
                "image": "page_2_fund_performance.png",
            },
        ],
    },
    {
        "section": "11.  Dashboard Screenshots ? Investor and Market Trends",
        "page_no": 13,
        "items": [
            {
                "heading": "Dashboard Page 3: Investor Analytics",
                "body": "The Investor Analytics page provides state-level transaction amount breakdowns, SIP / Lumpsum / Redemption mix, age group versus average SIP amount analysis, and monthly transaction volume trends. These views are designed to support investor segmentation and distribution planning.",
                "image": "page_3_investor_analytics.png",
            },
            {
                "heading": "Dashboard Page 4: SIP & Market Trends",
                "body": "The SIP & Market Trends page combines SIP inflow movement with NIFTY context, category inflow heatmap, and the leading FY25 categories by net inflow. It connects investor contribution behaviour with broader market conditions.",
                "image": "page_4_sip_market_trends.png",
            },
        ],
    },
]

BODY_STYLE = ParagraphStyle(
    "body",
    fontName="Helvetica",
    fontSize=8.2,
    leading=11.2,
    textColor=TEXT,
    alignment=TA_LEFT,
)
HEADING_STYLE = ParagraphStyle(
    "heading",
    fontName="Helvetica-Bold",
    fontSize=10.2,
    leading=12.2,
    textColor=BLUE,
)
TITLE_STYLE = ParagraphStyle(
    "title",
    fontName="Helvetica-Bold",
    fontSize=15,
    leading=18,
    textColor=colors.HexColor("#0b3778"),
)


def draw_header(pdf: canvas.Canvas, page_no: int) -> None:
    """Draw the report header/footer used by the enhanced report."""
    page_width, _ = A4
    pdf.setFont("Helvetica-Bold", 7.5)
    pdf.setFillColor(BLUE)
    pdf.drawString(0.72 * inch, 10.95 * inch, "BLUESTOCK")
    pdf.setFont("Helvetica", 7.5)
    pdf.setFillColor(MUTED)
    pdf.drawString(1.55 * inch, 10.95 * inch, "MUTUAL FUND ANALYTICS CAPSTONE")
    pdf.setFillColor(BLUE)
    pdf.drawRightString(page_width - 0.7 * inch, 10.95 * inch, "YASH VARDHAN BANSAL")
    pdf.setStrokeColor(BLUE)
    pdf.setLineWidth(0.7)
    pdf.line(0.72 * inch, 10.88 * inch, page_width - 0.7 * inch, 10.88 * inch)
    pdf.setFont("Helvetica", 7)
    pdf.setFillColor(MUTED)
    pdf.drawString(0.72 * inch, 0.42 * inch, "Bluestock Mutual Fund Analytics  |  Final Capstone Report")
    pdf.drawRightString(page_width - 0.7 * inch, 0.42 * inch, f"Page {page_no}")


def draw_paragraph(pdf: canvas.Canvas, text: str, style: ParagraphStyle, x: float, y: float, width: float) -> float:
    """Draw a paragraph and return the next y position."""
    paragraph = Paragraph(text, style)
    _, height = paragraph.wrap(width, 2 * inch)
    paragraph.drawOn(pdf, x, y - height)
    return y - height


def draw_image(pdf: canvas.Canvas, image_path: Path, x: float, y: float, max_width: float, max_height: float) -> float:
    """Draw an image preserving aspect ratio and return the next y position."""
    with Image.open(image_path) as image:
        image_width, image_height = image.size
    scale = min(max_width / image_width, max_height / image_height)
    draw_width = image_width * scale
    draw_height = image_height * scale
    pdf.drawImage(
        str(image_path),
        x + (max_width - draw_width) / 2,
        y - draw_height,
        width=draw_width,
        height=draw_height,
        preserveAspectRatio=True,
        mask="auto",
    )
    return y - draw_height


def draw_inline_page(pdf: canvas.Canvas, page_spec: dict) -> None:
    """Draw one dashboard-description page with inline screenshots."""
    page_width, _ = A4
    left = 0.72 * inch
    width = page_width - (1.42 * inch)
    draw_header(pdf, page_spec["page_no"])
    y = 10.55 * inch
    y = draw_paragraph(pdf, page_spec["section"], TITLE_STYLE, left, y, width) - 0.18 * inch
    pdf.setStrokeColor(BLUE)
    pdf.setLineWidth(1.0)
    pdf.line(left, y, page_width - 0.7 * inch, y)
    y -= 0.38 * inch

    image_height = 1.72 * inch
    for item_index, item in enumerate(page_spec["items"]):
        y = draw_paragraph(pdf, item["heading"], HEADING_STYLE, left, y, width)
        y -= 0.04 * inch
        y = draw_paragraph(pdf, item["body"], BODY_STYLE, left, y, width)
        y -= 0.12 * inch
        y = draw_image(pdf, SCREENSHOT_DIR / item["image"], left, y, width, image_height)
        y -= 0.28 * inch if item_index == 0 else 0
    pdf.showPage()


def build_inline_pages() -> Path:
    """Create a replacement PDF for the two dashboard screenshot pages."""
    missing = [
        item["image"]
        for page in DASHBOARD_INLINE_PAGES
        for item in page["items"]
        if not (SCREENSHOT_DIR / item["image"]).exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing dashboard screenshots: {', '.join(missing)}")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(INLINE_PATH), pagesize=A4)
    for page_spec in DASHBOARD_INLINE_PAGES:
        draw_inline_page(pdf, page_spec)
    pdf.save()
    return INLINE_PATH


def merge_report_with_dashboard_pages() -> Path:
    """Replace source pages 12 and 13 with inline dashboard screenshot pages."""
    if not SOURCE_REPORT.exists():
        raise FileNotFoundError(f"Source report not found: {SOURCE_REPORT}")

    inline_path = build_inline_pages()
    source_reader = PdfReader(str(SOURCE_REPORT))
    inline_reader = PdfReader(str(inline_path))
    writer = PdfWriter()
    start = REPLACE_PAGE_START - 1
    end = start + len(inline_reader.pages)

    for page in source_reader.pages[:start]:
        writer.add_page(page)
    for page in inline_reader.pages:
        writer.add_page(page)
    for page in source_reader.pages[end:]:
        writer.add_page(page)

    with OUTPUT_REPORT.open("wb") as output_file:
        writer.write(output_file)
    return OUTPUT_REPORT


def main() -> None:
    """CLI entrypoint for enhanced report generation."""
    output_path = merge_report_with_dashboard_pages()
    print(f"Enhanced report with inline dashboard screenshots generated: {output_path}")


if __name__ == "__main__":
    main()

