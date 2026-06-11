"""Streamlit dashboard for the Bluestock mutual fund analytics capstone."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EDA_CHART_DIR = PROJECT_ROOT / "reports" / "eda_charts"
PERFORMANCE_CHART_DIR = PROJECT_ROOT / "reports" / "performance_charts"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
APP_SCREENSHOT_DIR = REPORTS_DIR / "app_dashboard_screenshots"
DB_PATH = PROJECT_ROOT / "data" / "db" / "bluestock_mf.db"
APP_LOGO_PATH = PROJECT_ROOT / "dashboard" / "assets" / "bluestock_fintech_logo.svg"
PAGE_EXPORTS = {
    "Industry Overview": "page_1_industry_overview.png",
    "Fund Performance": "page_2_fund_performance.png",
    "Investor Analytics": "page_3_investor_analytics.png",
    "SIP & Market Trends": "page_4_sip_market_trends.png",
}
COLORS = ["#2563eb", "#0f766e", "#dc2626", "#7c3aed", "#f59e0b", "#0891b2", "#be123c", "#4b5563"]

st.set_page_config(page_title="Bluestock MF Analytics", page_icon="ðŸ“ˆ", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500&display=swap');

    /* â”€â”€ ROOT TOKENS â”€â”€ */
    :root {
        --bg-base:        #000000;
        --bg-surface:     #0a0a0a;
        --bg-card:        #0f0f0f;
        --bg-card-hover:  #141414;
        --border:         #1f1f1f;
        --border-accent:  #00e5ff22;
        --accent-cyan:    #00e5ff;
        --accent-blue:    #2979ff;
        --accent-teal:    #00bfa5;
        --accent-gold:    #ffd740;
        --text-primary:   #f0f0f0;
        --text-secondary: #888888;
        --text-muted:     #444444;
        --glow-cyan:      0 0 18px #00e5ff44;
        --glow-blue:      0 0 18px #2979ff44;
        --radius-sm:      6px;
        --radius-md:      10px;
        --radius-lg:      16px;
    }

    /* â”€â”€ GLOBAL RESET â”€â”€ */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background: var(--bg-base) !important;
        color: var(--text-primary) !important;
    }

    .stApp {
        background: var(--bg-base) !important;
    }

    /* Subtle grid texture on the background */
    .stApp::before {
        content: '';
        position: fixed;
        inset: 0;
        background-image:
            linear-gradient(rgba(0,229,255,0.018) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,229,255,0.018) 1px, transparent 1px);
        background-size: 48px 48px;
        pointer-events: none;
        z-index: 0;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-right: 3.25rem;
        padding-bottom: 2rem;
        position: relative;
        z-index: 1;
    }

    /* â”€â”€ SIDEBAR â”€â”€ */
    section[data-testid="stSidebar"] {
        background: #050505 !important;
        border-right: 1px solid var(--border) !important;
    }

    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="collapsedControl"] button {
        background: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #ffffff !important;
        border-radius: 8px !important;
        box-shadow: 0 0 10px rgba(255,255,255,0.22) !important;
    }

    [data-testid="stSidebarCollapseButton"] button {
        position: absolute !important;
        right: 10px !important;
        top: 8px !important;
        width: 30px !important;
        height: 30px !important;
        min-width: 30px !important;
        padding: 0 !important;
    }

    [data-testid="stSidebarCollapseButton"] svg,
    [data-testid="collapsedControl"] svg {
        fill: #000000 !important;
        color: #000000 !important;
    }

    section[data-testid="stSidebar"]::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent-cyan), var(--accent-blue), var(--accent-teal));
    }

    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] p {
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
    }

    section[data-testid="stSidebar"] .stRadio label {
        color: var(--text-secondary) !important;
        font-size: 0.85rem;
        letter-spacing: 0.02em;
        transition: color 0.2s;
    }

    section[data-testid="stSidebar"] .stRadio label:hover {
        color: var(--accent-cyan) !important;
    }

    /* Sidebar title */
    section[data-testid="stSidebar"] h1 {
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.12em !important;
        background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-transform: uppercase;
    }

    .sidebar-logo {
        margin: 0.2rem auto 0.8rem auto;
        width: 165px;
        max-width: 72%;
    }

    .sidebar-credit {
        width: 100%;
        max-width: 100%;
        box-sizing: border-box;
        color: #ffffff !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.68rem !important;
        letter-spacing: 0.01em;
        opacity: 0.88;
        background: rgba(5,5,5,0.92);
        padding: 10px 4px 4px 0;
        margin-top: 1rem;
        border-top: 1px solid #1f1f1f;
        white-space: normal;
        line-height: 1.35;
        overflow-wrap: anywhere;
    }

    /* â”€â”€ HEADINGS â”€â”€ */
    h1 {
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 2.1rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase;
        background: linear-gradient(135deg, #ffffff 0%, var(--accent-cyan) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0 !important;
    }

    h2 {
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 1.35rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.04em !important;
        color: var(--text-primary) !important;
        border-bottom: 1px solid var(--border);
        padding-bottom: 6px;
    }

    h3 {
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.03em !important;
        color: var(--accent-cyan) !important;
    }

    /* Caption / subtitle */
    .stCaption, [data-testid="stCaptionContainer"] p {
        color: var(--text-secondary) !important;
        font-size: 0.82rem !important;
        letter-spacing: 0.03em;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* â”€â”€ METRIC CARDS â”€â”€ */
    [data-testid="stMetric"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-top: 2px solid var(--accent-cyan) !important;
        border-radius: var(--radius-md) !important;
        padding: 16px 18px !important;
        transition: border-color 0.25s, box-shadow 0.25s;
        position: relative;
        overflow: hidden;
    }

    [data-testid="stMetric"]::after {
        content: '';
        position: absolute;
        inset: 0;
        background: radial-gradient(ellipse at top left, rgba(0,229,255,0.05) 0%, transparent 65%);
        pointer-events: none;
    }

    [data-testid="stMetric"]:hover {
        border-color: var(--accent-cyan) !important;
        box-shadow: var(--glow-cyan);
    }

    [data-testid="stMetricLabel"] p {
        color: var(--text-secondary) !important;
        font-size: 0.72rem !important;
        font-family: 'JetBrains Mono', monospace !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase;
    }

    div[data-testid="stMetricValue"] {
        color: var(--accent-cyan) !important;
        font-size: 1.5rem !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em;
    }

    div[data-testid="stMetricDelta"] {
        color: var(--accent-teal) !important;
        font-size: 0.8rem !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* â”€â”€ DATAFRAMES â”€â”€ */
    .stDataFrame, [data-testid="stDataFrameResizable"] {
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-md) !important;
        overflow: hidden;
        background: var(--bg-card) !important;
    }

    .stDataFrame thead tr th {
        background: #0a0a0a !important;
        color: var(--accent-cyan) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase;
        border-bottom: 1px solid var(--border) !important;
    }

    .stDataFrame tbody tr td {
        color: var(--text-primary) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.72rem !important;
        background: var(--bg-card) !important;
        border-bottom: 1px solid #111 !important;
    }

    [data-testid="stImage"] img {
        max-height: 230px;
        object-fit: contain;
    }

    [data-testid="stPyplot"] img,
    [data-testid="stPyplot"] canvas {
        max-height: 210px !important;
        object-fit: contain !important;
    }

    [data-testid="stDownloadButton"] button {
        width: auto !important;
        min-width: 92px !important;
        max-width: 130px !important;
        padding: 0.34rem 0.55rem !important;
        font-size: 0.72rem !important;
        white-space: nowrap !important;
    }

    [data-testid="stVerticalBlock"] {
        gap: 0.65rem !important;
    }

    [data-testid="stImage"] {
        margin-bottom: 0.15rem;
    }

    .stDataFrame tbody tr:hover td {
        background: var(--bg-card-hover) !important;
    }

    /* â”€â”€ SELECTBOX / MULTISELECT / SLIDER â”€â”€ */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-primary) !important;
        transition: border-color 0.2s;
    }

    .stSelectbox > div > div:focus-within,
    .stMultiSelect > div > div:focus-within {
        border-color: var(--accent-cyan) !important;
        box-shadow: var(--glow-cyan);
    }

    .stSelectbox label, .stMultiSelect label,
    .stSlider label {
        color: var(--text-secondary) !important;
        font-size: 0.75rem !important;
        font-family: 'JetBrains Mono', monospace !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase;
    }

    /* Slider track */
    .stSlider [data-baseweb="slider"] [data-testid="stSliderTrack"] {
        background: var(--border) !important;
    }

    .stSlider [data-baseweb="slider"] [data-testid="stSliderTrackFill"] {
        background: linear-gradient(90deg, var(--accent-cyan), var(--accent-blue)) !important;
    }

    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: var(--accent-cyan) !important;
        border: 2px solid #000 !important;
        box-shadow: var(--glow-cyan);
    }

    /* Multiselect tags */
    .stMultiSelect span[data-baseweb="tag"] {
        background: rgba(0,229,255,0.12) !important;
        border: 1px solid rgba(0,229,255,0.3) !important;
        color: var(--accent-cyan) !important;
        border-radius: 4px !important;
        font-size: 0.72rem !important;
    }

    /* Dropdown options */
    [data-baseweb="popover"] ul,
    [data-baseweb="menu"] ul {
        background: #0d0d0d !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
    }

    [data-baseweb="menu"] li {
        color: var(--text-primary) !important;
        font-size: 0.82rem !important;
    }

    [data-baseweb="menu"] li:hover {
        background: rgba(0,229,255,0.08) !important;
        color: var(--accent-cyan) !important;
    }

    /* â”€â”€ RADIO BUTTONS â”€â”€ */
    .stRadio [data-testid="stWidgetLabel"] p {
        color: var(--text-secondary) !important;
        font-size: 0.7rem !important;
        font-family: 'JetBrains Mono', monospace !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase;
    }

    /* â”€â”€ WARNINGS / ERRORS â”€â”€ */
    .stAlert {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-primary) !important;
    }

    .stAlert[data-baseweb="notification"] {
        border-left: 3px solid var(--accent-gold) !important;
    }

    /* â”€â”€ NOTE CALLOUT â”€â”€ */
    .note {
        border-left: 3px solid var(--accent-cyan);
        background: rgba(0,229,255,0.05);
        padding: 14px 18px;
        border-radius: var(--radius-sm);
        color: var(--text-secondary) !important;
        font-size: 0.82rem;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 1rem;
    }

    /* â”€â”€ PYPLOT CHART BACKGROUNDS â”€â”€ */
    .stPlotlyChart, [data-testid="stImage"] {
        border-radius: var(--radius-md);
        overflow: hidden;
        border: 1px solid var(--border);
    }

    /* â”€â”€ SCROLLBAR â”€â”€ */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: var(--bg-base); }
    ::-webkit-scrollbar-thumb { background: #1f1f1f; border-radius: 99px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--accent-cyan); }

    /* â”€â”€ HIDE SIDEBAR COLLAPSE BUTTON TEXT ONLY, keep icon white â”€â”€ */
    button[data-testid="collapsedControl"] span,
    [data-testid="stSidebarCollapsedControl"] span,
    header [data-testid="baseButton-headerNoPadding"] span,
    section[data-testid="stSidebarCollapsedControl"] span,
    [data-testid="stSidebarCollapseButton"] span {
        font-size: 0 !important;
        color: transparent !important;
        visibility: hidden !important;
        width: 0 !important;
        overflow: hidden !important;
    }
    /* Keep the sidebar toggle/reopen button visible as a white button */
    [data-testid="stSidebarCollapseButton"] button,
    button[data-testid="collapsedControl"],
    [data-testid="collapsedControl"] button,
    [data-testid="stSidebarCollapsedControl"] button,
    header [data-testid="baseButton-headerNoPadding"] {
        background: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #ffffff !important;
        border-radius: 8px !important;
        opacity: 1 !important;
        box-shadow: 0 0 10px rgba(255,255,255,0.22) !important;
        min-width: 30px !important;
        width: 30px !important;
        height: 30px !important;
        padding: 0 !important;
    }
    [data-testid="stSidebarCollapseButton"] svg,
    button[data-testid="collapsedControl"] svg,
    [data-testid="collapsedControl"] svg,
    [data-testid="stSidebarCollapsedControl"] svg,
    header [data-testid="baseButton-headerNoPadding"] svg {
        fill: #000000 !important;
        stroke: #000000 !important;
        color: #000000 !important;
    }
    /* Hide any stray material-icon TEXT (not svg) in the top header bar */
    header[data-testid="stHeader"] button span:not(:has(svg)) {
        font-size: 0 !important;
        color: transparent !important;
    }

    /* â”€â”€ TOP HEADER BAR â”€â”€ transparent bg, only icons white â”€â”€ */
    header[data-testid="stHeader"] {
        background: transparent !important;
        border-bottom: none !important;
    }

    /* SVG icons inside header buttons â€” white only */
    header[data-testid="stHeader"] button svg,
    header[data-testid="stHeader"] svg,
    [data-testid="stToolbarActions"] svg,
    [data-testid="stToolbarActionButton"] svg {
        fill: #ffffff !important;
        stroke: #ffffff !important;
        color: #ffffff !important;
        opacity: 1 !important;
    }

    /* Buttons themselves transparent, text white */
    header[data-testid="stHeader"] button,
    [data-testid="stToolbarActions"] button,
    [data-testid="stToolbarActionButton"] {
        color: #ffffff !important;
        background: transparent !important;
        border: none !important;
        opacity: 1 !important;
    }

    /* â”€â”€ TABS (if any) â”€â”€ */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid var(--border) !important;
        gap: 0;
    }

    .stTabs [data-baseweb="tab"] {
        color: var(--text-secondary) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase;
        border-radius: 0 !important;
        border-bottom: 2px solid transparent !important;
        padding: 8px 18px !important;
        background: transparent !important;
    }

    .stTabs [aria-selected="true"] {
        color: var(--accent-cyan) !important;
        border-bottom: 2px solid var(--accent-cyan) !important;
        background: transparent !important;
    }

    /* â”€â”€ COLUMN BORDERS (subtle dividers) â”€â”€ */
    [data-testid="column"] {
        padding: 0 8px;
    }

    /* â”€â”€ TOOLTIPS â”€â”€ */
    [data-baseweb="tooltip"] {
        background: #111 !important;
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.75rem !important;
    }

    @media (max-width: 900px) {
        .block-container {
            padding-left: 0.85rem !important;
            padding-right: 3.4rem !important;
        }
        h1 {
            font-size: 1.65rem !important;
        }
        [data-testid="stDownloadButton"] button {
            min-width: 104px !important;
            max-width: 118px !important;
            font-size: 0.68rem !important;
        }
        [data-testid="stToolbarActions"],
        [data-testid="stToolbarActionButton"] {
            margin-right: 0.2rem !important;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)
sns.set_theme(style="dark")
plt.rcParams.update({
    "figure.facecolor":  "#000000",
    "axes.facecolor":    "#0a0a0a",
    "axes.edgecolor":    "#1f1f1f",
    "axes.labelcolor":   "#888888",
    "xtick.color":       "#555555",
    "ytick.color":       "#555555",
    "text.color":        "#f0f0f0",
    "grid.color":        "#141414",
    "grid.linewidth":    0.6,
    "legend.facecolor":  "#0a0a0a",
    "legend.edgecolor":  "#1f1f1f",
    "legend.labelcolor": "#aaaaaa",
    "font.family":       "monospace",
})


@st.cache_data(show_spinner=False)
def read_csv(name: str, dates: list[str] | None = None) -> pd.DataFrame:
    path = PROCESSED_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=dates or [])


@st.cache_data(show_spinner=False)
def load_data() -> dict[str, pd.DataFrame]:
    """Load only lightweight data needed for sidebar filters and app boot."""
    return {
        "funds": read_csv("dim_fund.csv", ["launch_date"]),
        "scorecard": read_csv("fund_scorecard.csv"),
    }


PAGE_DATASETS = {
    "Industry Overview": {
        "aum": ("fact_aum.csv", ["aum_date"]),
        "sip": ("fact_sip.csv", ["month"]),
        "folios": ("fact_folios.csv", ["month"]),
    },
    "Fund Performance": {
        "nav": ("fact_nav.csv", ["nav_date"]),
        "benchmark": ("fact_benchmark.csv", ["bench_date"]),
    },
    "Investor Analytics": {
        "transactions": ("fact_transactions.csv", ["transaction_date"]),
        "cohort": ("cohort_analysis.csv", []),
        "continuity": ("sip_continuity.csv", ["last_sip_date"]),
    },
    "SIP & Market Trends": {
        "sip": ("fact_sip.csv", ["month"]),
        "inflows": ("fact_category_inflows.csv", ["month"]),
        "benchmark": ("fact_benchmark.csv", ["bench_date"]),
    },
    "Prediction & Portfolio": {
        "monte_carlo": ("monte_carlo_nav_projection.csv", []),
        "frontier": ("efficient_frontier.csv", []),
        "hhi": ("sector_hhi.csv", []),
    },
}


@st.cache_data(show_spinner=False)
def load_page_data(page: str) -> dict[str, pd.DataFrame]:
    """Load heavier page-specific datasets only when that page is opened."""
    datasets = {}
    for key, (file_name, date_cols) in PAGE_DATASETS.get(page, {}).items():
        datasets[key] = read_csv(file_name, date_cols)
    return datasets


@st.cache_data(show_spinner=False)
def compare_ml_models() -> tuple[pd.DataFrame, pd.DataFrame]:
    score_path = PROCESSED_DIR / "fund_scorecard.csv"
    if not score_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    df = pd.read_csv(score_path)
    features = ["expense_ratio_pct", "aum_crore", "sharpe_ratio", "sortino_ratio", "alpha_pct", "beta", "max_drawdown_pct"]
    target = "cagr_3yr_pct"
    needed = features + [target, "scheme_name", "fund_house"]
    model_df = df[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(model_df) < 12:
        return pd.DataFrame(), pd.DataFrame()
    x_train, x_test, y_train, y_test = train_test_split(
        model_df[features],
        model_df[target],
        test_size=0.25,
        random_state=42,
    )
    models = {
        "RandomForestRegressor": RandomForestRegressor(n_estimators=250, random_state=42, min_samples_leaf=2),
        "LinearRegression": LinearRegression(),
    }
    metrics = []
    prediction_frames = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        metrics.append(
            {
                "model": name,
                "mae_points": mean_absolute_error(y_test, preds),
                "r2_score": r2_score(y_test, preds),
            }
        )
        frame = model_df.loc[x_test.index, ["scheme_name", "fund_house", target]].copy()
        frame["model"] = name
        frame["predicted_cagr_3yr_pct"] = preds
        prediction_frames.append(frame)
    return pd.DataFrame(metrics).sort_values(["r2_score", "mae_points"], ascending=[False, True]), pd.concat(prediction_frames, ignore_index=True)


def style_chart_axes(ax, x_is_date: bool = False) -> None:
    ax.tick_params(axis="both", labelsize=7, colors="#9ca3af", pad=2)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    ax.title.set_size(10)
    ax.grid(True, color="#262626", linewidth=0.45, alpha=0.7)
    if x_is_date:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.tick_params(axis="x", rotation=0)
    else:
        labels = ax.get_xticklabels()
        if len(labels) > 4:
            ax.tick_params(axis="x", rotation=25)
            for label in labels:
                label.set_horizontalalignment("right")


def plot_fig(fig) -> None:
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def line_plot(df: pd.DataFrame, x: str, y: str, title: str, hue: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 2.25))
    if hue:
        sns.lineplot(data=df, x=x, y=y, hue=hue, ax=ax, palette=COLORS, linewidth=2)
        ax.legend(loc="best", fontsize=8)
    else:
        sns.lineplot(data=df, x=x, y=y, ax=ax, color=COLORS[0], linewidth=2)
    ax.set_title(title, loc="left", fontsize=10, weight="bold", color="#f0f0f0")
    ax.set_xlabel("")
    style_chart_axes(ax, pd.api.types.is_datetime64_any_dtype(df[x]))
    plot_fig(fig)


def bar_plot(df: pd.DataFrame, x: str, y: str, title: str, horizontal: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 2.25))
    if horizontal:
        sns.barplot(data=df, x=x, y=y, ax=ax, palette=COLORS, hue=y, legend=False)
    else:
        sns.barplot(data=df, x=x, y=y, ax=ax, palette=COLORS, hue=x, legend=False)
    ax.set_title(title, loc="left", fontsize=10, weight="bold", color="#f0f0f0")
    style_chart_axes(ax)
    plot_fig(fig)


def aum_trend_plot(trend: pd.DataFrame) -> None:
    """Plot AUM trend gracefully for quarterly data, including single-point ranges."""
    fig, ax = plt.subplots(figsize=(5.8, 2.25))
    ax.set_title("Industry AUM Trend (Rs crore)", loc="left", fontsize=10, weight="bold", color="#f0f0f0")
    ax.set_xlabel("")
    ax.set_ylabel("aum_crore")
    style_chart_axes(ax, True)
    if trend.empty:
        ax.text(
            0.5,
            0.5,
            "No quarterly AUM observation is available in this selected period.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#888888",
            fontsize=10,
        )
    elif len(trend) == 1:
        ax.scatter(trend["aum_date"], trend["aum_crore"], color=COLORS[0], s=90, zorder=3)
        ax.annotate(
            f"{money_cr(float(trend['aum_crore'].iloc[0]))}",
            xy=(trend["aum_date"].iloc[0], trend["aum_crore"].iloc[0]),
            xytext=(10, 12),
            textcoords="offset points",
            color="#00e5ff",
            fontsize=9,
        )
    else:
        sns.lineplot(data=trend, x="aum_date", y="aum_crore", ax=ax, color=COLORS[0], linewidth=2.2, marker="o", markersize=6)
    plot_fig(fig)


def money_cr(value: float) -> str:
    if pd.isna(value):
        return "NA"
    if abs(value) >= 100000:
        return f"Rs {value / 100000:.2f}L Cr"
    return f"Rs {value:,.0f} Cr"


def header(title: str) -> None:
    """Render the page title and compact export button without helper text."""
    st.title(title)
    page_export = PAGE_EXPORTS.get(title)
    if page_export:
        page_path = APP_SCREENSHOT_DIR / page_export
        if not page_path.exists():
            page_path = REPORTS_DIR / "day5_dashboard_pages" / page_export
        if page_path.exists():
            button_col, _ = st.columns([0.12, 0.88])
            with button_col:
                st.download_button(
                    "Download Page",
                    data=page_path.read_bytes(),
                    file_name=page_export,
                    mime="image/png",
                    use_container_width=False,
                    key=f"download_page_{title}",
                )


def require_outputs(data: dict[str, pd.DataFrame]) -> None:
    if data["funds"].empty or data["scorecard"].empty:
        st.error("Run `python scripts/etl_pipeline.py` and `python scripts/compute_metrics.py` before opening the dashboard.")
        st.stop()


def sidebar(data: dict[str, pd.DataFrame]) -> dict[str, object]:
    funds = data["funds"]
    st.sidebar.markdown('<div class="sidebar-logo">', unsafe_allow_html=True)
    st.sidebar.image(str(APP_LOGO_PATH), use_container_width=True)
    st.sidebar.markdown("</div>", unsafe_allow_html=True)
    page = st.sidebar.radio(
        "Dashboard Page",
        ["Industry Overview", "Fund Performance", "Investor Analytics", "SIP & Market Trends", "Data Quality", "Performance Analytics", "EDA Analysis", "Prediction & Portfolio", "Model & Report", "Power BI Export", "Fund Recommender"],
    )
    houses_all = sorted(funds["fund_house"].dropna().unique())
    categories_all = sorted(funds["category"].dropna().unique())
    plans_all = sorted(funds["plan"].dropna().unique())
    risks_all = sorted(funds["risk_category"].dropna().unique())
    filters = {
        "page": page,
        "houses": st.sidebar.multiselect("Fund House", houses_all, default=houses_all[:6]),
        "categories": st.sidebar.multiselect("Category", categories_all, default=categories_all),
        "plans": st.sidebar.multiselect("Plan", plans_all, default=plans_all),
        "risks": st.sidebar.multiselect("Risk", risks_all, default=risks_all),
    }
    st.sidebar.markdown('<div class="sidebar-credit">By - Yash vardhan bansal</div>', unsafe_allow_html=True)
    return filters


def score_filtered(data: dict[str, pd.DataFrame], filters: dict[str, object]) -> pd.DataFrame:
    score = data["scorecard"]
    return score[
        score["fund_house"].isin(filters["houses"])
        & score["category"].isin(filters["categories"])
        & score["plan"].isin(filters["plans"])
        & score["risk_category"].isin(filters["risks"])
    ].copy()


def industry_page(data: dict[str, pd.DataFrame], filters: dict[str, object]) -> None:
    header("Industry Overview")
    aum, sip, folios = data["aum"], data["sip"], data["folios"]
    amc_options = sorted(aum["fund_house"].unique())
    houses = st.multiselect("AMC comparison", amc_options, default=amc_options)
    if not houses:
        st.warning("Select at least one AMC to view AUM trends.")
        return
    _aum_years = sorted(aum["aum_date"].dt.year.unique())
    _aum_months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    _dcol1, _dcol2, _dcol3, _dcol4 = st.columns(4)
    _from_year  = _dcol1.selectbox("From Year",  _aum_years, index=0, key="aum_from_yr")
    _from_month = _dcol2.selectbox("From Month", _aum_months, index=0, key="aum_from_mo")
    _to_year    = _dcol3.selectbox("To Year",    _aum_years, index=len(_aum_years)-1, key="aum_to_yr")
    _to_month   = _dcol4.selectbox("To Month",   _aum_months, index=11, key="aum_to_mo")
    import datetime as _dt
    import calendar as _calendar
    _from_date = _dt.date(_from_year, _aum_months.index(_from_month)+1, 1)
    _to_month_idx = _aum_months.index(_to_month)+1
    _to_date = _dt.date(_to_year, _to_month_idx, _calendar.monthrange(_to_year, _to_month_idx)[1])
    date_range = (_from_date, _to_date)
    aum_view = aum[(aum["fund_house"].isin(houses)) & (aum["aum_date"].dt.date.between(*date_range))]
    latest_sip = sip.sort_values("month").iloc[-1]
    latest_folio = folios.sort_values("month").iloc[-1]
    latest_aum = aum.sort_values("aum_date").groupby("fund_house").tail(1)["aum_crore"].sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Industry AUM", money_cr(latest_aum))
    c2.metric("Latest SIP Inflow", money_cr(latest_sip["sip_inflow_crore"]))
    c3.metric("Folios", f"{latest_folio['total_folios_crore']:.2f} Cr")
    c4.metric("Tracked Schemes", f"{data['funds']['amfi_code'].nunique():,}")

    left, right = st.columns(2)
    with left:
        trend = aum_view.groupby("aum_date", as_index=False)["aum_crore"].sum()
        aum_trend_plot(trend)
    with right:
        if not aum_view.empty:
            latest = aum_view.sort_values("aum_date").groupby("fund_house").tail(1).sort_values("aum_crore")
            bar_plot(latest, "aum_crore", "fund_house", "AMC AUM Ranking", horizontal=True)
    folio_long = folios.melt("month", value_vars=["equity_folios_crore", "debt_folios_crore", "hybrid_folios_crore", "others_folios_crore"], var_name="segment", value_name="folios")
    folio_col, sip_col = st.columns(2)
    with folio_col:
        line_plot(folio_long, "month", "folios", "Folio Growth by Segment", "segment")
    with sip_col:
        line_plot(sip, "month", "sip_inflow_crore", "Monthly SIP Inflow")


def fund_page(data: dict[str, pd.DataFrame], filters: dict[str, object]) -> None:
    header("Fund Performance")
    score = score_filtered(data, filters)
    if score.empty:
        st.warning("No funds match the selected slicers.")
        return
    subcats = st.multiselect("Sub-category", sorted(score["sub_category"].unique()), default=sorted(score["sub_category"].unique()))
    min_score = st.slider("Minimum composite score", 0, 100, 0)
    score = score[(score["sub_category"].isin(subcats)) & (score["composite_score"] >= min_score)]
    if score.empty:
        st.warning("No funds match the selected sub-category and score filters.")
        return
    top = score.sort_values("composite_score", ascending=False).iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Fund", top["scheme_name"][:28])
    c2.metric("Score", f"{top['composite_score']:.1f}")
    c3.metric("Sharpe", f"{top['sharpe_ratio']:.2f}")
    c4.metric("Alpha", f"{top['alpha_pct']:.2f}%")

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(5.8, 2.25))
        size = np.clip(score["aum_crore"].fillna(score["aum_crore"].median()) / 150, 30, 450)
        sns.scatterplot(data=score, x="annualised_return_pct", y="annualised_volatility_pct", hue="risk_category", size=size, sizes=(40, 420), palette=COLORS, ax=ax, legend=False)
        ax.set_title("Return vs Risk (bubble = AUM)", loc="left", fontsize=10, weight="bold", color="#f0f0f0")
        ax.set_xlabel("Annualised return %")
        ax.set_ylabel("Volatility %")
        style_chart_axes(ax)
        plot_fig(fig)
    with right:
        cols = ["scheme_name", "fund_house", "plan", "risk_category", "composite_score", "sharpe_ratio", "alpha_pct", "max_drawdown_pct"]
        st.dataframe(score.sort_values("composite_score", ascending=False)[cols], use_container_width=True, hide_index=True, height=255)

    selected = st.selectbox("NAV drill-through fund", score.sort_values("composite_score", ascending=False)["scheme_name"])
    code = int(score.loc[score["scheme_name"] == selected, "amfi_code"].iloc[0])
    nav = data["nav"][(data["nav"]["amfi_code"] == code) & (data["nav"]["is_observed_nav"] == 1)]
    line_plot(nav, "nav_date", "nav", f"NAV Trend: {selected}")


def investor_page(data: dict[str, pd.DataFrame]) -> None:
    header("Investor Analytics")
    tx = data["transactions"]
    states = st.multiselect("State", sorted(tx["state"].unique()), default=sorted(tx["state"].unique())[:10])
    ages = st.multiselect("Age Group", sorted(tx["age_group"].unique()), default=sorted(tx["age_group"].unique()))
    tiers = st.multiselect("City Tier", sorted(tx["city_tier"].unique()), default=sorted(tx["city_tier"].unique()))
    tx = tx[tx["state"].isin(states) & tx["age_group"].isin(ages) & tx["city_tier"].isin(tiers)]
    if tx.empty:
        st.warning("No investor transactions match the selected filters. Try adding more states, age groups, or city tiers.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{len(tx):,}")
    c2.metric("Investors", f"{tx['investor_id'].nunique():,}")
    c3.metric("Total Amount", f"Rs {tx['amount_inr'].sum() / 1e7:,.2f} Cr")
    c4.metric("Avg Ticket", f"Rs {tx['amount_inr'].mean():,.0f}")

    left, right = st.columns(2)
    with left:
        state = tx.groupby("state", as_index=False)["amount_inr"].sum().sort_values("amount_inr").tail(10)
        bar_plot(state, "amount_inr", "state", "Transaction Amount by State", horizontal=True)
    with right:
        split = tx.groupby("transaction_type")["amount_inr"].sum()
        fig, ax = plt.subplots(figsize=(5.8, 2.25))
        ax.pie(
            split,
            labels=None,
            autopct="%1.0f%%",
            pctdistance=0.76,
            colors=COLORS[: len(split)],
            textprops={"fontsize": 7, "color": "#f0f0f0"},
            wedgeprops={"width": 0.45},
        )
        ax.legend(split.index, loc="center left", bbox_to_anchor=(0.92, 0.5), fontsize=7, frameon=False)
        ax.set_title("Transaction Type Split", loc="left", fontsize=10, weight="bold", color="#f0f0f0")
        plot_fig(fig)

    left, right = st.columns(2)
    with left:
        sip = tx[tx["transaction_type"].str.upper() == "SIP"]
        age = sip.groupby("age_group", as_index=False)["amount_inr"].mean()
        bar_plot(age, "age_group", "amount_inr", "Average SIP by Age Group")
    with right:
        monthly = tx.groupby(tx["transaction_date"].dt.to_period("M").dt.to_timestamp()).size().reset_index(name="txn_count")
        line_plot(monthly, "transaction_date", "txn_count", "Monthly Transaction Volume")

    c1, c2 = st.columns(2)
    c1.subheader("Cohort Analysis")
    c1.dataframe(data["cohort"], use_container_width=True, hide_index=True, height=230)
    c2.subheader("At-Risk SIP Investors")
    c2.dataframe(data["continuity"].sort_values("average_gap_days", ascending=False).head(20), use_container_width=True, hide_index=True, height=230)


def sip_page(data: dict[str, pd.DataFrame]) -> None:
    header("SIP & Market Trends")
    sip, inflows, bench = data["sip"], data["inflows"], data["benchmark"]
    cats = st.multiselect("Inflow category", sorted(inflows["category"].unique()), default=sorted(inflows["category"].unique())[:6])
    idx = st.multiselect("Benchmark index", sorted(bench["index_name"].unique()), default=["NIFTY50"])
    _sip_years  = sorted(sip["month"].dt.year.unique())
    _sip_months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    _sc1, _sc2, _sc3, _sc4 = st.columns(4)
    _s_from_yr  = _sc1.selectbox("From Year",  _sip_years, index=0, key="sip_from_yr")
    _s_from_mo  = _sc2.selectbox("From Month", _sip_months, index=0, key="sip_from_mo")
    _s_to_yr    = _sc3.selectbox("To Year",    _sip_years, index=len(_sip_years)-1, key="sip_to_yr")
    _s_to_mo    = _sc4.selectbox("To Month",   _sip_months, index=11, key="sip_to_mo")
    import datetime as _dt2
    _s_from_date = _dt2.date(_s_from_yr, _sip_months.index(_s_from_mo)+1, 1)
    _s_to_date   = _dt2.date(_s_to_yr,   _sip_months.index(_s_to_mo)+1,   28)
    date_range = (_s_from_date, _s_to_date)
    sip_view = sip[sip["month"].dt.date.between(*date_range)]
    if sip_view.empty:
        st.warning("No SIP records are available for the selected period. Try widening the date range.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Peak SIP", money_cr(sip_view["sip_inflow_crore"].max()))
    c2.metric("Latest SIP Accounts", f"{sip_view.sort_values('month').iloc[-1]['active_sip_accounts_crore']:.2f} Cr")
    c3.metric("Latest YoY Growth", f"{sip_view.sort_values('month').iloc[-1]['yoy_growth_pct']:.1f}%")

    fig, ax1 = plt.subplots(figsize=(5.8, 2.25))
    ax1.bar(sip_view["month"], sip_view["sip_inflow_crore"], color=COLORS[1], width=20, label="SIP Inflow")
    ax1.set_ylabel("SIP Inflow (Rs crore)")
    ax2 = ax1.twinx()
    for index in idx:
        b = bench[bench["index_name"] == index].copy()
        monthly = b.groupby(b["bench_date"].dt.to_period("M").dt.to_timestamp())["close_value"].last().reset_index(name="close_value")
        ax2.plot(monthly["bench_date"], monthly["close_value"], label=index, linewidth=2)
    ax2.set_ylabel("Index close")
    ax1.set_title("SIP Inflow vs Market Index", loc="left", fontsize=10, weight="bold", color="#f0f0f0")
    ax2.legend(loc="upper left")
    style_chart_axes(ax1, True)
    style_chart_axes(ax2, True)
    plot_fig(fig)

    pivot = inflows[inflows["category"].isin(cats)].pivot_table(index="category", columns=inflows["month"].dt.strftime("%Y-%m"), values="net_inflow_crore", aggfunc="sum")
    fig, ax = plt.subplots(figsize=(5.8, 2.25))
    sns.heatmap(pivot, cmap="RdYlGn", ax=ax, linewidths=.4)
    ax.set_title("Category Net Inflow Heatmap", loc="left", fontsize=10, weight="bold", color="#f0f0f0")
    ax.tick_params(axis="x", labelsize=6, rotation=35)
    ax.tick_params(axis="y", labelsize=7)
    plot_fig(fig)
    top = inflows[inflows["category"].isin(cats)].groupby("category", as_index=False)["net_inflow_crore"].sum().sort_values("net_inflow_crore", ascending=False).head(5)
    bar_plot(top, "category", "net_inflow_crore", "Top Categories by Net Inflow")


def prediction_page(data: dict[str, pd.DataFrame], filters: dict[str, object]) -> None:
    header("Prediction & Portfolio")
    mc, frontier = data["monte_carlo"], data["frontier"]
    fund = st.selectbox("Projection fund", sorted(mc["scheme_name"].unique()))
    min_sharpe = st.slider("Minimum portfolio Sharpe", 0.0, float(max(0.1, frontier["portfolio_sharpe"].max())), 0.0, 0.05)
    band = mc[mc["scheme_name"] == fund]
    fview = frontier[frontier["portfolio_sharpe"] >= min_sharpe]
    chart_left, chart_right = st.columns(2)
    with chart_left:
        fig, ax = plt.subplots(figsize=(5.0, 1.95))
        ax.fill_between(band["projection_day"], band["p05_nav"], band["p95_nav"], color=COLORS[0], alpha=.15, label="5-95% band")
        ax.plot(band["projection_day"], band["p50_nav"], color=COLORS[0], linewidth=1.8, label="Median NAV")
        ax.set_title(f"Monte Carlo NAV Projection", loc="left", fontsize=10, weight="bold", color="#f0f0f0")
        ax.set_xlabel("Trading day")
        ax.set_ylabel("Projected NAV")
        ax.legend(fontsize=7)
        plot_fig(fig)
    with chart_right:
        fig, ax = plt.subplots(figsize=(5.0, 1.95))
        scatter = ax.scatter(fview["portfolio_volatility_pct"], fview["portfolio_return_pct"], c=fview["portfolio_sharpe"], cmap="viridis", s=16)
        fig.colorbar(scatter, ax=ax, label="Sharpe")
        ax.set_title("Markowitz Efficient Frontier", loc="left", fontsize=10, weight="bold", color="#f0f0f0")
        ax.set_xlabel("Volatility %")
        ax.set_ylabel("Return %")
        plot_fig(fig)

    left, right = st.columns(2)
    best = frontier.sort_values("portfolio_sharpe", ascending=False).head(1).T.reset_index()
    best.columns = ["Metric", "Value"]
    left.subheader("Best Simulated Portfolio")
    left.dataframe(best, use_container_width=True, hide_index=True, height=220)
    right.subheader("Sector Concentration HHI")
    right.dataframe(data["hhi"].sort_values("sector_hhi", ascending=False).head(15), use_container_width=True, hide_index=True, height=220)
    st.subheader("Filtered Fund Risk Table")
    risk = score_filtered(data, filters)[["scheme_name", "risk_category", "var_95_pct", "cvar_95_pct", "max_drawdown_pct", "composite_score"]]
    st.dataframe(risk, use_container_width=True, hide_index=True, height=260)


def recommender_page(data: dict[str, pd.DataFrame], filters: dict[str, object]) -> None:
    header("Fund Recommender")
    score = score_filtered(data, filters)
    risk = st.selectbox("Investor risk appetite", ["Low", "Moderate", "High"])
    mode = st.selectbox("Investment mode", ["SIP", "Lumpsum", "Either"])
    max_expense = st.slider("Maximum expense ratio %", 0.0, 2.5, 1.5, 0.05)
    risk_map = {"Low": ["Low"], "Moderate": ["Moderate", "Moderately High"], "High": ["High", "Very High"]}
    picks = score[score["risk_category"].isin(risk_map[risk]) & (score["expense_ratio_pct"] <= max_expense)].copy()
    if mode == "SIP":
        picks = picks.sort_values(["sharpe_ratio", "max_drawdown_pct", "composite_score"], ascending=[False, False, False])
    elif mode == "Lumpsum":
        picks = picks.sort_values(["alpha_pct", "annualised_return_pct", "composite_score"], ascending=False)
    else:
        picks = picks.sort_values("composite_score", ascending=False)
    st.subheader("Top 3 Recommendations")
    cols = st.columns(3)
    for idx, (_, row) in enumerate(picks.head(3).iterrows()):
        with cols[idx]:
            st.metric(row["scheme_name"][:30], f"{row['composite_score']:.1f}", f"Sharpe {row['sharpe_ratio']:.2f}")
    st.dataframe(picks[["scheme_name", "fund_house", "sub_category", "plan", "risk_category", "expense_ratio_pct", "sharpe_ratio", "alpha_pct", "composite_score"]].head(15), use_container_width=True, hide_index=True)
    st.markdown("<div class='note'>Logic: risk match first, then quality ranking. SIP prioritises Sharpe and drawdown resilience; lumpsum prioritises alpha and annualised return.</div>", unsafe_allow_html=True)


def data_quality_page() -> None:
    """Show only the user-facing Day 2 outputs that matter in the app."""
    header("Data Quality")

    clean_files = sorted(PROCESSED_DIR.glob("*_clean.csv"))
    row_counts_path = PROCESSED_DIR / "sqlite_row_count_verification.csv"

    row_counts = pd.read_csv(row_counts_path) if row_counts_path.exists() else pd.DataFrame()
    nav_row = row_counts[row_counts["table"].eq("fact_nav")] if not row_counts.empty else pd.DataFrame()
    txn_row = row_counts[row_counts["table"].eq("fact_transactions")] if not row_counts.empty else pd.DataFrame()

    nav_rows = int(nav_row["db_rows"].iloc[0]) if not nav_row.empty else 0
    txn_rows = int(txn_row["db_rows"].iloc[0]) if not txn_row.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cleaned CSVs", len(clean_files))
    c2.metric("Database", "Ready" if DB_PATH.exists() else "Missing")
    c3.metric("NAV Rows", f"{nav_rows:,}")
    c4.metric("Transactions", f"{txn_rows:,}")

    if not row_counts.empty:
        st.subheader("Loaded Table Row Counts")
        visible_cols = ["table", "source_rows", "cleaned_rows", "db_rows", "row_count_note"]
        st.dataframe(row_counts[visible_cols], use_container_width=True, hide_index=True, height=220)

    st.subheader("Useful SQL Insights")
    if DB_PATH.exists():
        import sqlite3

        queries = {
            "Top 5 Funds by AUM": """
                SELECT scheme_name, fund_house, aum_crore
                FROM fact_performance
                ORDER BY aum_crore DESC
                LIMIT 5
            """,
            "Transactions by State": """
                SELECT state, COUNT(*) AS txn_count, SUM(amount_inr) AS total_amount_inr
                FROM fact_transactions
                GROUP BY state
                ORDER BY total_amount_inr DESC
                LIMIT 10
            """,
            "Low Expense Ratio Funds": """
                SELECT scheme_name, fund_house, plan, expense_ratio_pct
                FROM dim_fund
                WHERE expense_ratio_pct < 1
                ORDER BY expense_ratio_pct, scheme_name
                LIMIT 10
            """,
        }
        with sqlite3.connect(DB_PATH) as conn:
            for title, sql in queries.items():
                st.markdown(f"**{title}**")
                st.dataframe(pd.read_sql_query(sql, conn), use_container_width=True, hide_index=True, height=190)
    else:
        st.warning("SQLite database is missing. Run `python scripts\\etl_pipeline.py` first.")


def performance_analytics_page() -> None:
    """Render Day 4 performance analytics in a compact dashboard page."""
    header("Performance Analytics")

    required = {
        "fund_scorecard": PROCESSED_DIR / "fund_scorecard.csv",
        "alpha_beta": PROCESSED_DIR / "alpha_beta.csv",
        "cagr": PROCESSED_DIR / "cagr_comparison.csv",
        "drawdown": PROCESSED_DIR / "max_drawdown.csv",
        "tracking": PROCESSED_DIR / "tracking_error.csv",
        "distribution": PROCESSED_DIR / "return_distribution_summary.csv",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        st.warning("Day 4 outputs are missing. Run `python scripts\\generate_performance_analytics.py` first.")
        return

    score = pd.read_csv(required["fund_scorecard"], parse_dates=["drawdown_start_date", "drawdown_end_date"])
    alpha_beta = pd.read_csv(required["alpha_beta"])
    cagr = pd.read_csv(required["cagr"])
    drawdown = pd.read_csv(required["drawdown"], parse_dates=["drawdown_start_date", "drawdown_end_date"])
    tracking = pd.read_csv(required["tracking"])
    distribution = pd.read_csv(required["distribution"])

    top = score.sort_values("composite_score", ascending=False).iloc[0]
    best_sharpe = score.sort_values("sharpe_ratio", ascending=False).iloc[0]
    best_alpha = score.sort_values("alpha_pct", ascending=False).iloc[0]
    worst_dd = drawdown.sort_values("max_drawdown_pct").iloc[0]
    reasonable = int(distribution["reasonable_distribution_flag"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Top Score", f"{top['composite_score']:.1f}", top["scheme_name"][:24])
    c2.metric("Best Sharpe", f"{best_sharpe['sharpe_ratio']:.2f}", best_sharpe["scheme_name"][:24])
    c3.metric("Best Alpha", f"{best_alpha['alpha_pct']:.2f}%", best_alpha["scheme_name"][:24])
    c4.metric("Worst Drawdown", f"{worst_dd['max_drawdown_pct']:.2f}%", worst_dd["scheme_name"][:24])

    st.markdown(
        f"<div class='note'>Daily return distribution check: {reasonable} of {len(distribution)} funds passed the +/-20% daily-return reasonableness flag. Risk-free rate used for Sharpe/Sortino is 6.5% annually.</div>",
        unsafe_allow_html=True,
    )

    chart_col, table_col = st.columns(2)
    with chart_col:
        st.subheader("Top 5 Funds vs NIFTY50 & NIFTY100")
        nav = pd.read_csv(PROCESSED_DIR / "fact_nav.csv", parse_dates=["nav_date"])
        funds = pd.read_csv(PROCESSED_DIR / "dim_fund.csv")
        benchmark = pd.read_csv(PROCESSED_DIR / "fact_benchmark.csv", parse_dates=["bench_date"])
        top5_codes = score.sort_values("composite_score", ascending=False)["amfi_code"].head(5).tolist()
        end_date = nav["nav_date"].max()
        start_date = end_date - pd.DateOffset(years=3)

        benchmark_close = benchmark[
            benchmark["index_name"].isin(["NIFTY50", "NIFTY100"])
            & benchmark["bench_date"].between(start_date, end_date)
        ].copy()
        benchmark_close = benchmark_close.sort_values(["index_name", "bench_date"])
        benchmark_close["normalized_value"] = benchmark_close.groupby("index_name")["close_value"].transform(lambda s: s / s.iloc[0] * 100)
        comparison = benchmark_close.rename(columns={"bench_date": "date", "index_name": "series"})[["date", "series", "normalized_value"]]

        selected_nav = nav[
            nav["amfi_code"].isin(top5_codes)
            & nav["is_observed_nav"].eq(1)
            & nav["nav_date"].between(start_date, end_date)
        ].merge(funds[["amfi_code", "scheme_name"]], on="amfi_code", how="left")
        selected_nav = selected_nav.sort_values(["amfi_code", "nav_date"])
        selected_nav["normalized_value"] = selected_nav.groupby("amfi_code")["nav"].transform(lambda s: s / s.iloc[0] * 100)
        fund_comparison = selected_nav.rename(columns={"nav_date": "date", "scheme_name": "series"})[["date", "series", "normalized_value"]]
        comparison = pd.concat([comparison, fund_comparison], ignore_index=True)

        fig, ax = plt.subplots(figsize=(5.8, 2.25))
        sns.lineplot(data=comparison, x="date", y="normalized_value", hue="series", ax=ax, linewidth=2)
        ax.set_title("Top 5 Funds vs NIFTY50 and NIFTY100", loc="left", fontsize=10, weight="bold", color="#f0f0f0")
        ax.set_xlabel("")
        ax.set_ylabel("Normalized value (start = 100)")
        ax.legend(fontsize=7, ncol=2)
        style_chart_axes(ax, True)
        plot_fig(fig)
    with table_col:
        st.subheader("Top 10 Fund Scorecard")
        score_cols = ["scheme_name", "fund_house", "cagr_3yr_pct", "sharpe_ratio", "alpha_pct", "max_drawdown_pct", "expense_ratio_pct", "composite_score"]
        st.dataframe(score.sort_values("composite_score", ascending=False)[score_cols].head(10), use_container_width=True, hide_index=True, height=255)

    tabs = st.tabs(["CAGR", "Sharpe & Sortino", "Alpha/Beta", "Drawdown", "Tracking Error"])
    with tabs[0]:
        st.dataframe(cagr.sort_values("cagr_3yr_pct", ascending=False).head(20), use_container_width=True, hide_index=True, height=260)
    with tabs[1]:
        rank_cols = ["scheme_name", "fund_house", "sharpe_ratio", "sortino_ratio", "risk_category"]
        st.dataframe(score.sort_values("sharpe_ratio", ascending=False)[rank_cols].head(20), use_container_width=True, hide_index=True, height=260)
    with tabs[2]:
        st.dataframe(alpha_beta.sort_values("alpha_pct", ascending=False).head(20), use_container_width=True, hide_index=True, height=260)
    with tabs[3]:
        dd_cols = ["scheme_name", "fund_house", "max_drawdown_pct", "drawdown_start_date", "drawdown_end_date"]
        st.dataframe(drawdown.sort_values("max_drawdown_pct").head(20)[dd_cols], use_container_width=True, hide_index=True, height=260)
    with tabs[4]:
        st.dataframe(tracking.sort_values(["benchmark", "tracking_error_pct"]), use_container_width=True, hide_index=True, height=260)

    distribution_chart = PERFORMANCE_CHART_DIR / "daily_return_distribution.png"
    if distribution_chart.exists():
        with st.expander("View daily return distribution chart"):
            st.image(str(distribution_chart), use_container_width=True)


def eda_gallery_page() -> None:
    """Render Day 3 EDA exported charts inside Streamlit."""
    header("EDA Analysis")
    if not EDA_CHART_DIR.exists():
        st.warning("EDA chart folder not found. Run `python scripts\\generate_eda.py` first.")
        return

    chart_groups = [
        (
            "NAV Trend Analysis",
            [
                ("All 40 Schemes NAV Trend", "01_nav_trend_all_40_plotly.png"),
                ("Top 10 AUM Funds NAV Trend", "02_top10_aum_nav_trend_plotly.png"),
            ],
        ),
        (
            "AUM & SIP Trends",
            [
                ("AUM Growth by Fund House", "03_aum_growth_by_fund_house_seaborn.png"),
                ("SIP Inflow Time Series", "04_sip_inflow_trend_plotly.png"),
            ],
        ),
        (
            "Category, Investor & Geography",
            [
                ("Category Inflow Heatmap", "05_category_inflow_heatmap_seaborn.png"),
                ("Age Group Distribution", "06_age_group_distribution_pie.png"),
                ("SIP Amount by Age Group", "07_sip_amount_box_by_age_group.png"),
                ("Gender Split", "08_gender_split_pie.png"),
                ("SIP Amount by State", "09_sip_amount_by_state.png"),
                ("T30 vs B30 City Tier Split", "10_city_tier_split_pie.png"),
            ],
        ),
        (
            "Folios, Correlation & Sector Allocation",
            [
                ("Folio Count Growth", "11_folio_count_growth.png"),
                ("NAV Return Correlation Matrix", "12_nav_return_correlation_matrix.png"),
                ("Sector Allocation Donut", "13_sector_allocation_donut.png"),
            ],
        ),
        (
            "Additional EDA Charts",
            [
                ("Monthly Transaction Volume by Type", "14_monthly_transaction_volume_by_type.png"),
                ("Total Net Inflow by Category", "15_total_net_inflow_by_category.png"),
                ("Top 10 Funds by 3-Year Return", "16_top10_3yr_return_funds.png"),
            ],
        ),
    ]

    total_pngs = len(list(EDA_CHART_DIR.glob("*.png")))
    c1, c2, c3 = st.columns(3)
    c1.metric("EDA PNG Charts", total_pngs)
    c2.metric("Notebook", "EDA_Analysis.ipynb")
    c3.metric("Findings", "10 insights")

    for group_title, charts in chart_groups:
        st.subheader(group_title)
        columns_per_row = 3 if len(charts) >= 3 else 2
        for i in range(0, len(charts), columns_per_row):
            cols = st.columns(columns_per_row)
            for col, (caption, file_name) in zip(cols, charts[i : i + columns_per_row]):
                path = EDA_CHART_DIR / file_name
                with col:
                    if path.exists():
                        st.image(str(path), use_container_width=True)
                    else:
                        st.error(f"Missing chart: {file_name}")

def model_report_page() -> None:
    """Show optional ML model predictions and final report download."""
    header("Model & Report")

    metrics_path = MODEL_DIR / "fund_return_model_metrics.json"
    predictions_path = MODEL_DIR / "fund_return_model_predictions.csv"
    pdf_path = REPORTS_DIR / "Bluestock_MF_Enhanced_Report.pdf"

    if not metrics_path.exists() or not predictions_path.exists():
        st.warning("Model outputs are missing. Run `python scripts\\train_ml_model.py` first.")
        return

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(predictions_path)
    model_name = metrics.get("model", "RandomForestRegressor")
    target = metrics.get("target", "cagr_3yr_pct")

    c1, c2, c3 = st.columns(3)
    c1.metric("Model", model_name)
    c2.metric("MAE", f"{metrics.get('mae', 0):.2f} pts")
    c3.metric("R2 Score", f"{metrics.get('r2', 0):.2f}")

    comparison_metrics, comparison_predictions = compare_ml_models()
    if not comparison_metrics.empty:
        st.subheader("Model Accuracy Comparison")
        m_left, m_right = st.columns(2)
        with m_left:
            shown_metrics = comparison_metrics.copy()
            shown_metrics[["mae_points", "r2_score"]] = shown_metrics[["mae_points", "r2_score"]].round(3)
            st.dataframe(shown_metrics, use_container_width=True, hide_index=True, height=130)
        with m_right:
            best_model = comparison_metrics.iloc[0]
            st.metric("Better Model", best_model["model"])

    chart_df = predictions.dropna(subset=["actual_cagr_3yr_pct", "predicted_cagr_3yr_pct"]).copy()
    if not chart_df.empty:
        fig, ax = plt.subplots(figsize=(5.8, 2.25), facecolor="#000000")
        ax.set_facecolor("#0b0b0b")
        ax.scatter(
            chart_df["actual_cagr_3yr_pct"],
            chart_df["predicted_cagr_3yr_pct"],
            s=72,
            color="#00e5ff",
            edgecolor="#ffffff",
            linewidth=0.7,
            alpha=0.88,
        )
        low = min(chart_df["actual_cagr_3yr_pct"].min(), chart_df["predicted_cagr_3yr_pct"].min())
        high = max(chart_df["actual_cagr_3yr_pct"].max(), chart_df["predicted_cagr_3yr_pct"].max())
        ax.plot([low, high], [low, high], color="#ffd740", linewidth=1.4, linestyle="--", label="Perfect prediction")
        ax.set_title("Actual vs Predicted 3-Year CAGR", color="#f0f0f0", fontweight="bold")
        ax.set_xlabel("Actual CAGR (%)", color="#d1d5db")
        ax.set_ylabel("Predicted CAGR (%)", color="#d1d5db")
        ax.tick_params(colors="#9ca3af", labelsize=7)
        ax.grid(True, color="#262626", linewidth=0.7)
        ax.legend(facecolor="#0b0b0b", edgecolor="#333333", labelcolor="#f0f0f0")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    display_cols = [
        "scheme_name",
        "fund_house",
        "actual_cagr_3yr_pct",
        "predicted_cagr_3yr_pct",
        "sharpe_ratio",
        "alpha_pct",
        "max_drawdown_pct",
    ]
    available_cols = [col for col in display_cols if col in predictions.columns]
    shown = predictions[available_cols].copy()
    numeric_cols = shown.select_dtypes(include=[np.number]).columns
    shown[numeric_cols] = shown[numeric_cols].round(2)
    st.dataframe(shown.sort_values("predicted_cagr_3yr_pct", ascending=False), use_container_width=True, hide_index=True, height=260)

    if not comparison_predictions.empty:
        with st.expander("Compare predictions from both ML models"):
            compare_view = comparison_predictions.rename(columns={"cagr_3yr_pct": "actual_cagr_3yr_pct"}).copy()
            num_cols = compare_view.select_dtypes(include=[np.number]).columns
            compare_view[num_cols] = compare_view[num_cols].round(2)
            st.dataframe(compare_view.sort_values(["scheme_name", "model"]), use_container_width=True, hide_index=True, height=260)

    st.subheader("Download Final Report")
    if pdf_path.exists():
        st.download_button(
            "Download PDF ",
            data=pdf_path.read_bytes(),
            file_name="Bluestock_MF_Enhanced_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.warning("Final report PDF is missing. Run `python scripts\\generate_pdf_report.py` from the project folder.")


def powerbi_export_page() -> None:
    """Show Day 5 Power BI deliverables generated from the cleaned datasets."""
    header("Power BI Export")

    page_dir = REPORTS_DIR / "day5_dashboard_pages"
    pdf_path = REPORTS_DIR / "Dashboard.pdf"

    c1, c2, c3 = st.columns(3)
    c1.metric("Pages", "4")
    c2.metric("Export", "PDF")
    c3.metric("Screenshots", "PNG")

    if pdf_path.exists():
        st.download_button(
            "Download PDF",
            data=pdf_path.read_bytes(),
            file_name="Dashboard.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.warning("Dashboard PDF is missing. Run `python scripts\\generate_day5_dashboard_assets.py` from the project folder.")

    page_files = [
        ("Page 1 - Industry Overview", "page_1_industry_overview.png"),
        ("Page 2 - Fund Performance", "page_2_fund_performance.png"),
        ("Page 3 - Investor Analytics", "page_3_investor_analytics.png"),
        ("Page 4 - SIP & Market Trends", "page_4_sip_market_trends.png"),
    ]

    st.subheader("Page Screenshots")
    for i in range(0, len(page_files), 2):
        cols = st.columns(2)
        for col, (caption, file_name) in zip(cols, page_files[i : i + 2]):
            path = page_dir / file_name
            with col:
                if path.exists():
                    st.image(str(path), use_container_width=True)
                    st.download_button(
                        "Download PNG",
                        data=path.read_bytes(),
                        file_name=file_name,
                        mime="image/png",
                        use_container_width=True,
                        key=f"download_{file_name}",
                    )
                else:
                    st.error(f"Missing screenshot: {file_name}")


def main() -> None:
    data = load_data()
    require_outputs(data)
    filters = sidebar(data)
    data.update(load_page_data(filters["page"]))
    if filters["page"] == "Industry Overview":
        industry_page(data, filters)
    elif filters["page"] == "Data Quality":
        data_quality_page()
    elif filters["page"] == "Fund Performance":
        fund_page(data, filters)
    elif filters["page"] == "Performance Analytics":
        performance_analytics_page()
    elif filters["page"] == "Investor Analytics":
        investor_page(data)
    elif filters["page"] == "SIP & Market Trends":
        sip_page(data)
    elif filters["page"] == "EDA Analysis":
        eda_gallery_page()
    elif filters["page"] == "Prediction & Portfolio":
        prediction_page(data, filters)
    elif filters["page"] == "Model & Report":
        model_report_page()
    elif filters["page"] == "Power BI Export":
        powerbi_export_page()
    else:
        recommender_page(data, filters)


if __name__ == "__main__":
    main()

