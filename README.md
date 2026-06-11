# Bluestock Mutual Fund Analytics Capstone

End-to-end FinTech analytics project for mutual fund performance, industry growth, investor behaviour, advanced risk analytics and dashboard reporting.

Prepared by: Yash Vardhan Bansal

## Project Overview

This project ingests 10 mutual fund datasets, cleans and validates them, loads them into SQLite, computes risk-return analytics and presents the results in a Streamlit dashboard. The project also produces notebooks, CSV analytics outputs, validation reports, Power BI-style screenshots and a final PDF report.

Key capabilities:

- ETL pipeline using Pandas and SQLAlchemy.
- SQLite star schema with `dim_fund`, `dim_date` and fact tables.
- EDA charts for NAV, AUM, SIP, investor, geography, category inflow and sector allocation.
- Performance analytics: CAGR, Sharpe, Sortino, Alpha, Beta, Maximum Drawdown and Tracking Error.
- Advanced analytics: Historical VaR/CVaR, rolling 90-day Sharpe, investor cohorts, SIP continuity, recommender and sector HHI.
- Streamlit dashboard with interactive slicers and pages for industry, funds, investors, SIP trends, data quality, performance, EDA, prediction and exports.
- Final report and validation checks.

## Folder Structure

```text
Bluestock_mf/
├── api/
├── dashboard/
├── data/
│   ├── raw/
│   ├── processed/
│   └── db/
├── models/
├── notebooks/
├── reports/
├── scripts/
├── sql/
├── requirements.txt
└── README.md
```

## Setup Instructions

Create and activate a virtual environment:

```powershell
cd C:\Users\hp\Desktop\Bluestock_mf
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## How To Run The ETL

Run only the ETL pipeline:

```powershell
python scripts\etl_pipeline.py
```

This creates cleaned files in `data/processed/` and loads the SQLite database in `data/db/`.

## How To Run The Full Pipeline

Run the master project pipeline:

```powershell
python scripts\run_pipeline.py
```

Optional live NAV fetch:

```powershell
python scripts\run_pipeline.py --include-live-nav
```

The full pipeline runs ETL, metrics, EDA, performance analytics, advanced analytics, ML model generation, dashboard export assets, final PDF report and validation scripts.

## How To Open The Dashboard

Start the Streamlit dashboard:

```powershell
streamlit run dashboard\streamlit_app.py
```

Main dashboard pages:

- Industry Overview
- Fund Performance
- Investor Analytics
- SIP & Market Trends
- Data Quality
- Performance Analytics
- EDA Analysis
- Prediction & Portfolio
- Model & Report
- Power BI Export
- Fund Recommender

## Dataset Descriptions

| File | Description |
| --- | --- |
| `01_fund_master.csv` | Fund metadata: AMFI code, fund house, scheme name, category, sub-category, plan, benchmark, risk and expense details. |
| `02_nav_history.csv` | Historical NAV and date series used for daily returns and performance metrics. |
| `03_aum_by_fund_house.csv` | Quarterly AUM by AMC with scheme counts. |
| `04_monthly_sip_inflows.csv` | Monthly SIP inflow, SIP account and SIP AUM trend data. |
| `05_category_inflows.csv` | Category-level net inflows by month. |
| `06_industry_folio_count.csv` | Industry folio counts by segment. |
| `07_scheme_performance.csv` | Scheme-level return, risk, AUM, expense and rating data. |
| `08_investor_transactions.csv` | Investor transaction records with state, city tier, age group, gender, transaction type and amount. |
| `09_portfolio_holdings.csv` | Fund holdings, sector exposure, stock weights and market values. |
| `10_benchmark_indices.csv` | Benchmark index values and returns for NIFTY50, NIFTY100 and other indices. |

## Key Outputs

- `data/processed/fund_scorecard.csv`
- `data/processed/var_cvar_report.csv`
- `data/processed/rolling_sharpe_90d.csv`
- `data/processed/cohort_analysis.csv`
- `data/processed/sip_continuity.csv`
- `data/processed/sector_hhi.csv`
- `reports/Bluestock_MF_Final_Report.pdf`
- `reports/Dashboard.pdf`
- `reports/analysis_validation_report.md`
- `reports/chart_validation_report.md`
- `notebooks/Advanced_Analytics.ipynb`
- `notebooks/Performance_Analytics.ipynb`
- `notebooks/EDA_Analysis.ipynb`

## Validation

Run analysis validation:

```powershell
python scripts\validate_analysis_outputs.py
```

Run chart validation:

```powershell
python scripts\validate_chart_outputs.py
```

## Important Notes

- File paths use `pathlib.Path`; local absolute paths are avoided inside reusable scripts.
- NAV is forward-filled after daily reindexing to handle weekends and holidays.
- CAGR and annualised metrics use 252 trading days.
- AUM fields preserve units: `aum_crore` and `aum_lakh_crore`.
- SQLite `.db` files are ignored by Git; schema and scripts should be shared instead.
- The dashboard is an interactive Streamlit alternative to Power BI. A real `.pbix` must be created in Power BI Desktop.
