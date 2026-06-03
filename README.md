# Bluestock Mutual Fund Analytics Capstone

End-to-end mutual fund analytics platform for Bluestock Fintech. The project ingests 10 provided datasets, cleans and loads them into SQLite, computes risk and return metrics, and serves a Streamlit dashboard with industry, fund, investor, SIP, prediction, and portfolio-optimisation views.

## Objectives Covered

- ETL pipeline from raw AMFI-style datasets.
- SQLite schema with all raw datasets loaded into analysis-ready tables.
- EDA-ready processed CSVs and notebook placeholders.
- Performance metrics: CAGR, annualised return, Sharpe, Sortino, Alpha, Beta, Max Drawdown, VaR, CVaR, tracking error, information ratio.
- Streamlit dashboard alternative to Power BI with interactive slicers on every page.
- Advanced analytics: cohort analysis, SIP continuity, sector HHI, recommendations, Monte Carlo NAV projection, Markowitz efficient frontier.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Pipeline

```bash
python scripts/etl_pipeline.py
python scripts/compute_metrics.py
python scripts/live_nav_fetch.py
streamlit run dashboard/streamlit_app.py
```

## Folder Guide

- `data/raw/`: original CSVs and PDF brief.
- `data/processed/`: cleaned CSVs and analytics outputs.
- `data/db/`: local SQLite database. The `.db` file is ignored by Git.
- `scripts/`: ETL, live NAV fetcher, metrics, and recommender logic.
- `sql/`: database schema and 10 analytical queries.
- `dashboard/`: Streamlit dashboard.
- `notebooks/`: notebook placeholders for submission sections.
- `reports/`: final report and presentation can be added here.

## Important Modelling Choices

- Paths use `pathlib.Path`; no hard-coded local download paths.
- NAV is reindexed to a full daily range and forward-filled to handle weekends and market holidays.
- CAGR and annualised returns use observed trading days with `252 / n_trading_days`.
- AUM fields keep units explicit: `aum_lakh_crore` and `aum_crore`.
- SQLite `.db` files are excluded from GitHub; share `sql/schema.sql` and rerun the ETL.
