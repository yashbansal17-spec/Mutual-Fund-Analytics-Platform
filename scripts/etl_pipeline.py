"""Day 2 ETL: clean raw datasets, create star schema, and load SQLite."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DB_DIR = PROJECT_ROOT / "data" / "db"
DB_PATH = DB_DIR / "bluestock_mf.db"

RAW_FILES = {
    "fund_master": "01_fund_master.csv",
    "nav_history": "02_nav_history.csv",
    "aum_by_fund_house": "03_aum_by_fund_house.csv",
    "monthly_sip_inflows": "04_monthly_sip_inflows.csv",
    "category_inflows": "05_category_inflows.csv",
    "industry_folio_count": "06_industry_folio_count.csv",
    "scheme_performance": "07_scheme_performance.csv",
    "investor_transactions": "08_investor_transactions.csv",
    "portfolio_holdings": "09_portfolio_holdings.csv",
    "benchmark_indices": "10_benchmark_indices.csv",
}

TABLE_TO_RAW = {
    "dim_fund": "fund_master",
    "fact_nav": "nav_history",
    "fact_aum": "aum_by_fund_house",
    "fact_sip": "monthly_sip_inflows",
    "fact_category_inflows": "category_inflows",
    "fact_folios": "industry_folio_count",
    "fact_performance": "scheme_performance",
    "fact_transactions": "investor_transactions",
    "fact_holdings": "portfolio_holdings",
    "fact_benchmark": "benchmark_indices",
}

DATE_TABLES = {
    "dim_fund": ["launch_date"],
    "fact_nav": ["nav_date"],
    "fact_aum": ["aum_date"],
    "fact_sip": ["month"],
    "fact_category_inflows": ["month"],
    "fact_folios": ["month"],
    "fact_transactions": ["transaction_date"],
    "fact_holdings": ["portfolio_date"],
    "fact_benchmark": ["bench_date"],
}

TXN_TYPE_MAP = {
    "sip": "SIP",
    "systematic investment plan": "SIP",
    "lumpsum": "Lumpsum",
    "lump sum": "Lumpsum",
    "redemption": "Redemption",
    "redeem": "Redemption",
}
VALID_TXN_TYPES = {"SIP", "Lumpsum", "Redemption"}
VALID_KYC = {"Verified", "Pending", "Rejected"}
RETURN_COLUMNS = [
    "return_1yr_pct",
    "return_3yr_pct",
    "return_5yr_pct",
    "benchmark_3yr_pct",
    "alpha",
    "beta",
    "sharpe_ratio",
    "sortino_ratio",
    "std_dev_ann_pct",
    "max_drawdown_pct",
]


def load_raw() -> dict[str, pd.DataFrame]:
    """Load the 10 provided CSVs and print basic profiling output."""
    frames: dict[str, pd.DataFrame] = {}
    for name, file_name in RAW_FILES.items():
        path = RAW_DIR / file_name
        df = pd.read_csv(path)
        frames[name] = df
        print(f"\n{name} | {file_name}")
        print(f"shape: {df.shape}")
        print("dtypes:")
        print(df.dtypes)
        print("head:")
        print(df.head())
    return frames


def parse_dates(df: pd.DataFrame, columns: list[str], month_cols: set[str] | None = None) -> pd.DataFrame:
    """Parse date/month columns consistently."""
    out = df.copy()
    month_cols = month_cols or set()
    for col in columns:
        if col not in out.columns:
            continue
        out[col] = pd.to_datetime(out[col], errors="coerce")
        if col in month_cols:
            out[col] = out[col].dt.to_period("M").dt.to_timestamp()
    return out


def add_date_key(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Add a text date key for dim_date relationships."""
    out = df.copy()
    out["date_key"] = out[date_col].dt.strftime("%Y-%m-%d")
    return out


def clean_fund_master(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Clean fund master and validate AMFI uniqueness."""
    issues = []
    out = parse_dates(df, ["launch_date"])
    if out["amfi_code"].duplicated().any():
        issues.append(f"Duplicate AMFI codes: {out.loc[out['amfi_code'].duplicated(), 'amfi_code'].tolist()}")
    out = out.drop_duplicates("amfi_code").sort_values("amfi_code")
    return out, issues


def clean_nav_history(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Clean NAV history, validate NAV > 0, and forward-fill holidays/weekends."""
    issues = []
    nav = parse_dates(df, ["date"]).dropna(subset=["date"])
    before = len(nav)
    nav = nav.drop_duplicates(["amfi_code", "date"]).sort_values(["amfi_code", "date"])
    duplicates_removed = before - len(nav)
    if duplicates_removed:
        issues.append(f"Removed {duplicates_removed} duplicate NAV rows.")
    invalid_nav = nav[nav["nav"].le(0) | nav["nav"].isna()]
    if not invalid_nav.empty:
        issues.append(f"Invalid NAV rows removed: {len(invalid_nav)}")
        nav = nav[nav["nav"].gt(0)].copy()

    cleaned = []
    for amfi_code, group in nav.groupby("amfi_code", sort=False):
        group = group.set_index("date").sort_index()
        full_idx = pd.date_range(group.index.min(), group.index.max(), freq="D")
        full = group.reindex(full_idx)
        full["amfi_code"] = amfi_code
        full["is_observed_nav"] = full["nav"].notna().astype(int)
        full["nav"] = full["nav"].ffill()
        full["daily_return"] = full["nav"].pct_change()
        full = full.reset_index().rename(columns={"index": "nav_date"})
        full = add_date_key(full, "nav_date")
        cleaned.append(full[["amfi_code", "date_key", "nav_date", "nav", "daily_return", "is_observed_nav"]])

    out = pd.concat(cleaned, ignore_index=True).sort_values(["amfi_code", "nav_date"])
    if out["nav"].le(0).any():
        raise ValueError("NAV validation failed: non-positive values remain after cleaning.")
    return out, issues


def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Clean investor transactions and enforce transaction/KYC enums."""
    issues = []
    out = parse_dates(df, ["transaction_date"]).dropna(subset=["transaction_date"]).copy()
    out["transaction_type"] = (
        out["transaction_type"].astype(str).str.strip().str.lower().map(TXN_TYPE_MAP).fillna(out["transaction_type"].astype(str).str.strip())
    )
    invalid_type = sorted(set(out["transaction_type"]) - VALID_TXN_TYPES)
    if invalid_type:
        issues.append(f"Invalid transaction_type values: {invalid_type}")
    out = out[out["transaction_type"].isin(VALID_TXN_TYPES)].copy()

    out["amount_inr"] = pd.to_numeric(out["amount_inr"], errors="coerce")
    invalid_amount = out[out["amount_inr"].le(0) | out["amount_inr"].isna()]
    if not invalid_amount.empty:
        issues.append(f"Transactions with invalid amount removed: {len(invalid_amount)}")
        out = out[out["amount_inr"].gt(0)].copy()

    out["kyc_status"] = out["kyc_status"].astype(str).str.strip().str.title()
    invalid_kyc = sorted(set(out["kyc_status"]) - VALID_KYC)
    if invalid_kyc:
        issues.append(f"Invalid KYC status values: {invalid_kyc}")
    out = out[out["kyc_status"].isin(VALID_KYC)].copy()
    out["amount_inr"] = out["amount_inr"].astype(int)
    out = add_date_key(out, "transaction_date")
    return out.sort_values(["transaction_date", "investor_id", "amfi_code"]), issues


def clean_performance(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Clean performance table and flag numeric/range anomalies."""
    issues = []
    out = df.copy()
    for col in RETURN_COLUMNS + ["expense_ratio_pct", "aum_crore"]:
        before_na = out[col].isna().sum()
        out[col] = pd.to_numeric(out[col], errors="coerce")
        new_na = out[col].isna().sum()
        if new_na > before_na:
            issues.append(f"{col}: {new_na - before_na} non-numeric values coerced to NaN.")

    expense_bad = out[~out["expense_ratio_pct"].between(0.1, 2.5)]
    if not expense_bad.empty:
        issues.append(f"Expense ratio outside 0.1%-2.5%: {len(expense_bad)} rows.")

    return_bad = out[
        out[["return_1yr_pct", "return_3yr_pct", "return_5yr_pct", "benchmark_3yr_pct"]].lt(-100).any(axis=1)
        | out[["return_1yr_pct", "return_3yr_pct", "return_5yr_pct", "benchmark_3yr_pct"]].gt(100).any(axis=1)
    ]
    if not return_bad.empty:
        issues.append(f"Return anomaly outside -100% to 100%: {len(return_bad)} rows.")

    anomaly = out[expense_bad.index.intersection(out.index).union(return_bad.index)]
    anomaly.to_csv(PROCESSED_DIR / "scheme_performance_anomalies.csv", index=False)
    return out.sort_values("amfi_code"), issues


def clean_benchmarks(df: pd.DataFrame) -> pd.DataFrame:
    """Clean benchmark index prices and add daily returns."""
    out = parse_dates(df, ["date"]).dropna(subset=["date"])
    out = out.drop_duplicates(["date", "index_name"]).sort_values(["index_name", "date"])
    out["close_value"] = pd.to_numeric(out["close_value"], errors="coerce")
    out = out[out["close_value"].gt(0)].copy()
    out["daily_return"] = out.groupby("index_name")["close_value"].pct_change()
    out = out.rename(columns={"date": "bench_date"})
    return add_date_key(out, "bench_date")


def clean_generic_with_date(df: pd.DataFrame, source_date: str, target_date: str, month: bool = False) -> pd.DataFrame:
    """Clean a simple dated table and add date_key."""
    out = parse_dates(df, [source_date], {source_date} if month else set()).rename(columns={source_date: target_date})
    out = out.drop_duplicates().sort_values(target_date)
    return add_date_key(out, target_date)


def build_dim_date(clean: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build date dimension from every cleaned date column."""
    dates = []
    for table, cols in DATE_TABLES.items():
        if table not in clean:
            continue
        for col in cols:
            if col in clean[table].columns:
                dates.append(clean[table][col])
    all_dates = pd.concat(dates, ignore_index=True).dropna().drop_duplicates().sort_values()
    dim = pd.DataFrame({"full_date": all_dates})
    dim["date_key"] = dim["full_date"].dt.strftime("%Y-%m-%d")
    dim["year"] = dim["full_date"].dt.year
    dim["quarter"] = dim["full_date"].dt.quarter
    dim["month"] = dim["full_date"].dt.month
    dim["month_name"] = dim["full_date"].dt.month_name()
    dim["day"] = dim["full_date"].dt.day
    dim["day_of_week"] = dim["full_date"].dt.day_name()
    dim["is_weekend"] = dim["full_date"].dt.dayofweek.isin([5, 6]).astype(int)
    return dim[["date_key", "full_date", "year", "quarter", "month", "month_name", "day", "day_of_week", "is_weekend"]]


def clean_frames(frames: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict[str, list[str]]]:
    """Clean all source tables and collect validation issues."""
    issues: dict[str, list[str]] = {}
    clean: dict[str, pd.DataFrame] = {}
    clean["dim_fund"], issues["fund_master"] = clean_fund_master(frames["fund_master"])
    clean["fact_nav"], issues["nav_history"] = clean_nav_history(frames["nav_history"])
    clean["fact_aum"] = clean_generic_with_date(frames["aum_by_fund_house"], "date", "aum_date")
    clean["fact_sip"] = clean_generic_with_date(frames["monthly_sip_inflows"], "month", "month", month=True)
    clean["fact_category_inflows"] = clean_generic_with_date(frames["category_inflows"], "month", "month", month=True)
    clean["fact_folios"] = clean_generic_with_date(frames["industry_folio_count"], "month", "month", month=True)
    clean["fact_performance"], issues["scheme_performance"] = clean_performance(frames["scheme_performance"])
    clean["fact_transactions"], issues["investor_transactions"] = clean_transactions(frames["investor_transactions"])
    clean["fact_holdings"] = clean_generic_with_date(frames["portfolio_holdings"], "portfolio_date", "portfolio_date")
    clean["fact_benchmark"] = clean_benchmarks(frames["benchmark_indices"])
    clean["dim_date"] = build_dim_date(clean)
    return clean, issues


def write_processed(clean: dict[str, pd.DataFrame], frames: dict[str, pd.DataFrame]) -> None:
    """Write table CSVs plus 10 cleaned source-style CSVs."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    table_order = [
        "dim_fund",
        "dim_date",
        "fact_nav",
        "fact_transactions",
        "fact_performance",
        "fact_aum",
        "fact_holdings",
        "fact_benchmark",
        "fact_sip",
        "fact_category_inflows",
        "fact_folios",
    ]
    for name in table_order:
        clean[name].to_csv(PROCESSED_DIR / f"{name}.csv", index=False)

    cleaned_file_map = {
        "01_fund_master_clean.csv": clean["dim_fund"],
        "02_nav_history_clean.csv": clean["fact_nav"],
        "03_aum_by_fund_house_clean.csv": clean["fact_aum"],
        "04_monthly_sip_inflows_clean.csv": clean["fact_sip"],
        "05_category_inflows_clean.csv": clean["fact_category_inflows"],
        "06_industry_folio_count_clean.csv": clean["fact_folios"],
        "07_scheme_performance_clean.csv": clean["fact_performance"],
        "08_investor_transactions_clean.csv": clean["fact_transactions"],
        "09_portfolio_holdings_clean.csv": clean["fact_holdings"],
        "10_benchmark_indices_clean.csv": clean["fact_benchmark"],
    }
    for file_name, df in cleaned_file_map.items():
        df.to_csv(PROCESSED_DIR / file_name, index=False)


def validate_data_quality(frames: dict[str, pd.DataFrame], clean: dict[str, pd.DataFrame], issues: dict[str, list[str]]) -> None:
    """Write data-quality and validation summaries."""
    fund_codes = set(frames["fund_master"]["amfi_code"])
    nav_codes = set(frames["nav_history"]["amfi_code"])
    missing_nav = sorted(fund_codes - nav_codes)
    observed_nav_rows = int(clean["fact_nav"]["is_observed_nav"].sum())
    source_nav_rows = len(frames["nav_history"].drop_duplicates(["amfi_code", "date"]))

    lines = [
        "# Data Quality Summary",
        "",
        f"- Fund master schemes: {len(fund_codes)}",
        f"- Fund master duplicate AMFI codes: {int(frames['fund_master']['amfi_code'].duplicated().sum())}",
        f"- AMFI codes missing from NAV history: {missing_nav if missing_nav else 'None'}",
        f"- Source NAV rows after duplicate removal: {source_nav_rows:,}",
        f"- Observed NAV rows retained: {observed_nav_rows:,}",
        f"- Full forward-filled NAV rows: {len(clean['fact_nav']):,}",
        f"- Investor transaction types: {sorted(clean['fact_transactions']['transaction_type'].unique())}",
        f"- KYC statuses: {sorted(clean['fact_transactions']['kyc_status'].unique())}",
        f"- Scheme performance expense ratio range: {clean['fact_performance']['expense_ratio_pct'].min():.2f}% to {clean['fact_performance']['expense_ratio_pct'].max():.2f}%",
        "",
        "## Validation Issues",
    ]
    any_issue = False
    for source, source_issues in issues.items():
        if source_issues:
            any_issue = True
            lines.append(f"- {source}: " + "; ".join(source_issues))
    if not any_issue:
        lines.append("- No blocking validation issues found.")
    lines.append("")
    lines.append("NAV was sorted by AMFI code/date, duplicate dates per fund were removed, and missing calendar days were forward-filled for weekends/holidays.")
    (PROCESSED_DIR / "data_quality_summary.md").write_text("\n".join(lines), encoding="utf-8")


def sqlite_value(value: object) -> object:
    """Convert pandas values into SQLite-friendly scalar values."""
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def load_sqlite(clean: dict[str, pd.DataFrame], frames: dict[str, pd.DataFrame]) -> None:
    """Create SQLite database and load cleaned tables with SQLAlchemy."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    schema_sql = (PROJECT_ROOT / "sql" / "schema.sql").read_text(encoding="utf-8")
    engine = create_engine(f"sqlite:///{DB_PATH}")
    raw_conn = engine.raw_connection()
    try:
        raw_conn.executescript(schema_sql)
        raw_conn.commit()
    finally:
        raw_conn.close()

    table_order = [
        "dim_fund",
        "dim_date",
        "fact_nav",
        "fact_transactions",
        "fact_performance",
        "fact_aum",
        "fact_holdings",
        "fact_benchmark",
        "fact_sip",
        "fact_category_inflows",
        "fact_folios",
    ]
    for table in table_order:
        clean[table].to_sql(table, engine, if_exists="append", index=False)

    verification = []
    with engine.connect() as conn:
        for table in table_order:
            db_rows = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()
            raw_name = TABLE_TO_RAW.get(table)
            raw_rows = len(frames[raw_name]) if raw_name else len(clean[table])
            verification.append(
                {
                    "table": table,
                    "source_dataset": raw_name or "generated",
                    "source_rows": raw_rows,
                    "cleaned_rows": len(clean[table]),
                    "db_rows": db_rows,
                    "row_count_note": "matches cleaned output" if db_rows == len(clean[table]) else "mismatch",
                }
            )
    pd.DataFrame(verification).to_csv(PROCESSED_DIR / "sqlite_row_count_verification.csv", index=False)
    print(f"\nSQLite loaded with SQLAlchemy: {DB_PATH}")


def create_data_dictionary(clean: dict[str, pd.DataFrame]) -> None:
    """Create a Markdown data dictionary with business definitions and sources."""
    definitions = {
        "amfi_code": "AMFI scheme identifier used to join fund-level datasets.",
        "date_key": "YYYY-MM-DD key linking fact tables to dim_date.",
        "nav": "Net Asset Value of the scheme. Must be greater than zero.",
        "daily_return": "Daily percentage change as a decimal, computed from NAV or benchmark close.",
        "is_observed_nav": "1 for original source NAV row, 0 for weekend/holiday forward-fill row.",
        "transaction_type": "Standardised investor action: SIP, Lumpsum, or Redemption.",
        "amount_inr": "Transaction amount in Indian rupees. Must be greater than zero.",
        "kyc_status": "Investor KYC status enum: Verified, Pending, Rejected.",
        "expense_ratio_pct": "Scheme expense ratio in percent. Validated between 0.1 and 2.5.",
        "aum_crore": "Assets under management in Rs crore.",
        "aum_lakh_crore": "Assets under management in Rs lakh crore.",
    }
    sources = {
        "dim_fund": "01_fund_master.csv",
        "dim_date": "Generated from all cleaned date columns",
        "fact_nav": "02_nav_history.csv",
        "fact_transactions": "08_investor_transactions.csv",
        "fact_performance": "07_scheme_performance.csv",
        "fact_aum": "03_aum_by_fund_house.csv",
        "fact_holdings": "09_portfolio_holdings.csv",
        "fact_benchmark": "10_benchmark_indices.csv",
        "fact_sip": "04_monthly_sip_inflows.csv",
        "fact_category_inflows": "05_category_inflows.csv",
        "fact_folios": "06_industry_folio_count.csv",
    }

    lines = ["# Data Dictionary", "", "All processed tables are generated by `scripts/etl_pipeline.py`.", ""]
    for table, df in clean.items():
        lines.extend([f"## {table}", f"Source: `{sources.get(table, 'Derived')}`", "", "| Column | Data Type | Business Definition |", "|---|---:|---|"])
        for col in df.columns:
            definition = definitions.get(col, col.replace("_", " ").capitalize())
            lines.append(f"| {col} | {df[col].dtype} | {definition} |")
        lines.append("")
    (PROCESSED_DIR / "data_dictionary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run full Day 2 ETL."""
    frames = load_raw()
    clean, issues = clean_frames(frames)
    write_processed(clean, frames)
    validate_data_quality(frames, clean, issues)
    create_data_dictionary(clean)
    load_sqlite(clean, frames)


if __name__ == "__main__":
    main()
