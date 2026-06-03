PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS analytics_scorecard;
DROP TABLE IF EXISTS fact_folios;
DROP TABLE IF EXISTS fact_category_inflows;
DROP TABLE IF EXISTS fact_sip;
DROP TABLE IF EXISTS fact_benchmark;
DROP TABLE IF EXISTS fact_holdings;
DROP TABLE IF EXISTS fact_aum;
DROP TABLE IF EXISTS fact_performance;
DROP TABLE IF EXISTS fact_transactions;
DROP TABLE IF EXISTS fact_nav;
DROP TABLE IF EXISTS dim_date;
DROP TABLE IF EXISTS dim_fund;

CREATE TABLE dim_fund (
    amfi_code INTEGER PRIMARY KEY,
    fund_house TEXT NOT NULL,
    scheme_name TEXT NOT NULL,
    category TEXT,
    sub_category TEXT,
    plan TEXT,
    launch_date DATE,
    benchmark TEXT,
    expense_ratio_pct REAL CHECK (expense_ratio_pct BETWEEN 0.1 AND 2.5),
    exit_load_pct REAL,
    min_sip_amount INTEGER,
    min_lumpsum_amount INTEGER,
    fund_manager TEXT,
    risk_category TEXT,
    sebi_category_code TEXT
);

CREATE TABLE dim_date (
    date_key TEXT PRIMARY KEY,
    full_date DATE NOT NULL UNIQUE,
    year INTEGER NOT NULL,
    quarter INTEGER NOT NULL,
    month INTEGER NOT NULL,
    month_name TEXT NOT NULL,
    day INTEGER NOT NULL,
    day_of_week TEXT NOT NULL,
    is_weekend INTEGER NOT NULL CHECK (is_weekend IN (0, 1))
);

CREATE TABLE fact_nav (
    amfi_code INTEGER NOT NULL,
    date_key TEXT NOT NULL,
    nav_date DATE NOT NULL,
    nav REAL NOT NULL CHECK (nav > 0),
    daily_return REAL,
    is_observed_nav INTEGER NOT NULL CHECK (is_observed_nav IN (0, 1)),
    PRIMARY KEY (amfi_code, nav_date),
    FOREIGN KEY (amfi_code) REFERENCES dim_fund(amfi_code),
    FOREIGN KEY (date_key) REFERENCES dim_date(date_key)
);

CREATE TABLE fact_transactions (
    investor_id TEXT NOT NULL,
    transaction_date DATE NOT NULL,
    date_key TEXT NOT NULL,
    amfi_code INTEGER NOT NULL,
    transaction_type TEXT NOT NULL CHECK (transaction_type IN ('SIP', 'Lumpsum', 'Redemption')),
    amount_inr INTEGER NOT NULL CHECK (amount_inr > 0),
    state TEXT,
    city TEXT,
    city_tier TEXT,
    age_group TEXT,
    gender TEXT,
    annual_income_lakh REAL,
    payment_mode TEXT,
    kyc_status TEXT CHECK (kyc_status IN ('Verified', 'Pending', 'Rejected')),
    FOREIGN KEY (amfi_code) REFERENCES dim_fund(amfi_code),
    FOREIGN KEY (date_key) REFERENCES dim_date(date_key)
);

CREATE TABLE fact_performance (
    amfi_code INTEGER PRIMARY KEY,
    scheme_name TEXT,
    fund_house TEXT,
    category TEXT,
    plan TEXT,
    return_1yr_pct REAL,
    return_3yr_pct REAL,
    return_5yr_pct REAL,
    benchmark_3yr_pct REAL,
    alpha REAL,
    beta REAL,
    sharpe_ratio REAL,
    sortino_ratio REAL,
    std_dev_ann_pct REAL,
    max_drawdown_pct REAL,
    aum_crore INTEGER,
    expense_ratio_pct REAL CHECK (expense_ratio_pct BETWEEN 0.1 AND 2.5),
    morningstar_rating INTEGER,
    risk_grade TEXT,
    FOREIGN KEY (amfi_code) REFERENCES dim_fund(amfi_code)
);

CREATE TABLE fact_aum (
    aum_date DATE NOT NULL,
    date_key TEXT NOT NULL,
    fund_house TEXT NOT NULL,
    aum_lakh_crore REAL,
    aum_crore INTEGER,
    num_schemes INTEGER,
    PRIMARY KEY (aum_date, fund_house),
    FOREIGN KEY (date_key) REFERENCES dim_date(date_key)
);

CREATE TABLE fact_holdings (
    amfi_code INTEGER NOT NULL,
    stock_symbol TEXT,
    stock_name TEXT,
    sector TEXT,
    weight_pct REAL,
    market_value_cr REAL,
    current_price_inr REAL,
    portfolio_date DATE,
    date_key TEXT,
    FOREIGN KEY (amfi_code) REFERENCES dim_fund(amfi_code),
    FOREIGN KEY (date_key) REFERENCES dim_date(date_key)
);

CREATE TABLE fact_benchmark (
    bench_date DATE NOT NULL,
    date_key TEXT NOT NULL,
    index_name TEXT NOT NULL,
    close_value REAL NOT NULL,
    daily_return REAL,
    PRIMARY KEY (bench_date, index_name),
    FOREIGN KEY (date_key) REFERENCES dim_date(date_key)
);

CREATE TABLE fact_sip (
    month DATE PRIMARY KEY,
    date_key TEXT NOT NULL,
    sip_inflow_crore INTEGER,
    active_sip_accounts_crore REAL,
    new_sip_accounts_lakh REAL,
    sip_aum_lakh_crore REAL,
    yoy_growth_pct REAL,
    FOREIGN KEY (date_key) REFERENCES dim_date(date_key)
);

CREATE TABLE fact_category_inflows (
    month DATE NOT NULL,
    date_key TEXT NOT NULL,
    category TEXT NOT NULL,
    net_inflow_crore REAL,
    PRIMARY KEY (month, category),
    FOREIGN KEY (date_key) REFERENCES dim_date(date_key)
);

CREATE TABLE fact_folios (
    month DATE PRIMARY KEY,
    date_key TEXT NOT NULL,
    total_folios_crore REAL,
    equity_folios_crore REAL,
    debt_folios_crore REAL,
    hybrid_folios_crore REAL,
    others_folios_crore REAL,
    FOREIGN KEY (date_key) REFERENCES dim_date(date_key)
);

CREATE TABLE analytics_scorecard (
    amfi_code INTEGER PRIMARY KEY,
    scheme_name TEXT,
    fund_house TEXT,
    category TEXT,
    sub_category TEXT,
    plan TEXT,
    risk_category TEXT,
    cagr_pct REAL,
    annualised_return_pct REAL,
    annualised_volatility_pct REAL,
    sharpe_ratio REAL,
    sortino_ratio REAL,
    beta REAL,
    alpha_pct REAL,
    max_drawdown_pct REAL,
    var_95_pct REAL,
    cvar_95_pct REAL,
    tracking_error_pct REAL,
    information_ratio REAL,
    expense_ratio_pct REAL,
    aum_crore REAL,
    composite_score REAL,
    FOREIGN KEY (amfi_code) REFERENCES dim_fund(amfi_code)
);

CREATE INDEX idx_nav_code_date ON fact_nav(amfi_code, nav_date);
CREATE INDEX idx_txn_code_date ON fact_transactions(amfi_code, transaction_date);
CREATE INDEX idx_bench_name_date ON fact_benchmark(index_name, bench_date);
