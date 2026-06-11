-- PostgreSQL business queries for Bluestock mutual fund analytics.
-- These assume the cleaned tables are loaded into PostgreSQL with the same names
-- as the processed CSVs / SQLite star schema.

-- 1. Which funds have the highest AUM?
SELECT scheme_name, fund_house, aum_crore
FROM fact_performance
ORDER BY aum_crore DESC
LIMIT 5;

-- 2. How has average NAV moved month by month for each scheme?
SELECT
    amfi_code,
    DATE_TRUNC('month', nav_date::date) AS month,
    ROUND(AVG(nav)::numeric, 2) AS avg_nav
FROM fact_nav
WHERE is_observed_nav = 1
GROUP BY amfi_code, DATE_TRUNC('month', nav_date::date)
ORDER BY month, amfi_code;

-- 3. Which states contribute the highest transaction value?
SELECT
    state,
    COUNT(*) AS txn_count,
    SUM(amount_inr) AS total_amount_inr
FROM fact_transactions
GROUP BY state
ORDER BY total_amount_inr DESC
LIMIT 10;

-- 4. Which low-cost funds have expense ratios below 1%?
SELECT amfi_code, scheme_name, fund_house, plan, expense_ratio_pct
FROM dim_fund
WHERE expense_ratio_pct < 1
ORDER BY expense_ratio_pct, scheme_name;

-- 5. How fast are SIP inflows growing year over year?
SELECT
    month::date,
    sip_inflow_crore,
    yoy_growth_pct
FROM fact_sip
ORDER BY month::date;

-- 6. Which categories attract the most net inflow?
SELECT
    category,
    SUM(net_inflow_crore) AS total_net_inflow_crore
FROM fact_category_inflows
GROUP BY category
ORDER BY total_net_inflow_crore DESC;

-- 7. Which funds rank best by composite performance score?
SELECT
    scheme_name,
    fund_house,
    cagr_3yr_pct,
    sharpe_ratio,
    alpha_pct,
    max_drawdown_pct,
    composite_score
FROM fund_scorecard
ORDER BY composite_score DESC
LIMIT 10;

-- 8. What is the redemption share by state?
SELECT
    state,
    SUM(CASE WHEN transaction_type = 'Redemption' THEN amount_inr ELSE 0 END) AS redemption_amount_inr,
    SUM(amount_inr) AS total_amount_inr,
    ROUND(
        100.0 * SUM(CASE WHEN transaction_type = 'Redemption' THEN amount_inr ELSE 0 END) / NULLIF(SUM(amount_inr), 0),
        2
    ) AS redemption_share_pct
FROM fact_transactions
GROUP BY state
ORDER BY redemption_share_pct DESC;
