-- 1. Top 5 funds by scheme-level AUM
SELECT scheme_name, fund_house, aum_crore
FROM fact_performance
ORDER BY aum_crore DESC
LIMIT 5;

-- 2. Average observed NAV per month
SELECT amfi_code, substr(nav_date, 1, 7) AS month, ROUND(AVG(nav), 2) AS avg_nav
FROM fact_nav
WHERE is_observed_nav = 1
GROUP BY amfi_code, month
ORDER BY month, amfi_code;

-- 3. SIP inflow YoY growth
SELECT month, sip_inflow_crore, yoy_growth_pct
FROM fact_sip
ORDER BY month;

-- 4. Transactions by state
SELECT state, COUNT(*) AS txn_count, SUM(amount_inr) AS total_amount_inr
FROM fact_transactions
GROUP BY state
ORDER BY total_amount_inr DESC;

-- 5. Funds with expense ratio below 1%
SELECT amfi_code, scheme_name, plan, expense_ratio_pct
FROM dim_fund
WHERE expense_ratio_pct < 1
ORDER BY expense_ratio_pct, scheme_name;

-- 6. Latest AUM by fund house
SELECT fund_house, aum_date, aum_lakh_crore, aum_crore, num_schemes
FROM fact_aum
WHERE aum_date = (SELECT MAX(aum_date) FROM fact_aum)
ORDER BY aum_crore DESC;

-- 7. Monthly transaction volume and value
SELECT substr(transaction_date, 1, 7) AS month,
       COUNT(*) AS txn_count,
       SUM(amount_inr) AS total_amount_inr
FROM fact_transactions
GROUP BY month
ORDER BY month;

-- 8. Average 3-year return by fund category and plan
SELECT category, plan,
       ROUND(AVG(return_3yr_pct), 2) AS avg_return_3yr_pct,
       ROUND(AVG(expense_ratio_pct), 2) AS avg_expense_ratio_pct,
       COUNT(*) AS funds
FROM fact_performance
GROUP BY category, plan
ORDER BY avg_return_3yr_pct DESC;

-- 9. Redemption amount share by state
SELECT state,
       SUM(CASE WHEN transaction_type = 'Redemption' THEN amount_inr ELSE 0 END) AS redemption_amount_inr,
       SUM(amount_inr) AS total_amount_inr,
       ROUND(100.0 * SUM(CASE WHEN transaction_type = 'Redemption' THEN amount_inr ELSE 0 END) / SUM(amount_inr), 2) AS redemption_share_pct
FROM fact_transactions
GROUP BY state
ORDER BY redemption_share_pct DESC, total_amount_inr DESC;

-- 10. NAV rows added by weekend/holiday forward-fill
SELECT amfi_code,
       COUNT(*) AS full_calendar_rows,
       SUM(is_observed_nav) AS observed_source_rows,
       COUNT(*) - SUM(is_observed_nav) AS forward_filled_rows
FROM fact_nav
GROUP BY amfi_code
ORDER BY forward_filled_rows DESC;
