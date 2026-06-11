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