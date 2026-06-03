"""Compute performance, risk, advanced analytics, and prediction outputs."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DB_PATH = PROJECT_ROOT / "data" / "db" / "bluestock_mf.db"
RF_ANNUAL = 0.065
TRADING_DAYS = 252

BENCHMARK_MAP = {
    "NIFTY 100 TRI": "NIFTY100",
    "BSE 250 SmallCap TRI": "BSE_SMALLCAP",
    "CRISIL Dynamic Gilt Index": "CRISIL_GILT",
    "NIFTY Midcap 150 TRI": "NIFTY_MIDCAP150",
    "CRISIL Short Term Bond Index": "CRISIL_GILT",
    "NIFTY 500 TRI": "NIFTY500",
    "CRISIL Liquid Fund AI Index": "CRISIL_LIQUID",
    "NIFTY 50 TRI": "NIFTY50",
    "NIFTY Midcap 50 TRI": "NIFTY_MIDCAP150",
    "NIFTY Large Midcap 250 TRI": "NIFTY100",
}


def pct(series: pd.Series | float) -> pd.Series | float:
    """Convert decimal returns to percentages."""
    return series * 100


def safe_ratio(numerator: float, denominator: float) -> float:
    """Divide safely for metrics."""
    if pd.isna(denominator) or denominator == 0:
        return np.nan
    return numerator / denominator


def load_processed() -> dict[str, pd.DataFrame]:
    """Load processed or raw-cleanable inputs."""
    required = [
        "dim_fund",
        "fact_nav",
        "fact_performance",
        "fact_benchmark",
        "fact_transactions",
        "fact_holdings",
        "fact_sip",
        "fact_category_inflows",
        "fact_aum",
        "fact_folios",
    ]
    frames = {}
    missing = []
    for name in required:
        path = PROCESSED_DIR / f"{name}.csv"
        if path.exists():
            frames[name] = pd.read_csv(path)
        else:
            missing.append(name)
    if missing:
        raise FileNotFoundError(f"Run scripts/etl_pipeline.py first. Missing: {missing}")

    for frame_name, date_cols in {
        "fact_nav": ["nav_date"],
        "fact_benchmark": ["bench_date"],
        "fact_transactions": ["transaction_date"],
        "fact_sip": ["month"],
        "fact_category_inflows": ["month"],
        "fact_aum": ["aum_date"],
        "fact_folios": ["month"],
        "fact_holdings": ["portfolio_date"],
    }.items():
        for col in date_cols:
            frames[frame_name][col] = pd.to_datetime(frames[frame_name][col], errors="coerce")
    return frames


def compute_scorecard(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute scheme-level return and risk metrics."""
    funds = frames["dim_fund"]
    nav = frames["fact_nav"]
    perf = frames["fact_performance"][["amfi_code", "aum_crore"]].drop_duplicates("amfi_code")
    bench = frames["fact_benchmark"].copy()
    bench_returns = bench.pivot(index="bench_date", columns="index_name", values="daily_return")

    rows = []
    for amfi_code, group in nav[nav["is_observed_nav"] == 1].groupby("amfi_code"):
        group = group.sort_values("nav_date").dropna(subset=["nav"])
        returns = group["nav"].pct_change().dropna()
        if len(group) < 2 or returns.empty:
            continue

        fund_row = funds.loc[funds["amfi_code"] == amfi_code].iloc[0]
        n_trading_days = len(returns)
        total_return = group["nav"].iloc[-1] / group["nav"].iloc[0] - 1
        cagr = (1 + total_return) ** (TRADING_DAYS / n_trading_days) - 1
        ann_return = (1 + returns).prod() ** (TRADING_DAYS / n_trading_days) - 1
        ann_vol = returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
        excess_daily = returns - RF_ANNUAL / TRADING_DAYS
        sharpe = safe_ratio(excess_daily.mean(), returns.std(ddof=1)) * np.sqrt(TRADING_DAYS)
        downside = returns[returns < 0]
        sortino = safe_ratio(excess_daily.mean(), downside.std(ddof=1)) * np.sqrt(TRADING_DAYS)
        running_max = group["nav"].cummax()
        max_dd = (group["nav"] / running_max - 1).min()
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()

        bench_name = BENCHMARK_MAP.get(str(fund_row["benchmark"]), "NIFTY100")
        fund_returns = group[["nav_date"]].copy()
        fund_returns["fund_return"] = group["nav"].pct_change()
        joined = fund_returns.join(bench_returns[[bench_name]], on="nav_date").dropna()
        beta = alpha = tracking_error = information_ratio = np.nan
        if len(joined) > 30 and joined[bench_name].var() != 0:
            cov = np.cov(joined["fund_return"], joined[bench_name])[0, 1]
            beta = cov / joined[bench_name].var()
            alpha = (joined["fund_return"].mean() - beta * joined[bench_name].mean()) * TRADING_DAYS
            active = joined["fund_return"] - joined[bench_name]
            tracking_error = active.std(ddof=1) * np.sqrt(TRADING_DAYS)
            information_ratio = safe_ratio(active.mean() * TRADING_DAYS, tracking_error)

        rows.append(
            {
                "amfi_code": amfi_code,
                "scheme_name": fund_row["scheme_name"],
                "fund_house": fund_row["fund_house"],
                "category": fund_row["category"],
                "sub_category": fund_row["sub_category"],
                "plan": fund_row["plan"],
                "risk_category": fund_row["risk_category"],
                "benchmark": fund_row["benchmark"],
                "benchmark_index": bench_name,
                "cagr_pct": pct(cagr),
                "annualised_return_pct": pct(ann_return),
                "annualised_volatility_pct": pct(ann_vol),
                "sharpe_ratio": sharpe,
                "sortino_ratio": sortino,
                "beta": beta,
                "alpha_pct": pct(alpha),
                "max_drawdown_pct": pct(max_dd),
                "var_95_pct": pct(var_95),
                "cvar_95_pct": pct(cvar_95),
                "tracking_error_pct": pct(tracking_error),
                "information_ratio": information_ratio,
                "expense_ratio_pct": fund_row["expense_ratio_pct"],
                "n_trading_days": n_trading_days,
            }
        )

    score = pd.DataFrame(rows).merge(perf, on="amfi_code", how="left")
    higher = score["annualised_return_pct"].rank(pct=True) * 30
    sharpe_rank = score["sharpe_ratio"].rank(pct=True) * 25
    alpha_rank = score["alpha_pct"].rank(pct=True) * 20
    expense_rank = score["expense_ratio_pct"].rank(pct=True, ascending=False) * 15
    drawdown_rank = score["max_drawdown_pct"].rank(pct=True) * 10
    score["composite_score"] = (higher + sharpe_rank + alpha_rank + expense_rank + drawdown_rank).round(2)
    return score.sort_values("composite_score", ascending=False)


def compute_rolling_sharpe(frames: dict[str, pd.DataFrame], selected_codes: list[int]) -> pd.DataFrame:
    """Compute rolling 90-day Sharpe ratio for selected schemes."""
    nav = frames["fact_nav"]
    out = []
    for code in selected_codes:
        group = nav[(nav["amfi_code"] == code) & (nav["is_observed_nav"] == 1)].sort_values("nav_date")
        returns = group["nav"].pct_change()
        rolling = ((returns.rolling(90).mean() - RF_ANNUAL / TRADING_DAYS) / returns.rolling(90).std()) * np.sqrt(TRADING_DAYS)
        out.append(pd.DataFrame({"amfi_code": code, "nav_date": group["nav_date"], "rolling_sharpe_90d": rolling}))
    return pd.concat(out, ignore_index=True).dropna()


def compute_cohort_analysis(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Group investors by first transaction year."""
    tx = frames["fact_transactions"].copy()
    tx["first_year"] = tx.groupby("investor_id")["transaction_date"].transform("min").dt.year
    sip = tx[tx["transaction_type"].str.upper() == "SIP"]
    pref = tx.merge(frames["dim_fund"][["amfi_code", "sub_category"]], on="amfi_code", how="left")
    fav = pref.groupby(["first_year", "sub_category"])["amount_inr"].sum().reset_index()
    fav = fav.sort_values(["first_year", "amount_inr"], ascending=[True, False]).drop_duplicates("first_year")
    base = tx.groupby("first_year").agg(
        investors=("investor_id", "nunique"),
        total_invested_inr=("amount_inr", "sum"),
        transaction_count=("investor_id", "count"),
    )
    sip_avg = sip.groupby("first_year")["amount_inr"].mean().rename("average_sip_amount_inr")
    return base.join(sip_avg).reset_index().merge(fav[["first_year", "sub_category"]], on="first_year", how="left")


def compute_sip_continuity(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Flag investors with SIP gaps greater than 35 days."""
    sip = frames["fact_transactions"]
    sip = sip[sip["transaction_type"].str.upper() == "SIP"].sort_values(["investor_id", "transaction_date"]).copy()
    sip["gap_days"] = sip.groupby("investor_id")["transaction_date"].diff().dt.days
    summary = sip.groupby("investor_id").agg(
        sip_transactions=("transaction_date", "count"),
        average_gap_days=("gap_days", "mean"),
        total_sip_amount_inr=("amount_inr", "sum"),
        last_sip_date=("transaction_date", "max"),
    )
    summary = summary[summary["sip_transactions"] >= 6].reset_index()
    summary["continuity_status"] = np.where(summary["average_gap_days"] > 35, "At risk", "Regular")
    return summary


def compute_sector_hhi(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute Herfindahl-Hirschman Index by fund from sector weights."""
    holdings = frames["fact_holdings"].copy()
    hhi = holdings.groupby(["amfi_code", "sector"])["weight_pct"].sum().reset_index()
    hhi["weight_decimal"] = hhi["weight_pct"] / 100
    hhi = hhi.groupby("amfi_code")["weight_decimal"].apply(lambda s: float((s**2).sum())).reset_index(name="sector_hhi")
    return hhi.merge(frames["dim_fund"][["amfi_code", "scheme_name", "fund_house"]], on="amfi_code", how="left")


def monte_carlo_projection(frames: dict[str, pd.DataFrame], selected_codes: list[int], years: int = 5, sims: int = 500) -> pd.DataFrame:
    """Project NAV with geometric Brownian motion uncertainty bands."""
    rng = np.random.default_rng(42)
    nav = frames["fact_nav"]
    funds = frames["dim_fund"][["amfi_code", "scheme_name"]]
    rows = []
    horizon = years * TRADING_DAYS
    for code in selected_codes:
        group = nav[(nav["amfi_code"] == code) & (nav["is_observed_nav"] == 1)].sort_values("nav_date")
        returns = group["nav"].pct_change().dropna()
        if returns.empty:
            continue
        start_nav = float(group["nav"].iloc[-1])
        mu, sigma = returns.mean(), returns.std(ddof=1)
        shocks = rng.normal(mu, sigma, size=(sims, horizon))
        paths = start_nav * np.cumprod(1 + shocks, axis=1)
        percentiles = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
        scheme_name = funds.loc[funds["amfi_code"] == code, "scheme_name"].iloc[0]
        for day in range(0, horizon, 21):
            rows.append(
                {
                    "amfi_code": code,
                    "scheme_name": scheme_name,
                    "projection_day": day + 1,
                    "p05_nav": percentiles[0, day],
                    "p25_nav": percentiles[1, day],
                    "p50_nav": percentiles[2, day],
                    "p75_nav": percentiles[3, day],
                    "p95_nav": percentiles[4, day],
                }
            )
    return pd.DataFrame(rows)


def efficient_frontier(frames: dict[str, pd.DataFrame], selected_codes: list[int], portfolios: int = 2500) -> pd.DataFrame:
    """Random-weight Markowitz efficient frontier for selected funds."""
    rng = np.random.default_rng(7)
    nav = frames["fact_nav"]
    returns = []
    names = []
    for code in selected_codes:
        group = nav[(nav["amfi_code"] == code) & (nav["is_observed_nav"] == 1)].sort_values("nav_date")
        series = group.set_index("nav_date")["nav"].pct_change().rename(str(code))
        returns.append(series)
        names.append(str(code))
    matrix = pd.concat(returns, axis=1).dropna()
    mean = matrix.mean() * TRADING_DAYS
    cov = matrix.cov() * TRADING_DAYS
    rows = []
    for _ in range(portfolios):
        weights = rng.dirichlet(np.ones(len(selected_codes)))
        ret = float(np.dot(weights, mean))
        vol = float(np.sqrt(weights @ cov.to_numpy() @ weights))
        rows.append(
            {
                "portfolio_return_pct": pct(ret),
                "portfolio_volatility_pct": pct(vol),
                "portfolio_sharpe": safe_ratio(ret - RF_ANNUAL, vol),
                **{f"weight_{name}": weights[i] for i, name in enumerate(names)},
            }
        )
    return pd.DataFrame(rows)


def write_database_scorecard(scorecard: pd.DataFrame) -> None:
    """Load scorecard into SQLite if the ETL database exists."""
    if not DB_PATH.exists():
        return
    db_cols = [
        "amfi_code",
        "scheme_name",
        "fund_house",
        "category",
        "sub_category",
        "plan",
        "risk_category",
        "cagr_pct",
        "annualised_return_pct",
        "annualised_volatility_pct",
        "sharpe_ratio",
        "sortino_ratio",
        "beta",
        "alpha_pct",
        "max_drawdown_pct",
        "var_95_pct",
        "cvar_95_pct",
        "tracking_error_pct",
        "information_ratio",
        "expense_ratio_pct",
        "aum_crore",
        "composite_score",
    ]
    with sqlite3.connect(DB_PATH) as conn:
        scorecard[db_cols].to_sql("analytics_scorecard", conn, if_exists="replace", index=False)


def main() -> None:
    """Run all analytics and save CSV deliverables."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frames = load_processed()
    selected = [119551, 120503, 118632, 119092, 120841]

    scorecard = compute_scorecard(frames)
    scorecard.to_csv(PROCESSED_DIR / "fund_scorecard.csv", index=False)
    scorecard[["amfi_code", "scheme_name", "cagr_pct", "annualised_return_pct"]].to_csv(PROCESSED_DIR / "cagr_report.csv", index=False)
    scorecard[["amfi_code", "scheme_name", "sharpe_ratio"]].to_csv(PROCESSED_DIR / "sharpe_values.csv", index=False)
    scorecard[["amfi_code", "scheme_name", "sortino_ratio"]].to_csv(PROCESSED_DIR / "sortino_values.csv", index=False)
    scorecard[["amfi_code", "scheme_name", "alpha_pct", "beta"]].to_csv(PROCESSED_DIR / "alpha_beta.csv", index=False)
    scorecard[["amfi_code", "scheme_name", "max_drawdown_pct"]].to_csv(PROCESSED_DIR / "max_drawdown.csv", index=False)
    scorecard[["amfi_code", "scheme_name", "var_95_pct", "cvar_95_pct"]].to_csv(PROCESSED_DIR / "var_cvar_report.csv", index=False)

    compute_rolling_sharpe(frames, selected).to_csv(PROCESSED_DIR / "rolling_sharpe_90d.csv", index=False)
    compute_cohort_analysis(frames).to_csv(PROCESSED_DIR / "cohort_analysis.csv", index=False)
    compute_sip_continuity(frames).to_csv(PROCESSED_DIR / "sip_continuity.csv", index=False)
    compute_sector_hhi(frames).to_csv(PROCESSED_DIR / "sector_hhi.csv", index=False)
    monte_carlo_projection(frames, selected).to_csv(PROCESSED_DIR / "monte_carlo_nav_projection.csv", index=False)
    efficient_frontier(frames, selected).to_csv(PROCESSED_DIR / "efficient_frontier.csv", index=False)
    write_database_scorecard(scorecard)

    print("Analytics complete.")
    print(scorecard.head(10)[["scheme_name", "composite_score", "sharpe_ratio", "alpha_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
