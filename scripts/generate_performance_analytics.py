"""Generate Day 4 performance analytics outputs and notebook."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
CHART_DIR = PROJECT_ROOT / "reports" / "performance_charts"
NOTEBOOK_PATH = NOTEBOOK_DIR / "Performance_Analytics.ipynb"
RF_ANNUAL = 0.065
TRADING_DAYS = 252
COLORS = ["#2563eb", "#0f766e", "#dc2626", "#7c3aed", "#f59e0b", "#0891b2", "#be123c"]

CHART_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=0.95)


def read_csv(name: str, dates: list[str] | None = None) -> pd.DataFrame:
    """Read a processed CSV."""
    return pd.read_csv(PROCESSED_DIR / name, parse_dates=dates or [])


def save_fig(fig: plt.Figure, file_name: str) -> Path:
    """Save a Matplotlib chart."""
    path = CHART_DIR / file_name
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def trailing_cagr(group: pd.DataFrame, years: int) -> tuple[float, float]:
    """Compute CAGR for a requested trailing period; use available history if shorter."""
    end_date = group["nav_date"].max()
    target_start = end_date - pd.DateOffset(years=years)
    window = group[group["nav_date"] >= target_start].sort_values("nav_date")
    if len(window) < 2:
        return np.nan, np.nan
    start_nav = window["nav"].iloc[0]
    end_nav = window["nav"].iloc[-1]
    trading_days = len(window["daily_return"].dropna())
    years_used = trading_days / TRADING_DAYS
    if years_used <= 0:
        return np.nan, np.nan
    return ((end_nav / start_nav) ** (1 / years_used) - 1) * 100, years_used


def max_drawdown_with_dates(group: pd.DataFrame) -> tuple[float, pd.Timestamp, pd.Timestamp]:
    """Compute max drawdown and its peak-to-trough dates."""
    ordered = group.sort_values("nav_date").copy()
    ordered["running_max"] = ordered["nav"].cummax()
    ordered["drawdown"] = ordered["nav"] / ordered["running_max"] - 1
    trough_idx = ordered["drawdown"].idxmin()
    trough_date = ordered.loc[trough_idx, "nav_date"]
    peak_slice = ordered.loc[:trough_idx]
    peak_idx = peak_slice["nav"].idxmax()
    peak_date = ordered.loc[peak_idx, "nav_date"]
    return ordered.loc[trough_idx, "drawdown"] * 100, peak_date, trough_date


def main() -> None:
    """Create Day 4 CSVs, chart, and notebook."""
    funds = read_csv("dim_fund.csv", ["launch_date"])
    nav = read_csv("fact_nav.csv", ["nav_date"])
    benchmark = read_csv("fact_benchmark.csv", ["bench_date"])
    performance = read_csv("fact_performance.csv")

    nav = nav[nav["is_observed_nav"].eq(1)].sort_values(["amfi_code", "nav_date"]).copy()
    nav["daily_return"] = nav.groupby("amfi_code")["nav"].pct_change()
    returns = nav[["amfi_code", "nav_date", "nav", "daily_return"]].merge(
        funds[["amfi_code", "scheme_name", "fund_house", "category", "sub_category", "plan", "risk_category", "expense_ratio_pct"]],
        on="amfi_code",
        how="left",
    )
    returns.to_csv(PROCESSED_DIR / "daily_returns.csv", index=False)

    distribution = returns.groupby("amfi_code")["daily_return"].agg(["count", "mean", "std", "min", "max", "skew"]).reset_index()
    distribution = distribution.merge(funds[["amfi_code", "scheme_name"]], on="amfi_code", how="left")
    distribution["reasonable_distribution_flag"] = distribution["max"].lt(0.20) & distribution["min"].gt(-0.20)
    distribution.to_csv(PROCESSED_DIR / "return_distribution_summary.csv", index=False)

    nifty100 = benchmark[benchmark["index_name"].eq("NIFTY100")][["bench_date", "daily_return"]].rename(
        columns={"bench_date": "nav_date", "daily_return": "nifty100_return"}
    )
    nifty50 = benchmark[benchmark["index_name"].eq("NIFTY50")][["bench_date", "close_value"]].rename(
        columns={"bench_date": "nav_date", "close_value": "nifty50"}
    )
    nifty100_close = benchmark[benchmark["index_name"].eq("NIFTY100")][["bench_date", "close_value"]].rename(
        columns={"bench_date": "nav_date", "close_value": "nifty100"}
    )

    rows = []
    for amfi_code, group in returns.groupby("amfi_code"):
        group = group.sort_values("nav_date").dropna(subset=["nav"])
        ret = group["daily_return"].dropna()
        if ret.empty:
            continue
        fund = group.iloc[0]
        cagr_1, years_1 = trailing_cagr(group, 1)
        cagr_3, years_3 = trailing_cagr(group, 3)
        cagr_5, years_5 = trailing_cagr(group, 5)
        rp_daily = ret.mean()
        std_daily = ret.std(ddof=1)
        annualised_return = ((1 + ret).prod() ** (TRADING_DAYS / len(ret)) - 1) * 100
        annualised_volatility = std_daily * np.sqrt(TRADING_DAYS) * 100
        var_95 = np.percentile(ret, 5) * 100
        cvar_95 = ret[ret <= np.percentile(ret, 5)].mean() * 100
        downside_std = ret[ret < 0].std(ddof=1)
        sharpe = ((rp_daily - RF_ANNUAL / TRADING_DAYS) / std_daily) * np.sqrt(TRADING_DAYS) if std_daily else np.nan
        sortino = ((rp_daily - RF_ANNUAL / TRADING_DAYS) / downside_std) * np.sqrt(TRADING_DAYS) if downside_std else np.nan
        max_dd, peak_date, trough_date = max_drawdown_with_dates(group)

        joined = group[["nav_date", "daily_return"]].merge(nifty100, on="nav_date", how="inner").dropna()
        alpha = beta = r_value = p_value = np.nan
        if len(joined) > 30:
            regression = linregress(joined["nifty100_return"], joined["daily_return"])
            alpha = regression.intercept * TRADING_DAYS * 100
            beta = regression.slope
            r_value = regression.rvalue
            p_value = regression.pvalue

        rows.append(
            {
                "amfi_code": amfi_code,
                "scheme_name": fund["scheme_name"],
                "fund_house": fund["fund_house"],
                "category": fund["category"],
                "sub_category": fund["sub_category"],
                "plan": fund["plan"],
                "risk_category": fund["risk_category"],
                "cagr_1yr_pct": cagr_1,
                "cagr_3yr_pct": cagr_3,
                "cagr_5yr_pct": cagr_5,
                "annualised_return_pct": annualised_return,
                "annualised_volatility_pct": annualised_volatility,
                "years_used_1yr": years_1,
                "years_used_3yr": years_3,
                "years_used_5yr": years_5,
                "sharpe_ratio": sharpe,
                "sortino_ratio": sortino,
                "alpha_pct": alpha,
                "beta": beta,
                "ols_r_value": r_value,
                "ols_p_value": p_value,
                "max_drawdown_pct": max_dd,
                "var_95_pct": var_95,
                "cvar_95_pct": cvar_95,
                "drawdown_start_date": peak_date,
                "drawdown_end_date": trough_date,
                "expense_ratio_pct": fund["expense_ratio_pct"],
            }
        )

    metrics = pd.DataFrame(rows).merge(performance[["amfi_code", "aum_crore"]], on="amfi_code", how="left")
    metrics["return_rank_score"] = metrics["cagr_3yr_pct"].rank(pct=True) * 30
    metrics["sharpe_rank_score"] = metrics["sharpe_ratio"].rank(pct=True) * 25
    metrics["alpha_rank_score"] = metrics["alpha_pct"].rank(pct=True) * 20
    metrics["expense_rank_score"] = metrics["expense_ratio_pct"].rank(pct=True, ascending=False) * 15
    metrics["drawdown_rank_score"] = metrics["max_drawdown_pct"].rank(pct=True) * 10
    metrics["composite_score"] = (
        metrics["return_rank_score"]
        + metrics["sharpe_rank_score"]
        + metrics["alpha_rank_score"]
        + metrics["expense_rank_score"]
        + metrics["drawdown_rank_score"]
    ).round(2)
    metrics = metrics.sort_values("composite_score", ascending=False)

    cagr_cols = ["amfi_code", "scheme_name", "fund_house", "cagr_1yr_pct", "cagr_3yr_pct", "cagr_5yr_pct", "years_used_5yr"]
    metrics[cagr_cols].to_csv(PROCESSED_DIR / "cagr_comparison.csv", index=False)
    metrics[["amfi_code", "scheme_name", "fund_house", "sharpe_ratio"]].sort_values("sharpe_ratio", ascending=False).to_csv(
        PROCESSED_DIR / "sharpe_rankings.csv", index=False
    )
    metrics[["amfi_code", "scheme_name", "fund_house", "sortino_ratio"]].sort_values("sortino_ratio", ascending=False).to_csv(
        PROCESSED_DIR / "sortino_rankings.csv", index=False
    )
    metrics[["amfi_code", "scheme_name", "fund_house", "alpha_pct", "beta", "ols_r_value", "ols_p_value"]].to_csv(
        PROCESSED_DIR / "alpha_beta.csv", index=False
    )
    metrics[["amfi_code", "scheme_name", "fund_house", "max_drawdown_pct", "drawdown_start_date", "drawdown_end_date"]].to_csv(
        PROCESSED_DIR / "max_drawdown.csv", index=False
    )
    score_cols = [
        "amfi_code",
        "scheme_name",
        "fund_house",
        "category",
        "sub_category",
        "plan",
        "risk_category",
        "cagr_1yr_pct",
        "cagr_3yr_pct",
        "cagr_5yr_pct",
        "annualised_return_pct",
        "annualised_volatility_pct",
        "sharpe_ratio",
        "sortino_ratio",
        "alpha_pct",
        "beta",
        "max_drawdown_pct",
        "var_95_pct",
        "cvar_95_pct",
        "drawdown_start_date",
        "drawdown_end_date",
        "expense_ratio_pct",
        "aum_crore",
        "composite_score",
    ]
    metrics[score_cols].to_csv(PROCESSED_DIR / "fund_scorecard.csv", index=False)

    # Benchmark comparison for top 5 scorecard funds over trailing 3 years.
    top5 = metrics.head(5)
    end_date = returns["nav_date"].max()
    start_date = end_date - pd.DateOffset(years=3)
    comparison_rows = []
    normalized = []
    bench_close = nifty50.merge(nifty100_close, on="nav_date", how="inner")
    bench_close = bench_close[bench_close["nav_date"].between(start_date, end_date)].copy()
    for col in ["nifty50", "nifty100"]:
        bench_close[f"{col}_norm"] = bench_close[col] / bench_close[col].iloc[0] * 100
    normalized.append(bench_close[["nav_date", "nifty50_norm"]].rename(columns={"nifty50_norm": "normalized_value"}).assign(series="NIFTY50"))
    normalized.append(bench_close[["nav_date", "nifty100_norm"]].rename(columns={"nifty100_norm": "normalized_value"}).assign(series="NIFTY100"))

    benchmark_returns = benchmark[benchmark["index_name"].isin(["NIFTY50", "NIFTY100"])].pivot(
        index="bench_date", columns="index_name", values="daily_return"
    )
    for _, fund in top5.iterrows():
        series = returns[(returns["amfi_code"].eq(fund["amfi_code"])) & (returns["nav_date"].between(start_date, end_date))].copy()
        series["normalized_value"] = series["nav"] / series["nav"].iloc[0] * 100
        normalized.append(series[["nav_date", "normalized_value"]].assign(series=fund["scheme_name"][:36]))
        joined = series[["nav_date", "daily_return"]].join(benchmark_returns, on="nav_date").dropna()
        for bench_name in ["NIFTY50", "NIFTY100"]:
            tracking_error = (joined["daily_return"] - joined[bench_name]).std(ddof=1) * np.sqrt(TRADING_DAYS) * 100
            comparison_rows.append(
                {
                    "amfi_code": fund["amfi_code"],
                    "scheme_name": fund["scheme_name"],
                    "benchmark": bench_name,
                    "tracking_error_pct": tracking_error,
                }
            )
    tracking = pd.DataFrame(comparison_rows)
    tracking.to_csv(PROCESSED_DIR / "tracking_error.csv", index=False)

    comparison = pd.concat(normalized, ignore_index=True)
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(data=comparison, x="nav_date", y="normalized_value", hue="series", ax=ax, linewidth=2)
    ax.set_title("Top 5 Funds vs NIFTY50 and NIFTY100: 3-Year Normalized Performance")
    ax.set_ylabel("Normalized value (start = 100)")
    ax.set_xlabel("")
    ax.legend(fontsize=8, ncol=2)
    benchmark_chart = save_fig(fig, "benchmark_comparison_top5.png")

    # Small chart for app: return distribution.
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.histplot(returns["daily_return"].dropna(), bins=80, kde=True, ax=ax, color=COLORS[0])
    ax.set_title("Daily Return Distribution Across All Funds")
    ax.set_xlabel("Daily return")
    save_fig(fig, "daily_return_distribution.png")

    findings = [
        f"Daily return validation is reasonable: {distribution['reasonable_distribution_flag'].sum()} of {len(distribution)} funds stay within +/-20% daily return bounds.",
        f"The top composite fund is {metrics.iloc[0]['scheme_name']} with a score of {metrics.iloc[0]['composite_score']:.2f}/100.",
        f"The highest Sharpe ratio is {metrics['sharpe_ratio'].max():.2f}, using a 6.5% annual risk-free-rate proxy.",
        f"The largest drawdown is {metrics['max_drawdown_pct'].min():.2f}%, captured with peak-to-trough dates.",
        f"Alpha and beta were computed for all funds using OLS regression against NIFTY100 daily returns.",
    ]

    notebook_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Performance Analytics - Day 4\n",
                "\n",
                "This notebook documents daily returns, trailing CAGR, Sharpe, Sortino, Alpha, Beta, maximum drawdown, composite fund scorecard, and benchmark tracking error.\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Formula Reference\n",
                "- Daily return = `nav_t / nav_t-1 - 1`\n",
                "- CAGR = `(NAV_end / NAV_start) ** (1 / n_years) - 1`\n",
                "- Sharpe = `(Rp - Rf) / Std(Rp) * sqrt(252)`, with `Rf = 6.5%`\n",
                "- Sortino uses downside standard deviation from negative return days only\n",
                "- Alpha/Beta use `scipy.stats.linregress` against NIFTY100 daily returns; alpha = intercept * 252\n",
                "- Max drawdown = `min(NAV / running_max - 1)`\n",
                "- Tracking error = `std(fund_return - benchmark_return) * sqrt(252)`\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "import pandas as pd\n",
                "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n",
                "PROCESSED = PROJECT_ROOT / 'data' / 'processed'\n",
                "pd.read_csv(PROCESSED / 'fund_scorecard.csv').head(10)\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Benchmark Comparison Chart\n",
                f"![Benchmark Comparison](../reports/performance_charts/{benchmark_chart.name})\n",
                "\n",
                "## Daily Return Distribution\n",
                "![Daily Return Distribution](../reports/performance_charts/daily_return_distribution.png)\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Key Performance Findings\n", "\n"] + [f"{i}. {finding}\n" for i, finding in enumerate(findings, 1)],
        },
    ]
    nb = {
        "cells": notebook_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10+"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(json.dumps(nb, indent=2), encoding="utf-8")

    print(f"Created {PROCESSED_DIR / 'fund_scorecard.csv'}")
    print(f"Created {PROCESSED_DIR / 'alpha_beta.csv'}")
    print(f"Created {benchmark_chart}")
    print(f"Created {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
