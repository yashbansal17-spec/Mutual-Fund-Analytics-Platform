"""Simple rule-based mutual fund recommender."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RISK_MAP = {
    "low": ["Low"],
    "moderate": ["Moderate", "Moderately High"],
    "high": ["High", "Very High"],
}


def recommend_funds(risk_appetite: str = "Moderate", top_n: int = 3) -> pd.DataFrame:
    """Return top funds by Sharpe ratio within the selected risk appetite."""
    scorecard_path = PROCESSED_DIR / "fund_scorecard.csv"
    if not scorecard_path.exists():
        raise FileNotFoundError("Run scripts/compute_metrics.py before recommender.py")
    scorecard = pd.read_csv(scorecard_path)
    performance_path = PROCESSED_DIR / "fact_performance.csv"
    if performance_path.exists() and "risk_grade" not in scorecard.columns:
        perf = pd.read_csv(performance_path)[["amfi_code", "risk_grade"]].drop_duplicates()
        scorecard = scorecard.merge(perf, on="amfi_code", how="left")
    allowed = RISK_MAP.get(risk_appetite.strip().lower(), RISK_MAP["moderate"])
    risk_col = "risk_grade" if "risk_grade" in scorecard.columns and scorecard["risk_grade"].notna().any() else "risk_category"
    picks = scorecard[scorecard[risk_col].isin(allowed)].copy()
    if picks.empty:
        picks = scorecard.copy()
    cols = [
        "scheme_name",
        "fund_house",
        "category",
        "sub_category",
        risk_col,
        "sharpe_ratio",
        "sortino_ratio",
        "alpha_pct",
        "max_drawdown_pct",
        "composite_score",
    ]
    return picks.sort_values(["sharpe_ratio", "composite_score"], ascending=False).head(top_n)[cols]


def main() -> None:
    """Print sample recommendation tables."""
    for risk in ["Low", "Moderate", "High"]:
        print(f"\nTop recommendations for {risk} risk appetite")
        print(recommend_funds(risk).to_string(index=False))


if __name__ == "__main__":
    main()
