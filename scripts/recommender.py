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
    allowed = RISK_MAP.get(risk_appetite.strip().lower(), RISK_MAP["moderate"])
    picks = scorecard[scorecard["risk_category"].isin(allowed)].copy()
    if picks.empty:
        picks = scorecard.copy()
    return picks.sort_values(["sharpe_ratio", "composite_score"], ascending=False).head(top_n)


def main() -> None:
    """Print sample recommendation tables."""
    for risk in ["Low", "Moderate", "High"]:
        print(f"\nTop recommendations for {risk} risk appetite")
        cols = ["scheme_name", "fund_house", "risk_category", "sharpe_ratio", "composite_score"]
        print(recommend_funds(risk)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
