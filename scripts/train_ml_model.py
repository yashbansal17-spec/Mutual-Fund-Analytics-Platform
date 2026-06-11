"""Optional bonus ML model: predict 3-year fund return from fund metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Train a simple regression model and save predictions/metrics."""
    score = pd.read_csv(PROCESSED_DIR / "fund_scorecard.csv")
    features = ["expense_ratio_pct", "aum_crore", "sharpe_ratio", "sortino_ratio", "alpha_pct", "beta", "max_drawdown_pct"]
    target = "cagr_3yr_pct"
    model_data = score.dropna(subset=features + [target]).copy()

    x = model_data[features]
    y = model_data[target]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    model = RandomForestRegressor(n_estimators=150, random_state=42, min_samples_leaf=2)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    metrics = {
        "model": "RandomForestRegressor",
        "target": target,
        "features": features,
        "test_rows": int(len(y_test)),
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
    }
    prediction_output = x_test.copy()
    prediction_output["actual_cagr_3yr_pct"] = y_test.values
    prediction_output["predicted_cagr_3yr_pct"] = preds
    prediction_output = prediction_output.merge(score[["scheme_name", "fund_house"]], left_index=True, right_index=True, how="left")
    prediction_output.to_csv(MODEL_DIR / "fund_return_model_predictions.csv", index=False)
    (MODEL_DIR / "fund_return_model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("ML model outputs saved to models/.")


if __name__ == "__main__":
    main()
