"""Flask API serving Bluestock mutual fund analytics insights as JSON."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

app = Flask(__name__)


def read_csv(name: str) -> pd.DataFrame:
    """Load a processed CSV from the project."""
    path = PROCESSED_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing processed file: {path}")
    return pd.read_csv(path)


def records(df: pd.DataFrame, limit: int | None = None) -> list[dict]:
    """Convert a DataFrame to clean JSON records."""
    if limit is not None:
        df = df.head(limit)
    return df.where(pd.notna(df), None).to_dict(orient="records")


@app.get("/api/health")
def health() -> tuple:
    """Return API and data readiness status."""
    required = ["fund_scorecard.csv", "fact_nav.csv", "fact_transactions.csv", "fact_sip.csv"]
    missing = [name for name in required if not (PROCESSED_DIR / name).exists()]
    return jsonify({"status": "ok" if not missing else "missing_data", "missing_files": missing}), 200


@app.get("/api/top-funds")
def top_funds() -> tuple:
    """Return top funds by composite score."""
    limit = min(int(request.args.get("limit", 5)), 25)
    score = read_csv("fund_scorecard.csv").sort_values("composite_score", ascending=False)
    cols = ["amfi_code", "scheme_name", "fund_house", "cagr_3yr_pct", "sharpe_ratio", "alpha_pct", "max_drawdown_pct", "composite_score"]
    return jsonify({"limit": limit, "data": records(score[cols], limit)}), 200


@app.get("/api/fund/<int:amfi_code>")
def fund_detail(amfi_code: int) -> tuple:
    """Return one fund's scorecard and latest NAV information."""
    score = read_csv("fund_scorecard.csv")
    nav = read_csv("fact_nav.csv")
    fund = score[score["amfi_code"].eq(amfi_code)]
    if fund.empty:
        return jsonify({"error": "fund_not_found", "amfi_code": amfi_code}), 404
    latest_nav = nav[nav["amfi_code"].eq(amfi_code)].sort_values("nav_date").tail(1)
    return jsonify({"scorecard": records(fund, 1)[0], "latest_nav": records(latest_nav, 1)[0]}), 200


@app.get("/api/industry-summary")
def industry_summary() -> tuple:
    """Return high-level industry metrics for app/report use."""
    sip = read_csv("fact_sip.csv").sort_values("month")
    folios = read_csv("fact_folios.csv").sort_values("month")
    aum = read_csv("fact_aum.csv").sort_values("aum_date")
    latest_sip = sip.tail(1).iloc[0]
    latest_folio = folios.tail(1).iloc[0]
    latest_aum = aum.groupby("fund_house").tail(1)["aum_crore"].sum()
    return jsonify(
        {
            "latest_sip_inflow_crore": float(latest_sip["sip_inflow_crore"]),
            "latest_sip_month": str(latest_sip["month"]),
            "latest_total_folios_crore": float(latest_folio["total_folios_crore"]),
            "tracked_latest_aum_crore": float(latest_aum),
        }
    ), 200


@app.get("/api/state-transactions")
def state_transactions() -> tuple:
    """Return transaction amount by state."""
    limit = min(int(request.args.get("limit", 10)), 30)
    tx = read_csv("fact_transactions.csv")
    state = tx.groupby("state", as_index=False).agg(txn_count=("investor_id", "count"), total_amount_inr=("amount_inr", "sum"))
    state = state.sort_values("total_amount_inr", ascending=False)
    return jsonify({"limit": limit, "data": records(state, limit)}), 200


if __name__ == "__main__":
    if os.getenv("BLUESTOCK_ENABLE_API") != "1":
        print("Flask API is optional and disabled for the dashboard demo.")
        print(r"Run the Streamlit app instead: streamlit run dashboard\streamlit_app.py")
        print(r"To enable API testing only: $env:BLUESTOCK_ENABLE_API='1'; python api\app.py")
    else:
        app.run(debug=False, use_reloader=False, host="127.0.0.1", port=5000)
