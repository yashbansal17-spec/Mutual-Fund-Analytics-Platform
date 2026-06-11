"""Master execution script for the Bluestock Mutual Fund Analytics project.

The runner executes the project in dependency order:
1. ETL and SQLite load
2. Core metrics
3. EDA exports
4. Performance analytics
5. Advanced analytics
6. ML model
7. Notebook/report assets
8. Validation checks

Live NAV fetching is optional because it requires internet access.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


PIPELINE_STEPS = [
    ("ETL pipeline and SQLite load", "etl_pipeline.py"),
    ("Core metrics and bonus analytics", "compute_metrics.py"),
    ("EDA notebook and PNG charts", "generate_eda.py"),
    ("Performance analytics", "generate_performance_analytics.py"),
    ("Advanced analytics", "generate_advanced_analytics.py"),
    ("ML return model", "train_ml_model.py"),
    ("Final notebook report", "generate_final_report.py"),
    ("Power BI-style export assets", "generate_day5_dashboard_assets.py"),
    ("Final PDF report", "generate_pdf_report.py"),
    ("Analysis validation", "validate_analysis_outputs.py"),
    ("Chart validation", "validate_chart_outputs.py"),
]


def run_script(script_name: str) -> None:
    """Run a project script with the active Python interpreter."""
    script_path = PROJECT_ROOT / "scripts" / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")
    print(f"[Bluestock Pipeline] Running {script_name}")
    subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT, check=True)


def main() -> None:
    """Execute the full capstone pipeline."""
    parser = argparse.ArgumentParser(description="Run the Bluestock MF capstone pipeline.")
    parser.add_argument("--include-live-nav", action="store_true", help="Fetch live NAV data from mfapi.in.")
    args = parser.parse_args()

    if args.include_live_nav:
        run_script("live_nav_fetch.py")

    for _, script_name in PIPELINE_STEPS:
        run_script(script_name)

    print("[Bluestock Pipeline] Complete")
    print("Open dashboard with: streamlit run dashboard/streamlit_app.py")


if __name__ == "__main__":
    main()
