"""Run the full Bluestock capstone pipeline in sequence."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_step(script_name: str) -> None:
    """Execute a project script with the current Python interpreter."""
    script_path = PROJECT_ROOT / "scripts" / script_name
    print(f"\nRunning {script_name}")
    subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT, check=True)


def main() -> None:
    """Run ETL, metrics, and recommender sample output."""
    run_step("etl_pipeline.py")
    run_step("compute_metrics.py")
    run_step("recommender.py")
    print("\nPipeline complete. Start dashboard with:")
    print("streamlit run dashboard/streamlit_app.py")


if __name__ == "__main__":
    main()
