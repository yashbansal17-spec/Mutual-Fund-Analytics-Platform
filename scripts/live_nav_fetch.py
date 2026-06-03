"""Fetch live/historical NAV data from mfapi.in and save raw CSV output."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

SCHEMES = {
    125497: "HDFC Top 100 Direct",
    119551: "SBI Bluechip",
    120503: "ICICI Bluechip",
    118632: "Nippon Large Cap",
    119092: "Axis Bluechip",
    120841: "Kotak Bluechip",
}


def fetch_scheme(amfi_code: int, timeout: int = 20) -> pd.DataFrame:
    """Fetch one scheme from mfapi.in."""
    url = f"https://api.mfapi.in/mf/{amfi_code}"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("data", [])
    if not rows:
        raise ValueError(f"No NAV rows returned for {amfi_code}")
    df = pd.DataFrame(rows)
    df["amfi_code"] = amfi_code
    df["scheme_label"] = SCHEMES.get(amfi_code, payload.get("meta", {}).get("scheme_name", "Unknown"))
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    return df[["amfi_code", "scheme_label", "date", "nav"]].sort_values("date")


def main() -> int:
    """Fetch all required schemes and save a single CSV."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    errors = []
    for code in SCHEMES:
        try:
            frames.append(fetch_scheme(code))
            print(f"Fetched {code}: {SCHEMES[code]}")
        except Exception as exc:  # noqa: BLE001 - keep fetcher resilient for live API outages
            errors.append(f"{code} {SCHEMES[code]}: {exc}")
            print(f"Failed {code}: {exc}", file=sys.stderr)

    if not frames:
        raise RuntimeError("mfapi.in fetch failed for every scheme.")

    out = pd.concat(frames, ignore_index=True)
    output_path = RAW_DIR / "live_nav_key_schemes.csv"
    out.to_csv(output_path, index=False)
    print(f"Saved {len(out):,} rows to {output_path}")

    if errors:
        (RAW_DIR / "live_nav_fetch_errors.txt").write_text("\n".join(errors), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
