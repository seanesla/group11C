#!/usr/bin/env python3
"""Harvest US water quality samples to augment the ML training dataset."""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.us_training_data import merge_us_samples

DEFAULT_ZIP_CODES = [
    "90001",  # Los Angeles, CA
    "94102",  # San Francisco, CA
    "95814",  # Sacramento, CA
    "85001",  # Phoenix, AZ
    "77001",  # Houston, TX
    "33101",  # Miami, FL
    "30301",  # Atlanta, GA
    "10001",  # New York, NY
    "19104",  # Philadelphia, PA
    "21201",  # Baltimore, MD
    "48201",  # Detroit, MI
    "60601",  # Chicago, IL
    "80201",  # Denver, CO
    "98101",  # Seattle, WA
    "99501",  # Anchorage, AK
    "96801",  # Honolulu, HI
    "35203",  # Birmingham, AL
    "64101",  # Kansas City, MO
    "02108",  # Boston, MA
    "20001",  # Washington, DC
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build US training sample CSV")
    parser.add_argument(
        "--output",
        default="data/processed/us_training_samples.csv",
        help="Where to save the aggregated CSV",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=45.0,
        help="Base search radius in miles",
    )
    parser.add_argument(
        "--lookback-years",
        type=int,
        default=5,
        help="Historical window to capture",
    )
    parser.add_argument(
        "--zip",
        dest="zip_codes",
        nargs="*",
        default=DEFAULT_ZIP_CODES,
        help="Optional custom list of ZIP codes",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(
        f"Collecting US samples for {len(args.zip_codes)} ZIP codes "
        f"(radius={args.radius} mi, lookback={args.lookback_years} years)"
    )

    df = merge_us_samples(
        args.zip_codes,
        radius_miles=args.radius,
        lookback_years=args.lookback_years,
    )

    if df.empty:
        print("No samples collected; check network/API availability.")
        return 1

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output} at {datetime.utcnow().isoformat()}Z")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
