#!/usr/bin/env python3
"""Run the pytest suite in manageable chunks.

This helper splits the very large test suite into themed groups so long runs
can be resumed and debugged more easily. It respects the default marker filter
(`-m "not integration"`).

Examples
--------
Run everything, group by group:
    poetry run python scripts/run_tests_chunked.py

Run only the geo-related tests, stop on first failure:
    poetry run python scripts/run_tests_chunked.py --group geo --stop-on-fail

Run quietly with max one failure per group:
    poetry run python scripts/run_tests_chunked.py --maxfail 1 --quiet
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import List


# Curated groups to keep runtimes manageable and logically separated
TEST_GROUPS = {
    "core_fast": [
        "tests/test_calculate_wqi_labels.py",
        "tests/test_wqi_calculator.py",
        "tests/test_wqi_calculations.py",
        "tests/test_wqi_calculation_nitrate_input.py",
        "tests/test_wqi_scientific_validation.py",
        "tests/test_nitrate_unit_system.py",
        "tests/test_nitrate_bug_verification.py",
        "tests/test_us_data_features.py",
        "tests/test_feature_engineering.py",
        "tests/test_prepare_ml_dataset.py",
        "tests/test_load_kaggle_data.py",
        "tests/test_extract_wqi_parameters.py",
    ],
    "models": [
        "tests/test_classifier.py",
        "tests/test_regressor.py",
        "tests/test_ml_robustness.py",
        "tests/test_compare_models_fair.py",
        "tests/test_fairness_demographics.py",
        "tests/test_shap_contributions_hyperthorough_1.py",
        "tests/test_shap_contributions_hyperthorough_3.py",
        "tests/test_shap_contributions_hyperthorough_4.py",
    ],
    "geo": [
        "tests/test_zipcode_mapper.py",
        "tests/test_geographic_coverage.py",
        "tests/test_geographic_coverage_191_locations.py",
        "tests/test_geographic_diagnostic.py",
        "tests/test_geographic_dynamic_sampling.py",
    ],
    "clients": [
        "tests/test_wqp_client.py",
        "tests/test_usgs_client.py",
        "tests/test_data_fallback.py",
        "tests/test_kaggle_nitrate_integration.py",
        "tests/test_wqi_calculation_nitrate_input.py",
    ],
    "ui": [
        "tests/test_streamlit_app.py",
        "tests/test_streamlit_e2e.py",
        "tests/test_e2e_streamlit.py",
        "tests/test_phase4_ui_comprehensive.py",
        "tests/test_phase4_backend_comprehensive.py",
    ],
}


def run_group(name: str, patterns: List[str], extra_pytest_args: List[str]) -> int:
    """Execute pytest for a specific group of test patterns."""

    cmd = [
        "poetry",
        "run",
        "pytest",
        "-m",
        "not integration",
        *extra_pytest_args,
        *patterns,
    ]

    print(f"\n[run_tests_chunked] Running group '{name}' -> {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        choices=sorted(TEST_GROUPS.keys()),
        action="append",
        help="Group(s) to run. Omit to run all groups sequentially.",
    )
    parser.add_argument(
        "--maxfail",
        type=int,
        default=None,
        help="Stop after the given number of failures within each group.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Pass -q to pytest for quieter output.",
    )
    parser.add_argument(
        "--stop-on-fail",
        action="store_true",
        help="Stop running further groups when a group fails.",
    )

    args = parser.parse_args()

    groups_to_run = args.group or list(TEST_GROUPS.keys())

    extra: List[str] = []
    if args.maxfail is not None:
        extra.append(f"--maxfail={args.maxfail}")
    if args.quiet:
        extra.append("-q")

    overall_rc = 0
    for group_name in groups_to_run:
        rc = run_group(group_name, TEST_GROUPS[group_name], extra)
        if rc != 0:
            overall_rc = rc
            if args.stop_on_fail:
                break

    return overall_rc


if __name__ == "__main__":
    sys.exit(main())
