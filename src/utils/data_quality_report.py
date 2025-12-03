"""Data Quality Report Generator.

Generates JSON reports tracking data quality metrics for ML training datasets.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)


def generate_quality_report(
    df: pd.DataFrame,
    output_path: str = "data/processed/data_quality_report.json"
) -> Dict[str, Any]:
    """
    Generate comprehensive data quality report.

    Tracks: missing rates, value ranges, class distribution.

    Args:
        df: DataFrame with WQI parameters and labels
        output_path: Path to save JSON report (relative to CWD, which should be repo root)

    Returns:
        Dict containing quality metrics (also saved to output_path)

    Note:
        output_path assumes CWD is repo root, consistent with load_kaggle_data().
        For CWD-independent paths, pass absolute path or compute from __file__.
    """
    wqi_params = ['ph', 'dissolved_oxygen', 'temperature', 'turbidity', 'nitrate', 'conductance']

    report = {
        "generated_at": datetime.now().isoformat(),
        "total_samples": len(df),
        "missing_rates": {},
        "value_ranges": {},
        "class_distribution": {}
    }

    # Missing rates
    for param in wqi_params:
        if param in df.columns:
            missing_rate = df[param].isna().mean()
            report["missing_rates"][param] = round(missing_rate * 100, 2)

    # Value ranges (only for columns with non-null values)
    for param in wqi_params:
        if param in df.columns and df[param].notna().any():
            report["value_ranges"][param] = {
                "min": round(float(df[param].min()), 2),
                "max": round(float(df[param].max()), 2),
                "mean": round(float(df[param].mean()), 2),
                "std": round(float(df[param].std()), 2)
            }

    # Class distribution
    if 'is_safe' in df.columns:
        safe_count = int(df['is_safe'].sum())
        unsafe_count = len(df) - safe_count
        report["class_distribution"] = {
            "safe_count": safe_count,
            "unsafe_count": unsafe_count,
            "safe_percent": round(safe_count / len(df) * 100, 2),
            "unsafe_percent": round(unsafe_count / len(df) * 100, 2)
        }

    # Save report
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"Data quality report saved to {output_path}")
    return report
