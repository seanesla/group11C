#!/usr/bin/env python3
"""Train a US domain calibrator using harvested US samples.

Pipeline:
- Expect `data/processed/us_training_samples.csv` (built by
  `scripts/build_us_training_samples.py`). Each row has a WQI score computed
  with the same `WQICalculator` used online.
- Load the latest RandomForest regressor trained on the Kaggle core-feature
  dataset.
- Re-run the regressor on US samples to obtain `y_pred`.
- Fit an isotonic `DomainCalibrator` to map `y_pred → wqi_score`.
- Save the calibrator alongside the regressor as
  `data/models/calibrator_us_YYYYMMDD_HHMMSS.joblib` so
  `WQIPredictionRegressor.load(..., load_calibration=True)` will auto-attach it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model_utils import get_latest_model_path  # type: ignore
from src.models.regressor import WQIPredictionRegressor  # type: ignore
from src.models.domain_calibrator import DomainCalibrator  # type: ignore
from src.preprocessing.us_data_features import (  # type: ignore
    prepare_us_features_for_prediction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train US domain calibration from harvested US samples",
    )
    parser.add_argument(
        "--samples-path",
        default="data/processed/us_training_samples.csv",
        help="CSV produced by build_us_training_samples.py",
    )
    parser.add_argument(
        "--min-params",
        type=int,
        default=4,
        help="Minimum number of non-null core parameters required per row",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Fraction of samples to hold out for validation",
    )
    return parser.parse_args()


def load_us_samples(path: Path, min_params: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"US samples file not found at {path}. Run scripts/build_us_training_samples.py first."
        )

    df = pd.read_csv(path)

    required_cols = [
        "wqi_score",
        "year",
        "ph",
        "dissolved_oxygen",
        "temperature",
        "turbidity",
        "nitrate",
        "conductance",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"US samples missing required columns: {missing}")

    param_cols = [
        "ph",
        "dissolved_oxygen",
        "temperature",
        "turbidity",
        "nitrate",
        "conductance",
    ]
    non_null_counts = df[param_cols].notna().sum(axis=1)
    mask = non_null_counts >= min_params
    filtered = df.loc[mask].copy()

    if filtered.empty:
        raise ValueError(
            f"No US samples with at least {min_params} non-null parameters (had {len(df)} rows total)."
        )

    return filtered


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Recreate the inference-time feature matrix for the US regressor.

    Uses the same helper as the Streamlit app so calibration is aligned with
    real predictions.
    """

    feature_frames = []
    for _, row in df.iterrows():
        feature_frames.append(
            prepare_us_features_for_prediction(
                ph=row.get("ph"),
                dissolved_oxygen=row.get("dissolved_oxygen"),
                temperature=row.get("temperature"),
                turbidity=row.get("turbidity"),
                nitrate=row.get("nitrate"),
                conductance=row.get("conductance"),
                year=int(row.get("year")),
            )
        )

    features = pd.concat(feature_frames, ignore_index=True)
    return features


def main() -> int:
    args = parse_args()
    samples_path = Path(args.samples_path)

    print(f"Loading US samples from {samples_path} ...")
    df_us = load_us_samples(samples_path, min_params=args.min_params)
    print(f"Using {len(df_us)} US samples after parameter filter ≥ {args.min_params}.")

    # Ground truth WQI from NSF calculator
    y_true = df_us["wqi_score"].to_numpy(dtype=float)

    # Latest regressor
    regressor_path = get_latest_model_path("regressor")
    if regressor_path is None:
        raise RuntimeError("No regressor_*.joblib found under data/models.")

    print(f"Loading base regressor from {regressor_path} (without calibration)...")
    regressor = WQIPredictionRegressor.load(regressor_path, load_calibration=False)

    # Build feature matrix aligned to inference path
    X_features = build_feature_matrix(df_us)

    # Predict on US samples (no calibration yet)
    y_pred = regressor.predict(X_features, apply_calibration=False)

    # Train calibrator
    calibrator = DomainCalibrator(increasing=True, out_of_bounds="clip")
    metrics = calibrator.fit(
        y_pred=y_pred,
        y_true=y_true,
        validation_split=args.validation_split,
        random_state=42,
    )

    # Save calibrator alongside the regressor using the expected naming
    regressor_path_obj = Path(regressor_path)
    calibrator_name = regressor_path_obj.name.replace("regressor_", "calibrator_us_")
    calibrator_path = regressor_path_obj.with_name(calibrator_name)
    calibrator.save(str(calibrator_path))

    print("\nCalibration summary (training set):")
    train = metrics["training"]
    print(
        f"  MAE: {train['mae_before']:.2f} → {train['mae_after']:.2f}\n"
        f"  RMSE: {train['rmse_before']:.2f} → {train['rmse_after']:.2f}\n"
        f"  R²: {train['r2_before']:.3f} → {train['r2_after']:.3f}"
    )

    val = metrics.get("validation")
    if val:
        print("\nCalibration summary (validation set):")
        print(
            f"  MAE: {val['mae_before']:.2f} → {val['mae_after']:.2f}\n"
            f"  RMSE: {val['rmse_before']:.2f} → {val['rmse_after']:.2f}\n"
            f"  R²: {val['r2_before']:.3f} → {val['r2_after']:.3f}"
        )

    print(f"\nCalibrator saved to: {calibrator_path}")
    print("Regressor will now auto-load this calibration for US predictions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
