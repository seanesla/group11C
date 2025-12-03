#!/usr/bin/env python3
"""
Train US-Only RandomForest Model

This script trains a RandomForest regressor EXCLUSIVELY on US ground truth data,
addressing Agent 12's criticism: "You should have retrained instead of calibrating."

Scientific comparison: US-only model vs Calibrated EU model
- US model: 128 samples, RIGHT distribution (DO mean 8.82 mg/L)
- EU model: 2,939 samples, WRONG distribution (DO mean 1.67 mg/L) + calibration

Research question: Is quality (right distribution) better than quantity (more samples)?
"""

import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.us_data_features import prepare_us_features_for_prediction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_us_ground_truth(min_params: int = 6):
    """
    Load US samples with ground truth WQI.

    Returns:
        DataFrame with columns: features (18 cols), actual_wqi, metadata
    """
    logger.info("Loading US ground truth data...")

    with open('tests/geographic_coverage_191_results.json', 'r') as f:
        data = json.load(f)

    results = data['results']

    # Extract samples with complete data
    samples = []

    for r in results:
        # Check if has WQI ground truth
        if not r.get('wqi') or not r['wqi'].get('score'):
            continue

        # Check if has sufficient parameters
        if r['wqi']['parameter_count'] < min_params:
            continue

        # Extract water quality parameters
        params = r['wqi']['parameters']

        # Prepare features using same pipeline as production
        features_df = prepare_us_features_for_prediction(
            ph=params.get('ph'),
            dissolved_oxygen=params.get('dissolved_oxygen'),
            temperature=params.get('temperature'),
            turbidity=params.get('turbidity'),  # Will be None for most samples
            nitrate=params.get('nitrate'),
            conductance=params.get('conductance'),
            year=2024  # Use consistent year for US samples
        )

        # Add ground truth and metadata
        sample = {
            'features': features_df.values[0],  # 18 features
            'actual_wqi': r['wqi']['score'],
            'zip_code': r['zip_code'],
            'state': r['geolocation']['state_code'] if r['geolocation'] else 'Unknown',
            'parameter_count': r['wqi']['parameter_count'],
            'dissolved_oxygen': params.get('dissolved_oxygen'),
        }
        samples.append(sample)

    logger.info(f"Loaded {len(samples)} US samples with {min_params}/6 parameters")

    # Convert to arrays
    X = np.array([s['features'] for s in samples])
    y = np.array([s['actual_wqi'] for s in samples])

    # Metadata DataFrame
    metadata = pd.DataFrame([{
        'zip_code': s['zip_code'],
        'state': s['state'],
        'parameter_count': s['parameter_count'],
        'dissolved_oxygen': s['dissolved_oxygen']
    } for s in samples])

    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"WQI range: [{y.min():.2f}, {y.max():.2f}], mean={y.mean():.2f}, std={y.std():.2f}")
    logger.info(f"DO range: [{metadata['dissolved_oxygen'].min():.2f}, {metadata['dissolved_oxygen'].max():.2f}], mean={metadata['dissolved_oxygen'].mean():.2f}")

    return X, y, metadata


def train_us_model_with_cv(X, y, n_folds=10, random_state=42):
    """
    Train US-only RandomForest with proper regularization and cross-validation.

    Agent 12's recommendation: "128 samples IS enough for RandomForest with regularization"

    Args:
        X: Feature matrix (n_samples, 25)
        y: Ground truth WQI (n_samples,)
        n_folds: Number of CV folds
        random_state: Random seed

    Returns:
        Trained model, CV scores, training metrics
    """
    logger.info("=" * 100)
    logger.info("TRAINING US-ONLY RANDOMFOREST MODEL")
    logger.info("=" * 100)
    logger.info("")

    # Hyperparameters designed for small dataset (128 samples)
    # Regularization to prevent overfitting:
    # - max_depth=10: Limit tree depth
    # - min_samples_leaf=5: Require 5 samples per leaf (5/128 = 3.9%)
    # - n_estimators=100: Enough trees for stability, not too many for overfitting
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,  # Prevent overfitting on small dataset
        min_samples_leaf=5,  # Require 5 samples per leaf
        min_samples_split=10,  # Require 10 samples to split
        random_state=random_state,
        n_jobs=-1,
        bootstrap=True
    )

    logger.info("Model hyperparameters:")
    logger.info(f"  n_estimators: {model.n_estimators}")
    logger.info(f"  max_depth: {model.max_depth} (regularization)")
    logger.info(f"  min_samples_leaf: {model.min_samples_leaf} ({model.min_samples_leaf/len(X)*100:.1f}% of data)")
    logger.info(f"  min_samples_split: {model.min_samples_split}")
    logger.info("")

    # 10-Fold Cross-Validation
    logger.info(f"Running {n_folds}-Fold Cross-Validation...")
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    cv_mae_scores = -cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_absolute_error', n_jobs=-1)
    cv_rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1))
    cv_r2_scores = cross_val_score(model, X, y, cv=kfold, scoring='r2', n_jobs=-1)

    logger.info("")
    logger.info("Cross-Validation Results:")
    logger.info(f"  MAE:  {cv_mae_scores.mean():.2f} ± {cv_mae_scores.std():.2f} (range: {cv_mae_scores.min():.2f} - {cv_mae_scores.max():.2f})")
    logger.info(f"  RMSE: {cv_rmse_scores.mean():.2f} ± {cv_rmse_scores.std():.2f}")
    logger.info(f"  R²:   {cv_r2_scores.mean():.4f} ± {cv_r2_scores.std():.4f}")
    logger.info("")

    # Train on full dataset for final model
    logger.info("Training final model on full US dataset...")
    model.fit(X, y)

    # In-sample metrics (for comparison - will be optimistic)
    y_pred = model.predict(X)
    train_mae = mean_absolute_error(y, y_pred)
    train_rmse = np.sqrt(mean_squared_error(y, y_pred))
    train_r2 = r2_score(y, y_pred)

    logger.info("Full training set metrics (in-sample):")
    logger.info(f"  MAE:  {train_mae:.2f}")
    logger.info(f"  RMSE: {train_rmse:.2f}")
    logger.info(f"  R²:   {train_r2:.4f}")
    logger.info("")

    # Feature importances
    feature_names = [
        'ph', 'dissolved_oxygen', 'temperature', 'nitrate', 'conductance',
        'turbidity_missing', 'ph_deviation', 'do_temp_ratio', 'pollution_stress',
        'temp_stress', 'conductance_high', 'conductance_very_high',
        'nitrate_elevated', 'nitrate_concerning', 'year', 'decade',
        'years_since_1991', 'ph_missing', 'dissolved_oxygen_missing',
        'temperature_missing', 'nitrate_missing', 'conductance_missing',
        'missing_count', 'has_turbidity', 'complete_wqi_params'
    ]

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    logger.info("Top 10 Most Important Features:")
    for i, idx in enumerate(indices[:10], 1):
        logger.info(f"  {i}. {feature_names[idx]}: {importances[idx]:.4f}")
    logger.info("")

    return model, {
        'cv_mae': cv_mae_scores,
        'cv_rmse': cv_rmse_scores,
        'cv_r2': cv_r2_scores,
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'n_samples': len(X),
        'n_features': X.shape[1]
    }


def compare_to_calibrated_eu_model(X, y, metadata, us_model):
    """
    Compare US-only model to calibrated EU model.

    Loads existing calibrated EU model and compares performance on same 128 samples.

    Args:
        X: Feature matrix
        y: Ground truth WQI
        metadata: Sample metadata
        us_model: Trained US-only model

    Returns:
        Comparison metrics dict
    """
    logger.info("=" * 100)
    logger.info("COMPARING US-ONLY MODEL VS CALIBRATED EU MODEL")
    logger.info("=" * 100)
    logger.info("")

    # Load EU model
    from src.models.regressor import WQIPredictionRegressor
    from src.models.domain_calibrator import DomainCalibrator

    eu_model_path = 'data/models/regressor_20251117_042831.joblib'
    calibrator_path = 'data/models/calibrator_us_20251117_042831.joblib'

    logger.info(f"Loading EU model: {eu_model_path}")
    eu_regressor = WQIPredictionRegressor.load(eu_model_path, load_calibration=False)

    logger.info(f"Loading calibrator: {calibrator_path}")
    calibrator = DomainCalibrator.load(calibrator_path)

    # Predict with EU model (uncalibrated)
    y_pred_eu_uncalibrated = eu_regressor.predict(X, apply_calibration=False)

    # Predict with EU model (calibrated)
    y_pred_eu_calibrated = calibrator.calibrate(y_pred_eu_uncalibrated)

    # Predict with US model
    y_pred_us = us_model.predict(X)

    # Compute metrics
    mae_eu_uncal = mean_absolute_error(y, y_pred_eu_uncalibrated)
    mae_eu_cal = mean_absolute_error(y, y_pred_eu_calibrated)
    mae_us = mean_absolute_error(y, y_pred_us)

    rmse_eu_uncal = np.sqrt(mean_squared_error(y, y_pred_eu_uncalibrated))
    rmse_eu_cal = np.sqrt(mean_squared_error(y, y_pred_eu_calibrated))
    rmse_us = np.sqrt(mean_squared_error(y, y_pred_us))

    r2_eu_uncal = r2_score(y, y_pred_eu_uncalibrated)
    r2_eu_cal = r2_score(y, y_pred_eu_calibrated)
    r2_us = r2_score(y, y_pred_us)

    logger.info("PERFORMANCE COMPARISON (on 128 US samples):")
    logger.info("")
    logger.info("EU Model (Uncalibrated - trained on 2,939 European samples):")
    logger.info(f"  MAE:  {mae_eu_uncal:.2f} points")
    logger.info(f"  RMSE: {rmse_eu_uncal:.2f} points")
    logger.info(f"  R²:   {r2_eu_uncal:.4f}")
    logger.info("")
    logger.info("EU Model (Calibrated - EU model + isotonic calibration on 128 US samples):")
    logger.info(f"  MAE:  {mae_eu_cal:.2f} points")
    logger.info(f"  RMSE: {rmse_eu_cal:.2f} points")
    logger.info(f"  R²:   {r2_eu_cal:.4f}")
    logger.info("")
    logger.info("US Model (Trained from scratch on 128 US samples):")
    logger.info(f"  MAE:  {mae_us:.2f} points")
    logger.info(f"  RMSE: {rmse_us:.2f} points")
    logger.info(f"  R²:   {r2_us:.4f}")
    logger.info("")

    # Improvement analysis
    logger.info("IMPROVEMENT ANALYSIS:")
    logger.info("")
    logger.info(f"Calibration improvement (EU uncal → EU cal):")
    logger.info(f"  MAE:  {mae_eu_uncal:.2f} → {mae_eu_cal:.2f} ({((mae_eu_uncal - mae_eu_cal)/mae_eu_uncal*100):.1f}% reduction)")
    logger.info("")
    logger.info(f"US model vs Calibrated EU model:")
    if mae_us < mae_eu_cal:
        logger.info(f"  ✓ US model is BETTER: MAE {mae_us:.2f} vs {mae_eu_cal:.2f} ({((mae_eu_cal - mae_us)/mae_eu_cal*100):.1f}% better)")
    elif mae_us > mae_eu_cal:
        logger.info(f"  ✗ US model is WORSE: MAE {mae_us:.2f} vs {mae_eu_cal:.2f} ({((mae_us - mae_eu_cal)/mae_eu_cal*100):.1f}% worse)")
    else:
        logger.info(f"  = TIE: Both have MAE {mae_us:.2f}")
    logger.info("")

    # Edge case analysis: Low DO performance
    low_do_mask = metadata['dissolved_oxygen'] < 6.0
    n_low_do = low_do_mask.sum()

    if n_low_do > 0:
        logger.info(f"EDGE CASE ANALYSIS: Low DO (<6 mg/L) - {n_low_do} samples")
        logger.info("")

        mae_eu_cal_lowdo = mean_absolute_error(y[low_do_mask], y_pred_eu_calibrated[low_do_mask])
        mae_us_lowdo = mean_absolute_error(y[low_do_mask], y_pred_us[low_do_mask])

        logger.info(f"  EU Calibrated: MAE = {mae_eu_cal_lowdo:.2f}")
        logger.info(f"  US Model:      MAE = {mae_us_lowdo:.2f}")

        if mae_us_lowdo < mae_eu_cal_lowdo:
            logger.info(f"  ✓ US model better on low-DO: {((mae_eu_cal_lowdo - mae_us_lowdo)/mae_eu_cal_lowdo*100):.1f}% improvement")
        else:
            logger.info(f"  ✗ US model worse on low-DO: {((mae_us_lowdo - mae_eu_cal_lowdo)/mae_eu_cal_lowdo*100):.1f}% worse")
        logger.info("")

    return {
        'mae_eu_uncal': mae_eu_uncal,
        'mae_eu_cal': mae_eu_cal,
        'mae_us': mae_us,
        'rmse_eu_uncal': rmse_eu_uncal,
        'rmse_eu_cal': rmse_eu_cal,
        'rmse_us': rmse_us,
        'r2_eu_uncal': r2_eu_uncal,
        'r2_eu_cal': r2_eu_cal,
        'r2_us': r2_us,
        'predictions': {
            'eu_uncalibrated': y_pred_eu_uncalibrated,
            'eu_calibrated': y_pred_eu_calibrated,
            'us_model': y_pred_us,
            'actual': y
        }
    }


def generate_comparison_plots(comparison_results, save_dir='data/models'):
    """Generate side-by-side comparison plots."""
    predictions = comparison_results['predictions']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: EU Uncalibrated
    ax = axes[0]
    ax.scatter(predictions['eu_uncalibrated'], predictions['actual'], alpha=0.5, s=50, color='red')
    ax.plot([predictions['actual'].min(), predictions['actual'].max()],
            [predictions['actual'].min(), predictions['actual'].max()],
            'k--', lw=2, label='Perfect prediction')
    ax.set_xlabel('EU Model Prediction (Uncalibrated)', fontsize=12)
    ax.set_ylabel('Actual WQI', fontsize=12)
    ax.set_title(f'EU Model (Uncalibrated)\nMAE = {comparison_results["mae_eu_uncal"]:.2f}',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: EU Calibrated
    ax = axes[1]
    ax.scatter(predictions['eu_calibrated'], predictions['actual'], alpha=0.5, s=50, color='orange')
    ax.plot([predictions['actual'].min(), predictions['actual'].max()],
            [predictions['actual'].min(), predictions['actual'].max()],
            'k--', lw=2, label='Perfect prediction')
    ax.set_xlabel('EU Model Prediction (Calibrated)', fontsize=12)
    ax.set_ylabel('Actual WQI', fontsize=12)
    ax.set_title(f'EU Model + Calibration\nMAE = {comparison_results["mae_eu_cal"]:.2f}',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: US Model
    ax = axes[2]
    ax.scatter(predictions['us_model'], predictions['actual'], alpha=0.5, s=50, color='green')
    ax.plot([predictions['actual'].min(), predictions['actual'].max()],
            [predictions['actual'].min(), predictions['actual'].max()],
            'k--', lw=2, label='Perfect prediction')
    ax.set_xlabel('US Model Prediction', fontsize=12)
    ax.set_ylabel('Actual WQI', fontsize=12)
    ax.set_title(f'US-Only Model\nMAE = {comparison_results["mae_us"]:.2f}',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Model Comparison: EU Uncalibrated vs EU Calibrated vs US-Only',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = f"{save_dir}/us_model_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Comparison plot saved to {save_path}")
    plt.close()


def main():
    """Main execution function."""
    logger.info("=" * 100)
    logger.info("PHASE 1.1: TRAIN US-ONLY MODEL & COMPARE TO CALIBRATION")
    logger.info("=" * 100)
    logger.info("")
    logger.info("Research Question: Is retraining on 128 US samples better than")
    logger.info("                    calibrating a model trained on 2,939 EU samples?")
    logger.info("")
    logger.info("Agent 12's hypothesis: 'Quality (right distribution) > Quantity (wrong distribution)'")
    logger.info("")

    # Load US data
    X, y, metadata = load_us_ground_truth(min_params=6)

    # Train US model with cross-validation
    us_model, cv_metrics = train_us_model_with_cv(X, y, n_folds=10)

    # Compare to calibrated EU model
    comparison = compare_to_calibrated_eu_model(X, y, metadata, us_model)

    # Generate comparison plots
    generate_comparison_plots(comparison)

    # Final recommendation
    logger.info("=" * 100)
    logger.info("RECOMMENDATION")
    logger.info("=" * 100)
    logger.info("")

    if comparison['mae_us'] < comparison['mae_eu_cal']:
        logger.info("✓ RECOMMENDATION: Use US-only model")
        logger.info(f"  - Better accuracy: MAE {comparison['mae_us']:.2f} vs {comparison['mae_eu_cal']:.2f}")
        logger.info(f"  - Simpler: No need for calibration layer")
        logger.info(f"  - More interpretable: Trained on same distribution as predictions")
        logger.info(f"  - Agent 12 was RIGHT: Quality > Quantity for domain shift")
    else:
        logger.info("✓ RECOMMENDATION: Keep calibrated EU model")
        logger.info(f"  - Better accuracy: MAE {comparison['mae_eu_cal']:.2f} vs {comparison['mae_us']:.2f}")
        logger.info(f"  - Benefits from larger European dataset (2,939 samples)")
        logger.info(f"  - Calibration successfully corrects domain shift")
        logger.info(f"  - Agent 12 was WRONG: Calibration approach is superior")

    logger.info("")
    logger.info("Next steps:")
    logger.info("- Review comparison plots: data/models/us_model_comparison.png")
    logger.info("- Proceed to Phase 1.2: Run full validation suite on chosen model")
    logger.info("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
