#!/usr/bin/env python3
"""
Train US Domain Calibration

This script trains an isotonic regression calibrator on US ground truth data
to correct the systematic ~20 point under-prediction bias from EU-trained models.

Data source: 128 US samples with:
- ML predictions from European-trained RandomForest regressor
- Actual WQI scores from rule-based NSF calculation
- Complete 6/6 parameter coverage

Expected result: MAE reduction from ~20 points → 6-10 points
"""

import sys
import json
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.domain_calibrator import DomainCalibrator
from src.models.regressor import WQIPredictionRegressor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_us_ground_truth(min_params: int = 6):
    """
    Load US samples with ground truth WQI and ML predictions.

    Args:
        min_params: Minimum number of WQI parameters required

    Returns:
        Tuple of (ml_predictions, actual_wqi, metadata)
    """
    logger.info("Loading US ground truth data...")

    with open('tests/geographic_coverage_191_results.json', 'r') as f:
        data = json.load(f)

    results = data['results']

    # Extract samples with complete data
    ml_predictions = []
    actual_wqi = []
    metadata = []

    for r in results:
        # Check if has WQI ground truth
        if not r.get('wqi') or not r['wqi'].get('score'):
            continue

        # Check if has sufficient parameters
        if r['wqi']['parameter_count'] < min_params:
            continue

        # Check if has ML prediction
        ml_preds = r.get('ml_predictions')
        if not ml_preds or ml_preds.get('regressor_wqi') is None:
            continue

        ml_predictions.append(ml_preds['regressor_wqi'])
        actual_wqi.append(r['wqi']['score'])
        metadata.append({
            'zip_code': r['zip_code'],
            'state': r['geolocation']['state_code'] if r['geolocation'] else 'Unknown',
            'parameter_count': r['wqi']['parameter_count']
        })

    ml_predictions = np.array(ml_predictions)
    actual_wqi = np.array(actual_wqi)

    logger.info(f"Loaded {len(ml_predictions)} US samples")
    logger.info(f"  ML predictions: {ml_predictions.min():.2f} - {ml_predictions.max():.2f} (mean {ml_predictions.mean():.2f})")
    logger.info(f"  Actual WQI:     {actual_wqi.min():.2f} - {actual_wqi.max():.2f} (mean {actual_wqi.mean():.2f})")
    logger.info(f"  Systematic bias: {(actual_wqi - ml_predictions).mean():.2f} points (under-prediction)")
    logger.info(f"  Current MAE:     {np.abs(actual_wqi - ml_predictions).mean():.2f} points")

    return ml_predictions, actual_wqi, metadata


def plot_calibration_analysis(
    ml_pred_train, wqi_train,
    ml_pred_test, wqi_test,
    calibrator,
    save_dir: str = "data/models"
):
    """Generate calibration analysis plots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 1. Training data: Before vs After
    ax = axes[0, 0]
    wqi_train_calibrated = calibrator.calibrate(ml_pred_train)
    ax.scatter(ml_pred_train, wqi_train, alpha=0.5, s=50, label='Training data', color='blue')
    ax.plot([wqi_train.min(), wqi_train.max()],
            [wqi_train.min(), wqi_train.max()],
            'r--', lw=2, label='Perfect prediction')
    ax.set_xlabel('ML Prediction (Uncalibrated)', fontsize=12)
    ax.set_ylabel('Actual WQI', fontsize=12)
    ax.set_title(f'Training Set BEFORE Calibration\nMAE = {np.abs(wqi_train - ml_pred_train).mean():.2f} points', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Training data: After calibration
    ax = axes[0, 1]
    ax.scatter(wqi_train_calibrated, wqi_train, alpha=0.5, s=50, label='Training data', color='green')
    ax.plot([wqi_train.min(), wqi_train.max()],
            [wqi_train.min(), wqi_train.max()],
            'r--', lw=2, label='Perfect prediction')
    ax.set_xlabel('ML Prediction (Calibrated)', fontsize=12)
    ax.set_ylabel('Actual WQI', fontsize=12)
    ax.set_title(f'Training Set AFTER Calibration\nMAE = {np.abs(wqi_train - wqi_train_calibrated).mean():.2f} points', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Test data: Before vs After
    ax = axes[1, 0]
    wqi_test_calibrated = calibrator.calibrate(ml_pred_test)
    ax.scatter(ml_pred_test, wqi_test, alpha=0.5, s=50, label='Test data', color='orange')
    ax.plot([wqi_test.min(), wqi_test.max()],
            [wqi_test.min(), wqi_test.max()],
            'r--', lw=2, label='Perfect prediction')
    ax.set_xlabel('ML Prediction (Uncalibrated)', fontsize=12)
    ax.set_ylabel('Actual WQI', fontsize=12)
    ax.set_title(f'Test Set BEFORE Calibration\nMAE = {np.abs(wqi_test - ml_pred_test).mean():.2f} points', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Test data: After calibration
    ax = axes[1, 1]
    ax.scatter(wqi_test_calibrated, wqi_test, alpha=0.5, s=50, label='Test data', color='purple')
    ax.plot([wqi_test.min(), wqi_test.max()],
            [wqi_test.min(), wqi_test.max()],
            'r--', lw=2, label='Perfect prediction')
    ax.set_xlabel('ML Prediction (Calibrated)', fontsize=12)
    ax.set_ylabel('Actual WQI', fontsize=12)
    ax.set_title(f'Test Set AFTER Calibration\nMAE = {np.abs(wqi_test - wqi_test_calibrated).mean():.2f} points', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('US Domain Calibration: Isotonic Regression Results',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    save_path = f"{save_dir}/calibration_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Calibration analysis plot saved to {save_path}")
    plt.close()


def main():
    """Train and save US domain calibration."""
    logger.info("=" * 100)
    logger.info("TRAIN US DOMAIN CALIBRATION")
    logger.info("=" * 100)
    logger.info("")

    # Load US ground truth data
    ml_predictions, actual_wqi, metadata = load_us_ground_truth(min_params=6)

    logger.info("")
    logger.info("=" * 100)
    logger.info(f"Training calibration on {len(ml_predictions)} US samples")
    logger.info("=" * 100)
    logger.info("")

    # Create calibrator
    calibrator = DomainCalibrator(
        increasing=True,  # WQI should increase monotonically
        out_of_bounds='clip'  # Safe handling of out-of-range predictions
    )

    # Train calibrator (20% validation split)
    metrics = calibrator.fit(
        y_pred=ml_predictions,
        y_true=actual_wqi,
        validation_split=0.2,
        random_state=42
    )

    logger.info("")
    logger.info("=" * 100)
    logger.info("CALIBRATION TRAINING RESULTS")
    logger.info("=" * 100)
    logger.info("")

    # Training set results
    train_mae_before = metrics['training']['mae_before']
    train_mae_after = metrics['training']['mae_after']
    train_improvement = ((train_mae_before - train_mae_after) / train_mae_before) * 100

    logger.info(f"Training Set ({metrics['training']['n_samples']} samples):")
    logger.info(f"  MAE: {train_mae_before:.2f} → {train_mae_after:.2f} points")
    logger.info(f"  Improvement: {train_improvement:.1f}% error reduction")
    logger.info(f"  R²: {metrics['training']['r2_before']:.4f} → {metrics['training']['r2_after']:.4f}")
    logger.info("")

    # Validation set results
    if metrics['validation']:
        val_mae_before = metrics['validation']['mae_before']
        val_mae_after = metrics['validation']['mae_after']
        val_improvement = ((val_mae_before - val_mae_after) / val_mae_before) * 100

        logger.info(f"Validation Set ({metrics['validation']['n_samples']} samples):")
        logger.info(f"  MAE: {val_mae_before:.2f} → {val_mae_after:.2f} points")
        logger.info(f"  Improvement: {val_improvement:.1f}% error reduction")
        logger.info(f"  R²: {metrics['validation']['r2_before']:.4f} → {metrics['validation']['r2_after']:.4f}")
        logger.info("")

    # Check success criteria
    logger.info("=" * 100)
    logger.info("SUCCESS CRITERIA CHECK")
    logger.info("=" * 100)
    logger.info("")

    success = True

    # Criterion 1: Validation MAE < 10 points
    if metrics['validation'] and val_mae_after < 10:
        logger.info(f"✓ Validation MAE < 10 points: {val_mae_after:.2f}")
    elif metrics['validation']:
        logger.warning(f"✗ Validation MAE >= 10 points: {val_mae_after:.2f}")
        success = False

    # Criterion 2: Error reduction > 50%
    if metrics['validation'] and val_improvement > 50:
        logger.info(f"✓ Error reduction > 50%: {val_improvement:.1f}%")
    elif metrics['validation']:
        logger.warning(f"⚠ Error reduction < 50%: {val_improvement:.1f}%")

    # Criterion 3: Good generalization (val/train ratio < 2.0)
    if metrics['validation']:
        ratio = val_mae_after / train_mae_after
        if ratio < 2.0:
            logger.info(f"✓ Good generalization: validation/train ratio = {ratio:.2f}x")
        else:
            logger.warning(f"⚠ Possible overfitting: validation/train ratio = {ratio:.2f}x")

    logger.info("")

    # Save calibrator
    logger.info("=" * 100)
    logger.info("SAVING CALIBRATOR")
    logger.info("=" * 100)
    logger.info("")

    # Use same timestamp as current regressor models
    calibrator_path = calibrator.save("data/models/calibrator_us_20251117_042831.joblib")

    logger.info("")
    logger.info(f"Calibrator saved to: {calibrator_path}")
    logger.info("To use with regressor: place in same directory as regressor_20251117_042831.joblib")
    logger.info("")

    # Generate plots
    logger.info("=" * 100)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 100)
    logger.info("")

    # Split data for plotting (same split used in training)
    from sklearn.model_selection import train_test_split
    ml_train, ml_test, wqi_train, wqi_test = train_test_split(
        ml_predictions, actual_wqi,
        test_size=0.2,
        random_state=42
    )

    plot_calibration_analysis(ml_train, wqi_train, ml_test, wqi_test, calibrator)

    # Final summary
    logger.info("")
    logger.info("=" * 100)
    if success:
        logger.info("✓✓✓ CALIBRATION TRAINING SUCCESSFUL ✓✓✓")
    else:
        logger.info("⚠ CALIBRATION TRAINED BUT DID NOT MEET ALL SUCCESS CRITERIA")
    logger.info("=" * 100)
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Run full 191 ZIP validation: python3 tests/validate_core_models_quick.py")
    logger.info("2. Integrate into Streamlit app")
    logger.info("3. Update documentation")
    logger.info("")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
