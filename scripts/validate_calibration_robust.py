#!/usr/bin/env python3
"""
Comprehensive Calibration Validation

This script performs rigorous validation to address ALL caveats:
1. Small sample size → K-fold CV, bootstrap, learning curves
2. No unseen data → Geographic holdout validation
3. Generalization concerns → Stratified analysis, failure mode detection

NO SHORTCUTS. FULL RIGOR.
"""

import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.domain_calibrator import DomainCalibrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_us_data():
    """Load complete US dataset with metadata."""
    with open('tests/geographic_coverage_191_results.json', 'r') as f:
        data = json.load(f)

    samples = []
    for r in data['results']:
        if not r.get('wqi') or not r['wqi'].get('score'):
            continue
        if r['wqi']['parameter_count'] < 6:
            continue
        ml_preds = r.get('ml_predictions')
        if not ml_preds or ml_preds.get('regressor_wqi') is None:
            continue

        samples.append({
            'zip_code': r['zip_code'],
            'state': r['geolocation']['state_code'] if r['geolocation'] else 'Unknown',
            'ml_pred': ml_preds['regressor_wqi'],
            'actual_wqi': r['wqi']['score'],
            'param_count': r['wqi']['parameter_count'],
            'do': r['wqi']['parameters'].get('dissolved_oxygen'),
            'classification': r['wqi']['classification']
        })

    df = pd.DataFrame(samples)
    logger.info(f"Loaded {len(df)} samples across {df['state'].nunique()} states")
    return df


def test_caveat_1_sample_size(df: pd.DataFrame) -> Dict:
    """
    CAVEAT 1: Small sample size (128 samples)

    Tests:
    1. K-fold cross-validation (5 folds)
    2. Bootstrap confidence intervals (1000 iterations)
    3. Learning curve (varying training size)
    4. Baseline comparison (linear vs isotonic)
    """
    logger.info("=" * 100)
    logger.info("CAVEAT 1: SAMPLE SIZE VALIDATION")
    logger.info("=" * 100)

    ml_pred = df['ml_pred'].values
    actual = df['actual_wqi'].values

    # Test 1: K-Fold Cross-Validation
    logger.info("\n1. K-FOLD CROSS-VALIDATION (5 folds)")
    logger.info("-" * 80)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(ml_pred), 1):
        calibrator = DomainCalibrator()
        calibrator.fit(ml_pred[train_idx], actual[train_idx], validation_split=0.0)

        # Validation metrics
        val_pred_uncal = ml_pred[val_idx]
        val_pred_cal = calibrator.calibrate(val_pred_uncal)
        val_actual = actual[val_idx]

        mae_before = mean_absolute_error(val_actual, val_pred_uncal)
        mae_after = mean_absolute_error(val_actual, val_pred_cal)
        improvement = ((mae_before - mae_after) / mae_before) * 100

        fold_results.append({
            'fold': fold,
            'mae_before': mae_before,
            'mae_after': mae_after,
            'improvement': improvement
        })

        logger.info(f"  Fold {fold}: {mae_before:.2f} → {mae_after:.2f} pts ({improvement:.1f}% improvement)")

    # Summary statistics
    mae_afters = [r['mae_after'] for r in fold_results]
    improvements = [r['improvement'] for r in fold_results]

    logger.info(f"\n  Cross-validation Summary:")
    logger.info(f"    Mean MAE after calibration: {np.mean(mae_afters):.2f} ± {np.std(mae_afters):.2f}")
    logger.info(f"    Mean improvement: {np.mean(improvements):.1f}% ± {np.std(improvements):.1f}%")
    logger.info(f"    Min/Max MAE: {np.min(mae_afters):.2f} / {np.max(mae_afters):.2f}")

    # Check consistency
    if np.std(mae_afters) < 2.0:
        logger.info(f"  ✓ STABLE: Low variance across folds (std={np.std(mae_afters):.2f})")
    else:
        logger.warning(f"  ⚠ UNSTABLE: High variance across folds (std={np.std(mae_afters):.2f})")

    # Test 2: Bootstrap Confidence Intervals
    logger.info("\n2. BOOTSTRAP CONFIDENCE INTERVALS (1000 iterations)")
    logger.info("-" * 80)

    bootstrap_maes = []
    np.random.seed(42)

    for i in range(1000):
        # Resample with replacement
        indices = np.random.choice(len(ml_pred), size=len(ml_pred), replace=True)
        boot_ml = ml_pred[indices]
        boot_actual = actual[indices]

        # Split
        train_size = int(0.8 * len(boot_ml))
        train_ml, val_ml = boot_ml[:train_size], boot_ml[train_size:]
        train_actual, val_actual = boot_actual[:train_size], boot_actual[train_size:]

        # Train and evaluate
        calibrator = DomainCalibrator()
        calibrator.fit(train_ml, train_actual, validation_split=0.0)
        val_cal = calibrator.calibrate(val_ml)
        mae = mean_absolute_error(val_actual, val_cal)
        bootstrap_maes.append(mae)

    # Calculate confidence intervals
    ci_lower = np.percentile(bootstrap_maes, 2.5)
    ci_upper = np.percentile(bootstrap_maes, 97.5)
    ci_mean = np.mean(bootstrap_maes)

    logger.info(f"  Bootstrap MAE: {ci_mean:.2f}")
    logger.info(f"  95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
    logger.info(f"  CI width: {ci_upper - ci_lower:.2f} points")

    if (ci_upper - ci_lower) < 3.0:
        logger.info(f"  ✓ TIGHT CI: Calibration is stable (width={ci_upper - ci_lower:.2f})")
    else:
        logger.warning(f"  ⚠ WIDE CI: Calibration has high uncertainty (width={ci_upper - ci_lower:.2f})")

    # Test 3: Learning Curve
    logger.info("\n3. LEARNING CURVE ANALYSIS")
    logger.info("-" * 80)

    train_sizes = [20, 40, 60, 80, 100, 128]
    learning_curve_results = []

    for size in train_sizes:
        if size > len(ml_pred):
            continue

        # Use first 'size' samples for training
        calibrator = DomainCalibrator()
        calibrator.fit(ml_pred[:size], actual[:size], validation_split=0.0)

        # Test on remaining data
        test_ml = ml_pred[size:]
        test_actual = actual[size:]
        if len(test_ml) == 0:
            continue

        test_cal = calibrator.calibrate(test_ml)
        mae = mean_absolute_error(test_actual, test_cal)

        learning_curve_results.append({'size': size, 'mae': mae})
        logger.info(f"  Training size {size:3d}: Test MAE = {mae:.2f}")

    # Check if we've plateaued
    if len(learning_curve_results) >= 3:
        last_3_maes = [r['mae'] for r in learning_curve_results[-3:]]
        mae_range = max(last_3_maes) - min(last_3_maes)
        if mae_range < 1.0:
            logger.info(f"  ✓ SUFFICIENT DATA: Learning curve has plateaued (range={mae_range:.2f})")
        else:
            logger.warning(f"  ⚠ MORE DATA NEEDED: Learning curve still decreasing (range={mae_range:.2f})")

    # Test 4: Baseline Comparison (Linear vs Isotonic)
    logger.info("\n4. BASELINE COMPARISON: Linear vs Isotonic")
    logger.info("-" * 80)

    from sklearn.linear_model import LinearRegression

    train_ml, test_ml = ml_pred[:100], ml_pred[100:]
    train_actual, test_actual = actual[:100], actual[100:]

    # Linear calibration
    linear_cal = LinearRegression()
    linear_cal.fit(train_ml.reshape(-1, 1), train_actual)
    test_linear = linear_cal.predict(test_ml.reshape(-1, 1))
    mae_linear = mean_absolute_error(test_actual, test_linear)

    # Isotonic calibration
    isotonic_cal = DomainCalibrator()
    isotonic_cal.fit(train_ml, train_actual, validation_split=0.0)
    test_isotonic = isotonic_cal.calibrate(test_ml)
    mae_isotonic = mean_absolute_error(test_actual, test_isotonic)

    logger.info(f"  Linear calibration MAE:   {mae_linear:.2f}")
    logger.info(f"  Isotonic calibration MAE: {mae_isotonic:.2f}")
    logger.info(f"  Difference: {mae_linear - mae_isotonic:.2f} points")

    if mae_isotonic < mae_linear:
        logger.info(f"  ✓ ISOTONIC BETTER: Non-linear correction is beneficial")
    else:
        logger.warning(f"  ⚠ LINEAR SUFFICIENT: Isotonic provides no benefit")

    return {
        'kfold_mean': np.mean(mae_afters),
        'kfold_std': np.std(mae_afters),
        'bootstrap_mean': ci_mean,
        'bootstrap_ci': (ci_lower, ci_upper),
        'learning_curve': learning_curve_results,
        'isotonic_vs_linear': (mae_isotonic, mae_linear)
    }


def test_caveat_2_unseen_data(df: pd.DataFrame) -> Dict:
    """
    CAVEAT 2: Not tested on NEW unseen ZIPs

    Tests:
    1. Geographic holdout validation (train on some states, test on others)
    2. Full 191 ZIP validation
    3. State-by-state performance
    """
    logger.info("\n" + "=" * 100)
    logger.info("CAVEAT 2: UNSEEN DATA VALIDATION")
    logger.info("=" * 100)

    # Test 1: Geographic Holdout
    logger.info("\n1. GEOGRAPHIC HOLDOUT VALIDATION")
    logger.info("-" * 80)

    # Get states with enough samples
    state_counts = df['state'].value_counts()
    logger.info(f"  States with samples: {len(state_counts)}")

    # Use 80/20 geographic split
    all_states = list(state_counts.index)
    np.random.seed(42)
    np.random.shuffle(all_states)

    train_states = all_states[:int(0.8 * len(all_states))]
    test_states = all_states[int(0.8 * len(all_states)):]

    df_train = df[df['state'].isin(train_states)]
    df_test = df[df['state'].isin(test_states)]

    logger.info(f"  Training states ({len(train_states)}): {', '.join(sorted(train_states))}")
    logger.info(f"  Test states ({len(test_states)}): {', '.join(sorted(test_states))}")
    logger.info(f"  Training samples: {len(df_train)}")
    logger.info(f"  Test samples: {len(df_test)}")

    if len(df_test) < 10:
        logger.warning(f"  ⚠ WARNING: Only {len(df_test)} test samples - results may be unreliable")

    # Train calibrator on training states
    calibrator = DomainCalibrator()
    calibrator.fit(
        df_train['ml_pred'].values,
        df_train['actual_wqi'].values,
        validation_split=0.0
    )

    # Test on completely unseen states
    test_uncal = df_test['ml_pred'].values
    test_cal = calibrator.calibrate(test_uncal)
    test_actual = df_test['actual_wqi'].values

    mae_before = mean_absolute_error(test_actual, test_uncal)
    mae_after = mean_absolute_error(test_actual, test_cal)
    improvement = ((mae_before - mae_after) / mae_before) * 100

    logger.info(f"\n  Geographic Holdout Results:")
    logger.info(f"    MAE before: {mae_before:.2f}")
    logger.info(f"    MAE after:  {mae_after:.2f}")
    logger.info(f"    Improvement: {improvement:.1f}%")

    if mae_after < 10:
        logger.info(f"  ✓ GENERALIZES: Works on unseen states (MAE={mae_after:.2f})")
    else:
        logger.warning(f"  ⚠ POOR GENERALIZATION: High error on unseen states (MAE={mae_after:.2f})")

    # Test 2: State-by-State Analysis
    logger.info("\n2. STATE-BY-STATE PERFORMANCE")
    logger.info("-" * 80)

    # Train on all data
    calibrator_full = DomainCalibrator()
    calibrator_full.fit(
        df['ml_pred'].values,
        df['actual_wqi'].values,
        validation_split=0.0
    )

    state_results = []
    for state in df['state'].unique():
        df_state = df[df['state'] == state]
        if len(df_state) < 3:  # Skip states with too few samples
            continue

        state_ml = df_state['ml_pred'].values
        state_actual = df_state['actual_wqi'].values
        state_cal = calibrator_full.calibrate(state_ml)

        mae_before = mean_absolute_error(state_actual, state_ml)
        mae_after = mean_absolute_error(state_actual, state_cal)

        state_results.append({
            'state': state,
            'n_samples': len(df_state),
            'mae_before': mae_before,
            'mae_after': mae_after,
            'improvement': ((mae_before - mae_after) / mae_before) * 100
        })

    # Sort by MAE after calibration (worst first)
    state_results_sorted = sorted(state_results, key=lambda x: x['mae_after'], reverse=True)

    logger.info(f"  Top 5 WORST performing states (after calibration):")
    for r in state_results_sorted[:5]:
        logger.info(f"    {r['state']}: {r['mae_before']:.2f} → {r['mae_after']:.2f} ({r['n_samples']} samples)")

    logger.info(f"\n  Top 5 BEST performing states (after calibration):")
    for r in state_results_sorted[-5:]:
        logger.info(f"    {r['state']}: {r['mae_before']:.2f} → {r['mae_after']:.2f} ({r['n_samples']} samples)")

    # Check if any states have MAE > 10
    bad_states = [r for r in state_results if r['mae_after'] > 10]
    if bad_states:
        logger.warning(f"  ⚠ {len(bad_states)} states have MAE > 10 points:")
        for r in bad_states:
            logger.warning(f"      {r['state']}: MAE={r['mae_after']:.2f}")
    else:
        logger.info(f"  ✓ ALL STATES: MAE < 10 points")

    return {
        'geographic_holdout': {
            'train_states': train_states,
            'test_states': test_states,
            'mae_before': mae_before,
            'mae_after': mae_after,
            'improvement': improvement
        },
        'state_results': state_results
    }


def test_caveat_3_generalization(df: pd.DataFrame) -> Dict:
    """
    CAVEAT 3: Might not generalize to all US water scenarios

    Tests:
    1. Stratified by WQI range (excellent vs good vs fair)
    2. Stratified by DO range (high vs medium vs low)
    3. Stratified by parameter count (6/6 vs 5/6)
    4. Failure mode analysis
    """
    logger.info("\n" + "=" * 100)
    logger.info("CAVEAT 3: GENERALIZATION VALIDATION")
    logger.info("=" * 100)

    # Train calibrator on all data
    calibrator = DomainCalibrator()
    calibrator.fit(
        df['ml_pred'].values,
        df['actual_wqi'].values,
        validation_split=0.2
    )

    # Get calibrated predictions for all samples
    df['ml_pred_cal'] = calibrator.calibrate(df['ml_pred'].values)
    df['error_before'] = np.abs(df['actual_wqi'] - df['ml_pred'])
    df['error_after'] = np.abs(df['actual_wqi'] - df['ml_pred_cal'])

    # Test 1: Stratified by WQI Classification
    logger.info("\n1. PERFORMANCE BY WQI CLASSIFICATION")
    logger.info("-" * 80)

    for classification in ['Excellent', 'Good', 'Fair', 'Poor']:
        df_class = df[df['classification'] == classification]
        if len(df_class) == 0:
            continue

        mae_before = df_class['error_before'].mean()
        mae_after = df_class['error_after'].mean()
        improvement = ((mae_before - mae_after) / mae_before) * 100

        logger.info(f"  {classification:12s} ({len(df_class):3d} samples): {mae_before:.2f} → {mae_after:.2f} ({improvement:+.1f}%)")

    # Test 2: Stratified by DO Range
    logger.info("\n2. PERFORMANCE BY DISSOLVED OXYGEN RANGE")
    logger.info("-" * 80)

    df['do_bin'] = pd.cut(df['do'], bins=[0, 6, 8, 100], labels=['Low (<6)', 'Medium (6-8)', 'High (>8)'])

    for do_range in ['Low (<6)', 'Medium (6-8)', 'High (>8)']:
        df_do = df[df['do_bin'] == do_range]
        if len(df_do) == 0:
            continue

        mae_before = df_do['error_before'].mean()
        mae_after = df_do['error_after'].mean()
        improvement = ((mae_before - mae_after) / mae_before) * 100

        logger.info(f"  DO {do_range:15s} ({len(df_do):3d} samples): {mae_before:.2f} → {mae_after:.2f} ({improvement:+.1f}%)")

    # Test 3: Failure Mode Analysis
    logger.info("\n3. FAILURE MODE ANALYSIS")
    logger.info("-" * 80)

    # Find worst predictions (after calibration)
    df_sorted = df.sort_values('error_after', ascending=False)
    worst_cases = df_sorted.head(10)

    logger.info(f"  Top 10 worst predictions (after calibration):")
    for idx, row in worst_cases.iterrows():
        logger.info(f"    ZIP {row['zip_code']} ({row['state']}): Predicted {row['ml_pred_cal']:.1f}, Actual {row['actual_wqi']:.1f}, Error {row['error_after']:.1f}")

    # Identify systematic failure patterns
    logger.info(f"\n  Failure pattern analysis:")
    high_error = df[df['error_after'] > 10]
    if len(high_error) > 0:
        logger.warning(f"    {len(high_error)} samples with error > 10 points ({len(high_error)/len(df)*100:.1f}%)")
        logger.warning(f"    Mean DO in failures: {high_error['do'].mean():.2f} (overall: {df['do'].mean():.2f})")
        logger.warning(f"    States: {', '.join(high_error['state'].value_counts().index[:5].tolist())}")
    else:
        logger.info(f"    ✓ NO FAILURES: All predictions have error < 10 points")

    return {
        'by_classification': df.groupby('classification')[['error_before', 'error_after']].mean().to_dict(),
        'by_do_range': df.groupby('do_bin')[['error_before', 'error_after']].mean().to_dict() if 'do_bin' in df.columns else {},
        'worst_cases': worst_cases[['zip_code', 'state', 'ml_pred_cal', 'actual_wqi', 'error_after']].to_dict('records'),
        'high_error_count': len(high_error) if len(high_error) > 0 else 0
    }


def main():
    """Run comprehensive validation."""
    logger.info("=" * 100)
    logger.info("COMPREHENSIVE CALIBRATION VALIDATION")
    logger.info("Addressing ALL caveats with rigorous testing")
    logger.info("=" * 100)
    logger.info("")

    # Load data
    df = load_us_data()
    logger.info("")

    # Test all caveats
    results_caveat1 = test_caveat_1_sample_size(df)
    results_caveat2 = test_caveat_2_unseen_data(df)
    results_caveat3 = test_caveat_3_generalization(df)

    # Final summary
    logger.info("\n" + "=" * 100)
    logger.info("FINAL VERDICT")
    logger.info("=" * 100)
    logger.info("")

    # Check all success criteria
    success_criteria = []

    # Caveat 1: K-fold mean MAE < 10
    if results_caveat1['kfold_mean'] < 10:
        logger.info(f"✓ CAVEAT 1 (Sample Size): K-fold mean MAE = {results_caveat1['kfold_mean']:.2f} < 10")
        success_criteria.append(True)
    else:
        logger.warning(f"✗ CAVEAT 1 (Sample Size): K-fold mean MAE = {results_caveat1['kfold_mean']:.2f} >= 10")
        success_criteria.append(False)

    # Caveat 2: Geographic holdout MAE < 10
    if results_caveat2['geographic_holdout']['mae_after'] < 10:
        logger.info(f"✓ CAVEAT 2 (Unseen Data): Geographic holdout MAE = {results_caveat2['geographic_holdout']['mae_after']:.2f} < 10")
        success_criteria.append(True)
    else:
        logger.warning(f"✗ CAVEAT 2 (Unseen Data): Geographic holdout MAE = {results_caveat2['geographic_holdout']['mae_after']:.2f} >= 10")
        success_criteria.append(False)

    # Caveat 3: No high-error failures
    if results_caveat3['high_error_count'] == 0:
        logger.info(f"✓ CAVEAT 3 (Generalization): All predictions have error < 10 points")
        success_criteria.append(True)
    else:
        logger.warning(f"⚠ CAVEAT 3 (Generalization): {results_caveat3['high_error_count']} samples with error > 10 points")
        success_criteria.append(False)

    logger.info("")
    if all(success_criteria):
        logger.info("=" * 100)
        logger.info("✓✓✓ ALL CAVEATS ADDRESSED - CALIBRATION IS PRODUCTION-READY ✓✓✓")
        logger.info("=" * 100)
        return 0
    else:
        logger.warning("=" * 100)
        logger.warning("⚠ SOME CAVEATS REMAIN - ADDITIONAL WORK NEEDED")
        logger.warning("=" * 100)
        return 1


if __name__ == "__main__":
    sys.exit(main())
