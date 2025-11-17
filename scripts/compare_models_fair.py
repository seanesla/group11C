#!/usr/bin/env python3
"""
Fair Model Comparison: US-Only vs Calibrated EU Model

This script fixes the critical methodological flaw identified by 10 independent reviewers:
The original comparison used IN-SAMPLE metrics (optimistic) for US model vs
OUT-OF-SAMPLE metrics (realistic) for calibrated model.

FAIR COMPARISON:
- Both models evaluated using 10-fold cross-validation
- Identical train/test splits
- Statistical significance tests (t-test, Wilcoxon, permutation)
- Bootstrap confidence intervals
- Proper reporting of generalization performance

Agent feedback: "Comparing in-sample (1.98) to out-of-sample (3.54) is fundamentally invalid."
"""

import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.isotonic import IsotonicRegression
from scipy import stats
import joblib
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.us_data_features import prepare_us_features_for_prediction
from src.models.regressor import WQIPredictionRegressor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_us_ground_truth(min_params: int = 6):
    """
    Load US samples with ground truth WQI, features, and EU model predictions.

    Returns:
        Tuple of (X_features, y_actual, ml_predictions, metadata)
    """
    logger.info("Loading US ground truth data...")

    with open('tests/geographic_coverage_191_results.json', 'r') as f:
        data = json.load(f)

    results = data['results']

    features_list = []
    actual_wqi_list = []
    ml_pred_list = []
    metadata_list = []

    for r in results:
        # Must have WQI ground truth
        if not r.get('wqi') or not r['wqi'].get('score'):
            continue

        # Must have sufficient parameters
        if r['wqi']['parameter_count'] < min_params:
            continue

        # Must have ML prediction from EU model
        ml_preds = r.get('ml_predictions')
        if not ml_preds or ml_preds.get('regressor_wqi') is None:
            continue

        # Extract parameters
        params = r['wqi']['parameters']

        # Prepare features using same pipeline as production
        features_df = prepare_us_features_for_prediction(
            ph=params.get('ph'),
            dissolved_oxygen=params.get('dissolved_oxygen'),
            temperature=params.get('temperature'),
            turbidity=params.get('turbidity'),
            nitrate=params.get('nitrate'),
            conductance=params.get('conductance'),
            year=2024  # Consistent year
        )

        features_list.append(features_df.values[0])
        actual_wqi_list.append(r['wqi']['score'])
        ml_pred_list.append(ml_preds['regressor_wqi'])
        metadata_list.append({
            'zip_code': r['zip_code'],
            'state': r['geolocation']['state_code'] if r['geolocation'] else 'Unknown',
            'parameter_count': r['wqi']['parameter_count'],
            'dissolved_oxygen': params.get('dissolved_oxygen')
        })

    X = np.array(features_list)
    y = np.array(actual_wqi_list)
    ml_preds = np.array(ml_pred_list)

    logger.info(f"Loaded {len(X)} US samples with {min_params}/6 parameters")
    logger.info(f"  Features shape: {X.shape}")
    logger.info(f"  Actual WQI: {y.min():.2f} - {y.max():.2f} (mean {y.mean():.2f})")
    logger.info(f"  EU ML predictions: {ml_preds.min():.2f} - {ml_preds.max():.2f} (mean {ml_preds.mean():.2f})")

    return X, y, ml_preds, metadata_list


def cross_val_us_only_model(X, y, n_folds=10, random_state=42):
    """
    10-fold cross-validation for US-only RandomForest model.

    Returns:
        Dictionary with CV scores and per-fold predictions
    """
    logger.info(f"\n{'='*80}")
    logger.info("US-ONLY MODEL: Cross-Validation")
    logger.info(f"{'='*80}")

    # Model configuration (same as train_us_only_model.py)
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,  # Regularization for small dataset
        min_samples_leaf=5,  # 3.9% of data
        min_samples_split=10,
        random_state=random_state,
        n_jobs=-1,
        bootstrap=True
    )

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    cv_predictions = np.zeros(len(y))
    fold_scores = {'mae': [], 'rmse': [], 'r2': []}

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train on fold
        model.fit(X_train, y_train)

        # Predict on validation
        y_pred = model.predict(X_val)
        cv_predictions[val_idx] = y_pred

        # Score fold
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)

        fold_scores['mae'].append(mae)
        fold_scores['rmse'].append(rmse)
        fold_scores['r2'].append(r2)

        logger.info(f"  Fold {fold_idx+1}/{n_folds}: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}")

    # Overall CV metrics
    cv_mae = np.mean(fold_scores['mae'])
    cv_mae_std = np.std(fold_scores['mae'])
    cv_rmse = np.mean(fold_scores['rmse'])
    cv_r2 = np.mean(fold_scores['r2'])

    logger.info(f"\n10-Fold CV Results:")
    logger.info(f"  MAE:  {cv_mae:.2f} ± {cv_mae_std:.2f} (range: {min(fold_scores['mae']):.2f} - {max(fold_scores['mae']):.2f})")
    logger.info(f"  RMSE: {cv_rmse:.2f} ± {np.std(fold_scores['rmse']):.2f}")
    logger.info(f"  R²:   {cv_r2:.3f} ± {np.std(fold_scores['r2']):.3f}")

    return {
        'cv_predictions': cv_predictions,
        'fold_scores': fold_scores,
        'cv_mae': cv_mae,
        'cv_mae_std': cv_mae_std,
        'cv_rmse': cv_rmse,
        'cv_r2': cv_r2
    }


def cross_val_calibrated_model(X, y, ml_preds, n_folds=10, random_state=42):
    """
    10-fold cross-validation for Calibrated EU model.

    For each fold:
    - Train isotonic calibrator on train ML predictions
    - Apply calibrator to validation ML predictions
    - Score on validation actual WQI

    Returns:
        Dictionary with CV scores and per-fold predictions
    """
    logger.info(f"\n{'='*80}")
    logger.info("CALIBRATED EU MODEL: Cross-Validation")
    logger.info(f"{'='*80}")

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    cv_predictions = np.zeros(len(y))
    fold_scores = {'mae': [], 'rmse': [], 'r2': []}

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
        ml_train, ml_val = ml_preds[train_idx], ml_preds[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train calibrator on fold
        calibrator = IsotonicRegression(
            y_min=0,
            y_max=100,
            increasing=True,
            out_of_bounds='clip'
        )
        calibrator.fit(ml_train, y_train)

        # Apply calibrator to validation
        y_pred = calibrator.predict(ml_val)
        cv_predictions[val_idx] = y_pred

        # Score fold
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)

        fold_scores['mae'].append(mae)
        fold_scores['rmse'].append(rmse)
        fold_scores['r2'].append(r2)

        logger.info(f"  Fold {fold_idx+1}/{n_folds}: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}")

    # Overall CV metrics
    cv_mae = np.mean(fold_scores['mae'])
    cv_mae_std = np.std(fold_scores['mae'])
    cv_rmse = np.mean(fold_scores['rmse'])
    cv_r2 = np.mean(fold_scores['r2'])

    logger.info(f"\n10-Fold CV Results:")
    logger.info(f"  MAE:  {cv_mae:.2f} ± {cv_mae_std:.2f} (range: {min(fold_scores['mae']):.2f} - {max(fold_scores['mae']):.2f})")
    logger.info(f"  RMSE: {cv_rmse:.2f} ± {np.std(fold_scores['rmse']):.2f}")
    logger.info(f"  R²:   {cv_r2:.3f} ± {np.std(fold_scores['r2']):.3f}")

    return {
        'cv_predictions': cv_predictions,
        'fold_scores': fold_scores,
        'cv_mae': cv_mae,
        'cv_mae_std': cv_mae_std,
        'cv_rmse': cv_rmse,
        'cv_r2': cv_r2
    }


def statistical_significance_tests(y_true, pred_us, pred_cal):
    """
    Test if US model is statistically better than calibrated model.

    Returns:
        Dictionary with test results and p-values
    """
    logger.info(f"\n{'='*80}")
    logger.info("STATISTICAL SIGNIFICANCE TESTS")
    logger.info(f"{'='*80}")

    # Compute errors for each sample
    errors_us = np.abs(y_true - pred_us)
    errors_cal = np.abs(y_true - pred_cal)

    results = {}

    # 1. Paired t-test (parametric)
    t_stat, p_ttest = stats.ttest_rel(errors_cal, errors_us)
    results['t_test'] = {'t_stat': t_stat, 'p_value': p_ttest}

    logger.info(f"\n1. Paired t-test:")
    logger.info(f"   H0: Calibrated and US models have equal mean error")
    logger.info(f"   t-statistic: {t_stat:.4f}")
    logger.info(f"   p-value: {p_ttest:.6f}")
    if p_ttest < 0.001:
        logger.info(f"   ✅ US model is SIGNIFICANTLY better (p < 0.001)")
    elif p_ttest < 0.05:
        logger.info(f"   ✅ US model is significantly better (p < 0.05)")
    else:
        logger.info(f"   ❌ No significant difference (p >= 0.05)")

    # 2. Wilcoxon signed-rank test (non-parametric)
    w_stat, p_wilcoxon = stats.wilcoxon(errors_cal, errors_us)
    results['wilcoxon'] = {'w_stat': w_stat, 'p_value': p_wilcoxon}

    logger.info(f"\n2. Wilcoxon signed-rank test (non-parametric):")
    logger.info(f"   H0: Calibrated and US models have equal median error")
    logger.info(f"   W-statistic: {w_stat:.4f}")
    logger.info(f"   p-value: {p_wilcoxon:.6f}")
    if p_wilcoxon < 0.05:
        logger.info(f"   ✅ US model is significantly better (p < 0.05)")
    else:
        logger.info(f"   ❌ No significant difference (p >= 0.05)")

    # 3. Permutation test (10,000 permutations)
    observed_diff = errors_cal.mean() - errors_us.mean()

    n_perm = 10000
    perm_diffs = []

    for _ in range(n_perm):
        # Randomly swap errors
        swap = np.random.binomial(1, 0.5, len(y_true))
        perm_errors_cal = np.where(swap, errors_cal, errors_us)
        perm_errors_us = np.where(swap, errors_us, errors_cal)
        perm_diff = perm_errors_cal.mean() - perm_errors_us.mean()
        perm_diffs.append(perm_diff)

    perm_diffs = np.array(perm_diffs)
    p_perm = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
    results['permutation'] = {'observed_diff': observed_diff, 'p_value': p_perm}

    logger.info(f"\n3. Permutation test ({n_perm:,} permutations):")
    logger.info(f"   Observed MAE difference: {observed_diff:.4f} points")
    logger.info(f"   p-value: {p_perm:.6f}")
    if p_perm < 0.05:
        logger.info(f"   ✅ US model is significantly better (p < 0.05)")
    else:
        logger.info(f"   ❌ No significant difference (p >= 0.05)")

    return results


def bootstrap_confidence_intervals(y_true, pred_us, pred_cal, n_bootstrap=1000):
    """
    Compute bootstrap 95% confidence intervals on MAE difference.

    Returns:
        Dictionary with CI bounds
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"BOOTSTRAP CONFIDENCE INTERVALS ({n_bootstrap} iterations)")
    logger.info(f"{'='*80}")

    mae_diffs = []

    for i in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(len(y_true), size=len(y_true), replace=True)

        y_boot = y_true[indices]
        pred_us_boot = pred_us[indices]
        pred_cal_boot = pred_cal[indices]

        mae_us = mean_absolute_error(y_boot, pred_us_boot)
        mae_cal = mean_absolute_error(y_boot, pred_cal_boot)
        mae_diff = mae_cal - mae_us  # Positive = US better

        mae_diffs.append(mae_diff)

        if (i + 1) % 250 == 0:
            logger.info(f"  Completed {i+1}/{n_bootstrap} iterations...")

    mae_diffs = np.array(mae_diffs)
    ci_lower = np.percentile(mae_diffs, 2.5)
    ci_upper = np.percentile(mae_diffs, 97.5)
    mean_diff = np.mean(mae_diffs)

    logger.info(f"\nBootstrap Results:")
    logger.info(f"  Mean MAE difference: {mean_diff:.4f} points (Calibrated - US)")
    logger.info(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}] points")

    if ci_lower > 0:
        logger.info(f"  ✅ US model is SIGNIFICANTLY better (CI excludes zero)")
    else:
        logger.info(f"  ⚠️ Confidence interval includes zero (not significant)")

    return {
        'mean_diff': mean_diff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'mae_diffs': mae_diffs
    }


def generate_comparison_report(us_results, cal_results, stat_tests, bootstrap):
    """Generate comprehensive comparison report."""
    logger.info(f"\n{'='*80}")
    logger.info("FINAL COMPARISON: US-Only vs Calibrated EU Model")
    logger.info(f"{'='*80}")

    logger.info(f"\n{'─'*80}")
    logger.info("MODEL PERFORMANCE (10-Fold Cross-Validation)")
    logger.info(f"{'─'*80}")

    logger.info(f"\nCalibrated EU Model:")
    logger.info(f"  MAE:  {cal_results['cv_mae']:.2f} ± {cal_results['cv_mae_std']:.2f}")
    logger.info(f"  RMSE: {cal_results['cv_rmse']:.2f}")
    logger.info(f"  R²:   {cal_results['cv_r2']:.3f}")

    logger.info(f"\nUS-Only Model:")
    logger.info(f"  MAE:  {us_results['cv_mae']:.2f} ± {us_results['cv_mae_std']:.2f}")
    logger.info(f"  RMSE: {us_results['cv_rmse']:.2f}")
    logger.info(f"  R²:   {us_results['cv_r2']:.3f}")

    # Calculate improvement
    mae_improvement = (cal_results['cv_mae'] - us_results['cv_mae']) / cal_results['cv_mae'] * 100
    r2_improvement = (us_results['cv_r2'] - cal_results['cv_r2']) / cal_results['cv_r2'] * 100

    logger.info(f"\n{'─'*80}")
    logger.info("IMPROVEMENT ANALYSIS")
    logger.info(f"{'─'*80}")
    logger.info(f"  MAE improvement: {mae_improvement:.1f}% ({cal_results['cv_mae']:.2f} → {us_results['cv_mae']:.2f})")
    logger.info(f"  R² improvement:  {r2_improvement:.1f}% ({cal_results['cv_r2']:.3f} → {us_results['cv_r2']:.3f})")
    logger.info(f"  Absolute MAE reduction: {cal_results['cv_mae'] - us_results['cv_mae']:.2f} points")

    logger.info(f"\n{'─'*80}")
    logger.info("STATISTICAL SIGNIFICANCE")
    logger.info(f"{'─'*80}")
    logger.info(f"  Paired t-test:        p = {stat_tests['t_test']['p_value']:.6f}")
    logger.info(f"  Wilcoxon test:        p = {stat_tests['wilcoxon']['p_value']:.6f}")
    logger.info(f"  Permutation test:     p = {stat_tests['permutation']['p_value']:.6f}")
    logger.info(f"  Bootstrap 95% CI:     [{bootstrap['ci_lower']:.4f}, {bootstrap['ci_upper']:.4f}] points")

    # Overall verdict
    logger.info(f"\n{'='*80}")
    logger.info("VERDICT")
    logger.info(f"{'='*80}")

    significant_tests = sum([
        stat_tests['t_test']['p_value'] < 0.05,
        stat_tests['wilcoxon']['p_value'] < 0.05,
        stat_tests['permutation']['p_value'] < 0.05,
        bootstrap['ci_lower'] > 0
    ])

    if significant_tests >= 3:
        logger.info(f"✅ US-only model is STATISTICALLY SIGNIFICANTLY better")
        logger.info(f"   ({significant_tests}/4 tests significant at p < 0.05)")
        logger.info(f"   Improvement: {mae_improvement:.1f}% MAE reduction")
    elif significant_tests >= 2:
        logger.info(f"⚠️ US-only model is PROBABLY better")
        logger.info(f"   ({significant_tests}/4 tests significant)")
        logger.info(f"   Improvement: {mae_improvement:.1f}% MAE reduction")
    else:
        logger.info(f"❌ NO SIGNIFICANT DIFFERENCE between models")
        logger.info(f"   ({significant_tests}/4 tests significant)")
        logger.info(f"   Observed improvement ({mae_improvement:.1f}%) may be chance")

    # Corrected comparison vs original claim
    logger.info(f"\n{'─'*80}")
    logger.info("CORRECTION TO ORIGINAL CLAIM")
    logger.info(f"{'─'*80}")
    logger.info(f"  Original claim:  \"44% better\" (comparing 1.98 in-sample vs 3.54 mixed)")
    logger.info(f"  Corrected claim: \"{mae_improvement:.1f}% better\" (comparing {us_results['cv_mae']:.2f} CV vs {cal_results['cv_mae']:.2f} CV)")
    logger.info(f"  Inflation factor: {44 / mae_improvement:.1f}×")


def main():
    """Run fair model comparison."""
    logger.info("="*80)
    logger.info("FAIR MODEL COMPARISON: US-Only vs Calibrated EU Model")
    logger.info("="*80)
    logger.info("\nFIXING CRITICAL METHODOLOGY FLAW:")
    logger.info("  Original: Compared in-sample (1.98) to mixed (3.54)")
    logger.info("  Corrected: Compare CV (US) to CV (Calibrated) on identical splits")
    logger.info("="*80)

    # Load data
    X, y, ml_preds, metadata = load_us_ground_truth(min_params=6)

    # Run fair cross-validation for both models
    us_results = cross_val_us_only_model(X, y, n_folds=10, random_state=42)
    cal_results = cross_val_calibrated_model(X, y, ml_preds, n_folds=10, random_state=42)

    # Statistical tests
    stat_tests = statistical_significance_tests(y, us_results['cv_predictions'], cal_results['cv_predictions'])

    # Bootstrap CIs
    bootstrap = bootstrap_confidence_intervals(y, us_results['cv_predictions'], cal_results['cv_predictions'], n_bootstrap=1000)

    # Final report
    generate_comparison_report(us_results, cal_results, stat_tests, bootstrap)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = f'data/models/fair_comparison_{timestamp}.json'

    results_data = {
        'timestamp': timestamp,
        'n_samples': len(y),
        'us_only': {
            'cv_mae': float(us_results['cv_mae']),
            'cv_mae_std': float(us_results['cv_mae_std']),
            'cv_rmse': float(us_results['cv_rmse']),
            'cv_r2': float(us_results['cv_r2'])
        },
        'calibrated': {
            'cv_mae': float(cal_results['cv_mae']),
            'cv_mae_std': float(cal_results['cv_mae_std']),
            'cv_rmse': float(cal_results['cv_rmse']),
            'cv_r2': float(cal_results['cv_r2'])
        },
        'statistical_tests': {
            't_test_p': float(stat_tests['t_test']['p_value']),
            'wilcoxon_p': float(stat_tests['wilcoxon']['p_value']),
            'permutation_p': float(stat_tests['permutation']['p_value'])
        },
        'bootstrap': {
            'mean_diff': float(bootstrap['mean_diff']),
            'ci_lower': float(bootstrap['ci_lower']),
            'ci_upper': float(bootstrap['ci_upper'])
        }
    }

    Path('data/models').mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)

    logger.info(f"\n✅ Results saved to: {results_path}")


if __name__ == '__main__':
    main()
