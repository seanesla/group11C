#!/usr/bin/env python3
"""
Train Water Quality ML Models

This script trains both the classifier and regressor models on the
Kaggle Water Quality dataset.

Usage:
    # Standard training (full feature set):
    poetry run python train_models.py

    # Core parameters only (universal water quality features):
    poetry run python train_models.py --core-params-only
"""

import sys
import logging
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.preprocessing.feature_engineering import prepare_ml_dataset
from src.models.model_utils import train_and_save_models
from src.utils.logging_config import configure_logging

# Configure logging once at startup
configure_logging()
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train water quality ML models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with full feature set (59 features):
  python train_models.py

  # Train with core parameters only (24 features):
  python train_models.py --core-params-only
        """
    )

    parser.add_argument(
        '--core-params-only',
        action='store_true',
        help='Train using only core water quality features (~24 features) instead of '
             'full feature set (59 features). This excludes dataset-specific context features '
             '(geographic, environmental, economic, waste management) to enable better '
             'generalization beyond the original training regions.'
    )

    return parser.parse_args()


def main():
    """Main training pipeline"""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("WATER QUALITY ML MODEL TRAINING")
    if args.core_params_only:
        logger.info("MODE: CORE PARAMETERS ONLY (universal water quality features)")
    else:
        logger.info("MODE: FULL FEATURE SET (includes dataset-specific context features)")
    logger.info("=" * 80)

    try:
        # Step 1: Prepare dataset
        logger.info("\nStep 1: Preparing ML dataset...")
        df = prepare_ml_dataset(
            save_processed=True,
            core_params_only=args.core_params_only
        )
        logger.info(f"Dataset prepared: {df.shape[0]} samples, {df.shape[1]} features")

        # Step 2: Train both models
        logger.info("\nStep 2: Training models...")
        results = train_and_save_models(
            df,
            classifier_type='random_forest',
            regressor_type='random_forest',
            n_jobs=-1  # Use all CPU cores
        )

        # Step 3: Summary
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE - SUMMARY")
        logger.info("=" * 80)

        clf_results = results['classifier']['results']
        reg_results = results['regressor']['results']

        clf_train = clf_results['train_metrics']
        clf_val = clf_results['val_metrics']
        clf_test = clf_results['test_metrics']
        clf_cv_mean = clf_results['cv_mean']
        clf_cv_std = clf_results['cv_std']

        reg_train = reg_results['train_metrics']
        reg_val = reg_results['val_metrics']
        reg_test = reg_results['test_metrics']
        reg_cv_mean = reg_results['cv_mean']
        reg_cv_std = reg_results['cv_std']

        logger.info(f"\nClassifier Performance:")
        logger.info(f"  In-Sample (may be optimistic):")
        logger.info(f"    Train F1:       {clf_train['f1_score']:.4f}")
        logger.info(f"    Train Accuracy: {clf_train['accuracy']:.4f}")
        logger.info(f"  Cross-Validation (more realistic):")
        logger.info(f"    CV Mean F1:     {clf_cv_mean:.4f} ± {clf_cv_std:.4f}")
        logger.info(f"  Held-out Sets:")
        logger.info(f"    Val F1:         {clf_val['f1_score']:.4f}")
        logger.info(f"    Val Accuracy:   {clf_val['accuracy']:.4f}")
        logger.info(f"    Test F1:        {clf_test['f1_score']:.4f}")
        logger.info(f"    Test Accuracy:  {clf_test['accuracy']:.4f}")
        logger.info(f"    Test Precision: {clf_test['precision']:.4f}")
        logger.info(f"    Test Recall:    {clf_test['recall']:.4f}")
        logger.info(f"    Test ROC-AUC:   {clf_test['roc_auc']:.4f}")

        logger.info(f"\nRegressor Performance:")
        logger.info(f"  In-Sample (may be optimistic):")
        logger.info(f"    Train R²:       {reg_train['r2_score']:.4f}")
        logger.info(f"    Train RMSE:     {reg_train['rmse']:.2f}")
        logger.info(f"  Cross-Validation (more realistic):")
        logger.info(f"    CV Mean R²:     {reg_cv_mean:.4f} ± {reg_cv_std:.4f}")
        logger.info(f"  Held-out Sets:")
        logger.info(f"    Val R²:         {reg_val['r2_score']:.4f}")
        logger.info(f"    Val RMSE:       {reg_val['rmse']:.2f}")
        logger.info(f"    Test R²:        {reg_test['r2_score']:.4f}")
        logger.info(f"    Test RMSE:      {reg_test['rmse']:.2f}")
        logger.info(f"    Test MAE:       {reg_test['mae']:.2f}")

        logger.info(f"\nModels saved:")
        logger.info(f"  Classifier: {results['classifier']['path']}")
        logger.info(f"  Regressor:  {results['regressor']['path']}")
        logger.info(f"  Metadata:   {results['metadata_path']}")

        # Check if models meet success criteria
        logger.info("\n" + "=" * 80)
        logger.info("SUCCESS CRITERIA CHECK")
        logger.info("=" * 80)
        logger.info("Using Cross-Validation metrics (more realistic than test-only)")
        logger.info("")

        success = True

        # Classifier criteria: CV F1 > 0.70
        if clf_cv_mean >= 0.70:
            logger.info(f"✓ Classifier CV F1 >= 0.70 (actual: {clf_cv_mean:.4f})")
        else:
            logger.warning(f"✗ Classifier CV F1 {clf_cv_mean:.4f} < 0.70")
            success = False

        # Also check test accuracy for reference
        if clf_test['accuracy'] >= 0.75:
            logger.info(f"✓ Classifier test accuracy >= 75% (actual: {clf_test['accuracy']:.2%})")
        else:
            logger.warning(f"⚠ Classifier test accuracy {clf_test['accuracy']:.2%} < 75%")

        # Regressor criteria: CV R² > 0.6
        if reg_cv_mean >= 0.6:
            logger.info(f"✓ Regressor CV R² >= 0.6 (actual: {reg_cv_mean:.4f})")
        else:
            logger.warning(f"✗ Regressor CV R² {reg_cv_mean:.4f} < 0.6")
            success = False

        # Also check test R² for reference
        if reg_test['r2_score'] >= 0.6:
            logger.info(f"✓ Regressor test R² >= 0.6 (actual: {reg_test['r2_score']:.4f})")
        else:
            logger.warning(f"⚠ Regressor test R² {reg_test['r2_score']:.4f} < 0.6")

        if success:
            logger.info("\n✓✓✓ ALL SUCCESS CRITERIA MET ✓✓✓")
        else:
            logger.warning("\n⚠ Some success criteria not met - models may need tuning")

        logger.info("\n" + "=" * 80)
        logger.info("Training complete! Models ready for use in Streamlit app.")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
