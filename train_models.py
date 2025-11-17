#!/usr/bin/env python3
"""
Train Water Quality ML Models

This script trains both the classifier and regressor models on the
Kaggle European water quality dataset.

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
             'full feature set (59 features). This excludes European-specific features '
             '(geographic, environmental, economic, waste management) to enable better '
             'generalization to non-European water quality data.'
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
        logger.info("MODE: FULL FEATURE SET (includes European-specific features)")
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

        clf_metrics = results['classifier']['results']['test_metrics']
        reg_metrics = results['regressor']['results']['test_metrics']

        logger.info(f"\nClassifier Performance:")
        logger.info(f"  Accuracy:  {clf_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {clf_metrics['precision']:.4f}")
        logger.info(f"  Recall:    {clf_metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {clf_metrics['f1_score']:.4f}")
        logger.info(f"  ROC-AUC:   {clf_metrics['roc_auc']:.4f}")

        logger.info(f"\nRegressor Performance:")
        logger.info(f"  R² Score: {reg_metrics['r2_score']:.4f}")
        logger.info(f"  MAE:      {reg_metrics['mae']:.2f}")
        logger.info(f"  RMSE:     {reg_metrics['rmse']:.2f}")

        logger.info(f"\nModels saved:")
        logger.info(f"  Classifier: {results['classifier']['path']}")
        logger.info(f"  Regressor:  {results['regressor']['path']}")
        logger.info(f"  Metadata:   {results['metadata_path']}")

        # Check if models meet success criteria
        logger.info("\n" + "=" * 80)
        logger.info("SUCCESS CRITERIA CHECK")
        logger.info("=" * 80)

        success = True

        # Classifier criteria: >75% accuracy
        if clf_metrics['accuracy'] >= 0.75:
            logger.info("✓ Classifier accuracy >= 75%")
        else:
            logger.warning(f"✗ Classifier accuracy {clf_metrics['accuracy']:.2%} < 75%")
            success = False

        # Regressor criteria: R² > 0.6
        if reg_metrics['r2_score'] >= 0.6:
            logger.info("✓ Regressor R² >= 0.6")
        else:
            logger.warning(f"✗ Regressor R² {reg_metrics['r2_score']:.4f} < 0.6")
            success = False

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
