"""
Utility functions for model management and loading.
"""

import glob
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging
import json
import joblib
from datetime import datetime

from .classifier import WaterQualityClassifier
from .regressor import WQIPredictionRegressor

logger = logging.getLogger(__name__)

# Project root and default paths (computed from file location for robustness)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_MODELS_DIR = str(PROJECT_ROOT / "data" / "models")


def get_latest_model_path(model_type: str = 'classifier', models_dir: str = None) -> Optional[str]:
    """
    Find the most recent model file by timestamp.

    Args:
        model_type: 'classifier' or 'regressor'
        models_dir: Directory containing model files

    Returns:
        Path to latest model file, or None if not found
    """
    if models_dir is None:
        models_dir = DEFAULT_MODELS_DIR
    pattern = f"{models_dir}/{model_type}_*.joblib"
    model_files = glob.glob(pattern)

    if not model_files:
        logger.warning(f"No {model_type} models found in {models_dir}")
        return None

    # Sort by timestamp in filename (format: model_YYYYMMDD_HHMMSS.joblib)
    latest = sorted(model_files, reverse=True)[0]
    logger.info(f"Found latest {model_type}: {latest}")

    return latest


def load_latest_models(models_dir: str = None) -> Tuple[Optional[WaterQualityClassifier], Optional[WQIPredictionRegressor]]:
    """
    Load the most recent classifier and regressor models.

    Args:
        models_dir: Directory containing model files (default: data/models relative to project root)

    Returns:
        Tuple of (classifier, regressor). Either may be None if not found.
    """
    if models_dir is None:
        models_dir = DEFAULT_MODELS_DIR
    logger.info("Loading latest ML models...")

    # Load classifier
    classifier_path = get_latest_model_path('classifier', models_dir)
    classifier = None
    if classifier_path:
        try:
            classifier = WaterQualityClassifier.load(classifier_path)
            logger.info("✓ Classifier loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load classifier: {e}")

    # Load regressor
    regressor_path = get_latest_model_path('regressor', models_dir)
    regressor = None
    if regressor_path:
        try:
            regressor = WQIPredictionRegressor.load(regressor_path)
            logger.info("✓ Regressor loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load regressor: {e}")

    if classifier is None and regressor is None:
        logger.warning("No models found. Please train models first.")

    return classifier, regressor


def save_model_metadata(
    classifier_path: Optional[str],
    regressor_path: Optional[str],
    classifier_metrics: Optional[Dict],
    regressor_metrics: Optional[Dict],
    metadata_dir: str = None,
    n_samples: int = None,
    n_features: int = None,
    feature_names: list = None
) -> str:
    """
    Save metadata about trained models.

    Args:
        classifier_path: Path to saved classifier
        regressor_path: Path to saved regressor
        classifier_metrics: Classifier metrics
        regressor_metrics: Regressor metrics
        metadata_dir: Directory to save metadata (default: data/models relative to project root)
        n_samples: Number of samples used for training
        n_features: Number of features used for training
        feature_names: List of feature names used for training

    Returns:
        Path to saved metadata file
    """
    if metadata_dir is None:
        metadata_dir = DEFAULT_MODELS_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata_path = f"{metadata_dir}/metadata_{timestamp}.json"

    metadata = {
        'timestamp': datetime.now().isoformat(),
        'classifier': {
            'path': classifier_path,
            'metrics': classifier_metrics
        },
        'regressor': {
            'path': regressor_path,
            'metrics': regressor_metrics
        },
        'training_info': {
            'dataset': 'Kaggle Water Quality Dataset (1991-2017, non‑US monitoring sites)',
            'samples': n_samples if n_samples is not None else 2939,
            'features': n_features if n_features is not None else 69,
            'feature_names': feature_names,
            'target_classifier': 'is_safe (WQI >= 70)',
            'target_regressor': 'wqi_score (0-100)'
        }
    }

    Path(metadata_dir).mkdir(parents=True, exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Metadata saved to {metadata_path}")
    return metadata_path


def train_and_save_models(
    df,
    classifier_type: str = 'random_forest',
    regressor_type: str = 'random_forest',
    n_jobs: int = -1
) -> Dict[str, Any]:
    """
    Train both models and save them with metadata.

    Args:
        df: Prepared ML dataset
        classifier_type: Type of classifier
        regressor_type: Type of regressor
        n_jobs: Number of parallel jobs

    Returns:
        Dictionary with paths and results
    """
    logger.info("=" * 80)
    logger.info("TRAINING BOTH MODELS")
    logger.info("=" * 80)

    results = {}

    # Train classifier
    logger.info("\n" + "=" * 80)
    logger.info("1. TRAINING CLASSIFIER")
    logger.info("=" * 80)

    classifier = WaterQualityClassifier(model_type=classifier_type)
    X_clf, y_clf, features_clf = classifier.prepare_data(df)
    clf_results = classifier.train(X_clf, y_clf, n_jobs=n_jobs)
    clf_importance = classifier.get_feature_importance(top_n=15)
    classifier_path = classifier.save()

    results['classifier'] = {
        'path': classifier_path,
        'results': clf_results,
        'importance': clf_importance
    }

    # Train regressor
    logger.info("\n" + "=" * 80)
    logger.info("2. TRAINING REGRESSOR")
    logger.info("=" * 80)

    regressor = WQIPredictionRegressor(model_type=regressor_type)
    X_reg, y_reg, features_reg = regressor.prepare_data(df)
    reg_results = regressor.train(X_reg, y_reg, n_jobs=n_jobs)
    reg_importance = regressor.get_feature_importance(top_n=15)
    regressor_path = regressor.save()

    results['regressor'] = {
        'path': regressor_path,
        'results': reg_results,
        'importance': reg_importance
    }

    # Save metadata with actual training info
    n_samples = X_clf.shape[0]
    n_features = X_clf.shape[1]
    metadata_path = save_model_metadata(
        classifier_path,
        regressor_path,
        clf_results['test_metrics'],
        reg_results['test_metrics'],
        n_samples=n_samples,
        n_features=n_features,
        feature_names=list(features_clf)
    )

    results['metadata_path'] = metadata_path

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"\nClassifier:")
    logger.info(f"  Path: {classifier_path}")
    logger.info(f"  Test Accuracy: {clf_results['test_metrics']['accuracy']:.4f}")
    logger.info(f"  Test F1 Score: {clf_results['test_metrics']['f1_score']:.4f}")

    logger.info(f"\nRegressor:")
    logger.info(f"  Path: {regressor_path}")
    logger.info(f"  Test R² Score: {reg_results['test_metrics']['r2_score']:.4f}")
    logger.info(f"  Test RMSE: {reg_results['test_metrics']['rmse']:.4f}")

    logger.info(f"\nMetadata: {metadata_path}")

    return results


if __name__ == "__main__":
    # Test loading
    classifier, regressor = load_latest_models()

    if classifier:
        print(f"\nClassifier loaded:")
        print(f"  Features: {len(classifier.feature_names)}")
        print(f"  Metrics: {classifier.metrics}")

    if regressor:
        print(f"\nRegressor loaded:")
        print(f"  Features: {len(regressor.feature_names)}")
        print(f"  Metrics: {regressor.metrics}")
