"""
Domain Calibration for Water Quality Models

This module provides calibration functionality to correct systematic prediction bias
when applying models trained on one data distribution (e.g., Kaggle water quality
training data) to a different distribution (e.g., US water quality).

Uses isotonic regression to learn a monotonic mapping from biased predictions to
calibrated predictions based on ground truth samples from the target domain.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import joblib
from datetime import datetime
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Allowed directories for model loading (security measure)
_ALLOWED_MODEL_DIRS = [
    Path(__file__).parent.parent.parent / "data" / "models",
]


def _validate_model_path(filepath: str) -> None:
    """Validate that model path is within allowed directories.

    Prevents loading arbitrary files from untrusted locations,
    mitigating joblib deserialization risks.
    """
    filepath_resolved = Path(filepath).resolve()
    for allowed_dir in _ALLOWED_MODEL_DIRS:
        try:
            if filepath_resolved.is_relative_to(allowed_dir.resolve()):
                return
        except ValueError:
            continue
    raise ValueError(
        f"Model path '{filepath}' is outside allowed directories. "
        f"Models must be in: {[str(d) for d in _ALLOWED_MODEL_DIRS]}"
    )


class DomainCalibrator:
    """
    Calibrates ML model predictions to correct for domain shift.

    Problem: Models trained on the Kaggle training distribution (DO mean 1.67 mg/L, WQI 66.73)
    systematically under-predict US water quality (DO mean 8.82 mg/L, WQI 86.53)
    by ~20 points.

    Solution: Learn monotonic correction mapping using isotonic regression on US
    ground truth samples.

    Example:
        >>> calibrator = DomainCalibrator()
        >>> calibrator.fit(ml_predictions, actual_us_wqi)
        >>> corrected = calibrator.calibrate(new_predictions)
    """

    def __init__(self, increasing: bool = True, out_of_bounds: str = 'clip'):
        """
        Initialize domain calibrator.

        Args:
            increasing: Whether calibration function is monotonically increasing
            out_of_bounds: How to handle predictions outside training range:
                - 'clip': Clip to min/max of training range (safe, recommended)
                - 'extrapolate': Linear extrapolation (risky)
        """
        self.calibrator = IsotonicRegression(
            increasing=increasing,
            out_of_bounds=out_of_bounds
        )
        self.is_fitted = False
        self.training_stats = {}
        self.validation_stats = {}

        logger.info(f"Initialized DomainCalibrator (increasing={increasing}, out_of_bounds={out_of_bounds})")

    def fit(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        validation_split: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Fit calibration curve on ground truth samples.

        Args:
            y_pred: ML model predictions (biased)
            y_true: Actual ground truth values
            validation_split: Fraction of data to hold out for validation
            random_state: Random seed for reproducibility

        Returns:
            Dictionary with training and validation metrics
        """
        logger.info("=" * 80)
        logger.info("Fitting domain calibration")
        logger.info("=" * 80)

        # Convert to numpy arrays
        y_pred = np.asarray(y_pred).flatten()
        y_true = np.asarray(y_true).flatten()

        if len(y_pred) != len(y_true):
            raise ValueError(f"Prediction and ground truth arrays must have same length: {len(y_pred)} vs {len(y_true)}")

        logger.info(f"Total samples: {len(y_pred)}")
        logger.info(f"Prediction range: [{np.min(y_pred):.2f}, {np.max(y_pred):.2f}]")
        logger.info(f"Ground truth range: [{np.min(y_true):.2f}, {np.max(y_true):.2f}]")

        # Split train/validation
        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                y_pred, y_true,
                test_size=validation_split,
                random_state=random_state
            )
            logger.info(f"Split: {len(X_train)} train, {len(X_val)} validation")
        else:
            X_train, y_train = y_pred, y_true
            X_val, y_val = None, None

        # Fit isotonic regression
        logger.info("Fitting isotonic regression...")
        self.calibrator.fit(X_train, y_train)
        self.is_fitted = True

        # Training metrics
        y_train_calibrated = self.calibrator.predict(X_train)
        self.training_stats = {
            'n_samples': len(X_train),
            'mae_before': float(mean_absolute_error(y_train, X_train)),
            'mae_after': float(mean_absolute_error(y_train, y_train_calibrated)),
            'rmse_before': float(np.sqrt(mean_squared_error(y_train, X_train))),
            'rmse_after': float(np.sqrt(mean_squared_error(y_train, y_train_calibrated))),
            'r2_before': float(r2_score(y_train, X_train)),
            'r2_after': float(r2_score(y_train, y_train_calibrated))
        }

        logger.info("\nTraining Set Performance:")
        logger.info(f"  MAE:  {self.training_stats['mae_before']:.2f} → {self.training_stats['mae_after']:.2f} ({self.training_stats['mae_after']/self.training_stats['mae_before']*100:.1f}% of original)")
        logger.info(f"  RMSE: {self.training_stats['rmse_before']:.2f} → {self.training_stats['rmse_after']:.2f}")
        logger.info(f"  R²:   {self.training_stats['r2_before']:.4f} → {self.training_stats['r2_after']:.4f}")

        # Validation metrics
        if X_val is not None:
            y_val_calibrated = self.calibrator.predict(X_val)
            self.validation_stats = {
                'n_samples': len(X_val),
                'mae_before': float(mean_absolute_error(y_val, X_val)),
                'mae_after': float(mean_absolute_error(y_val, y_val_calibrated)),
                'rmse_before': float(np.sqrt(mean_squared_error(y_val, X_val))),
                'rmse_after': float(np.sqrt(mean_squared_error(y_val, y_val_calibrated))),
                'r2_before': float(r2_score(y_val, X_val)),
                'r2_after': float(r2_score(y_val, y_val_calibrated))
            }

            logger.info("\nValidation Set Performance:")
            logger.info(f"  MAE:  {self.validation_stats['mae_before']:.2f} → {self.validation_stats['mae_after']:.2f} ({self.validation_stats['mae_after']/self.validation_stats['mae_before']*100:.1f}% of original)")
            logger.info(f"  RMSE: {self.validation_stats['rmse_before']:.2f} → {self.validation_stats['rmse_after']:.2f}")
            logger.info(f"  R²:   {self.validation_stats['r2_before']:.4f} → {self.validation_stats['r2_after']:.4f}")

            # Check for overfitting
            mae_ratio = self.validation_stats['mae_after'] / self.training_stats['mae_after']
            if mae_ratio > 1.5:
                logger.warning(f"⚠ Possible overfitting: validation MAE is {mae_ratio:.1f}x training MAE")
            else:
                logger.info(f"✓ Good generalization: validation MAE is {mae_ratio:.2f}x training MAE")

        # Check monotonicity
        if not self._check_monotonicity(X_train, y_train_calibrated):
            logger.warning("⚠ Calibration function is not strictly monotonic")
        else:
            logger.info("✓ Calibration function is monotonically increasing")

        logger.info("=" * 80)
        logger.info("Calibration training complete!")
        logger.info("=" * 80)

        return {
            'training': self.training_stats,
            'validation': self.validation_stats if X_val is not None else None
        }

    def calibrate(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Apply calibration to predictions.

        Args:
            y_pred: Uncalibrated predictions from ML model

        Returns:
            Calibrated predictions
        """
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        y_pred = np.asarray(y_pred).flatten()
        y_calibrated = self.calibrator.predict(y_pred)

        return y_calibrated

    def _check_monotonicity(self, x: np.ndarray, y: np.ndarray) -> bool:
        """Check if calibration function is monotonically increasing."""
        sorted_indices = np.argsort(x)
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]

        # Check if y increases (or stays same) as x increases
        return np.all(np.diff(y_sorted) >= -1e-10)  # Allow tiny numerical errors

    def plot_calibration_curve(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Plot calibration curve showing before/after correction.

        Args:
            y_pred: Uncalibrated predictions
            y_true: Ground truth values
            save_path: Path to save plot (optional)
            show: Whether to display plot
        """
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        y_pred = np.asarray(y_pred).flatten()
        y_true = np.asarray(y_true).flatten()
        y_calibrated = self.calibrate(y_pred)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Before calibration
        ax1.scatter(y_pred, y_true, alpha=0.5, s=50, label='Data points')
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                 'r--', lw=2, label='Perfect prediction')
        ax1.set_xlabel('ML Prediction (Uncalibrated)', fontsize=12)
        ax1.set_ylabel('Actual WQI', fontsize=12)
        ax1.set_title(f'Before Calibration\nMAE = {mean_absolute_error(y_true, y_pred):.2f}', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # After calibration
        ax2.scatter(y_calibrated, y_true, alpha=0.5, s=50, label='Data points', color='green')
        ax2.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                 'r--', lw=2, label='Perfect prediction')
        ax2.set_xlabel('ML Prediction (Calibrated)', fontsize=12)
        ax2.set_ylabel('Actual WQI', fontsize=12)
        ax2.set_title(f'After Calibration\nMAE = {mean_absolute_error(y_true, y_calibrated):.2f}', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Calibration plot saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save calibrator to disk.

        Args:
            filepath: Path to save (auto-generated if None)

        Returns:
            Path where calibrator was saved
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted calibrator")

        # Generate filepath with timestamp (using absolute path for robustness)
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            models_dir = Path(__file__).parent.parent.parent / "data" / "models"
            filepath = str(models_dir / f"calibrator_us_{timestamp}.joblib")

        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Save calibrator and metadata
        calibrator_data = {
            'calibrator': self.calibrator,
            'training_stats': self.training_stats,
            'validation_stats': self.validation_stats,
            'timestamp': datetime.now().isoformat()
        }

        joblib.dump(calibrator_data, filepath)
        logger.info(f"Calibrator saved to {filepath}")

        return filepath

    @classmethod
    def load(cls, filepath: str) -> 'DomainCalibrator':
        """
        Load saved calibrator from disk.

        Args:
            filepath: Path to saved calibrator

        Returns:
            Loaded DomainCalibrator instance
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Calibrator file not found: {filepath}")

        # Security: validate path is within allowed directories
        _validate_model_path(filepath)

        logger.info(f"Loading calibrator from {filepath}")
        calibrator_data = joblib.load(filepath)

        # Create instance
        instance = cls()
        instance.calibrator = calibrator_data['calibrator']
        instance.is_fitted = True
        instance.training_stats = calibrator_data['training_stats']
        instance.validation_stats = calibrator_data['validation_stats']

        logger.info(f"Calibrator loaded successfully")
        logger.info(f"  Training MAE: {instance.training_stats['mae_before']:.2f} → {instance.training_stats['mae_after']:.2f}")
        if instance.validation_stats:
            logger.info(f"  Validation MAE: {instance.validation_stats['mae_before']:.2f} → {instance.validation_stats['mae_after']:.2f}")
        logger.info(f"  Saved: {calibrator_data.get('timestamp', 'unknown')}")

        return instance


if __name__ == "__main__":
    # Example usage
    logger.info("=" * 80)
    logger.info("Domain Calibrator Demo")
    logger.info("=" * 80)

    # Simulate domain shift: European model predictions vs US ground truth
    np.random.seed(42)
    n_samples = 100

    # European model predictions (biased low by ~20 points)
    y_pred_eu = np.random.normal(70, 10, n_samples)  # Mean 70

    # US ground truth (actually higher)
    y_true_us = y_pred_eu + 20 + np.random.normal(0, 3, n_samples)  # Mean 90

    # Fit calibrator
    calibrator = DomainCalibrator()
    metrics = calibrator.fit(y_pred_eu, y_true_us, validation_split=0.2)

    # Test on new data
    y_test_pred = np.array([65, 70, 75, 80, 85])
    y_test_calibrated = calibrator.calibrate(y_test_pred)

    print("\nTest predictions:")
    for pred, calib in zip(y_test_pred, y_test_calibrated):
        print(f"  {pred:.1f} → {calib:.1f} (correction: +{calib-pred:.1f})")

    # Save
    save_path = calibrator.save()
    print(f"\nCalibrator saved to: {save_path}")
