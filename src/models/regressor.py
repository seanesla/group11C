"""
Water Quality Regression Model

Regression model to predict continuous WQI scores (0-100) and analyze trends.
Uses RandomForest and GradientBoosting with hyperparameter tuning.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional, Any
import logging
from pathlib import Path
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    explained_variance_score
)
from scipy import stats

# Dual import to work in both training (src.utils.*) and Streamlit (utils.*) contexts
try:
    from utils.validation_metrics import bootstrap_confidence_interval
except ImportError:
    from src.utils.validation_metrics import bootstrap_confidence_interval

logger = logging.getLogger(__name__)

# Allowed directories for model loading (security measure)
_ALLOWED_MODEL_DIRS = [
    Path(__file__).parent.parent.parent / "data" / "models",
]


def _validate_model_path(filepath: str) -> None:
    """Validate that model path is within allowed directories.

    Prevents loading arbitrary files from untrusted locations,
    mitigating joblib deserialization risks.

    Set WQI_SKIP_PATH_VALIDATION=1 in test environments to bypass.
    """
    if os.getenv("WQI_SKIP_PATH_VALIDATION"):
        return

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


class WQIPredictionRegressor:
    """
    Regression model for WQI score prediction and trend analysis.

    Predicts continuous WQI scores (0-100) based on water quality
    parameters and contextual features. Supports temporal trend analysis.
    """

    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the regressor.

        Args:
            model_type: 'random_forest' or 'gradient_boosting'
        """
        self.model_type = model_type
        self.model = None
        # Impute missing values with median (robust to outliers), then scale
        # KNN imputation was tested but performed poorly with 60-75% missingness
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.feature_names = None
        self.metrics = {}
        self.grid_search = None
        self.calibrator = None  # Optional domain calibration for US predictions

        logger.info(f"Initialized WQIPredictionRegressor with {model_type}")

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'wqi_score',
        exclude_cols: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features and target for training.

        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            exclude_cols: Columns to exclude from features

        Returns:
            Tuple of (X, y, feature_names)
        """
        logger.info("Preparing data for regression")

        # Default exclude list
        if exclude_cols is None:
            exclude_cols = [
                'waterBodyIdentifier',
                'wqi_score',
                'wqi_classification',
                'is_safe',
                'parameter_scores',
                'Country',  # Already one-hot encoded
                'Country_grouped',
                'parameterWaterBodyCategory',  # Already one-hot encoded
                'nitrate_pollution_level',  # Categorical, already processed
                # Missingness indicators are useful for diagnostics but too brittle
                # to serve as primary predictors in production models.
                'ph_missing',
                'dissolved_oxygen_missing',
                'temperature_missing',
                'turbidity_missing',
                'nitrate_missing',
                'conductance_missing',
                'n_params_available',
                # Turbidity is not present as a real-valued feature in the Kaggle
                # training data (the column is entirely NaN in the core feature
                # pipeline). Excluding it keeps the regressor schema aligned with
                # the 18-feature inference schema.
                'turbidity',
            ]

        # Extract target
        y = df[target_col].values

        # Extract features
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols].copy()

        # Handle any remaining categorical variables
        for col in X.columns:
            if X[col].dtype == 'object':
                logger.warning(f"Found object column {col}, dropping it")
                X = X.drop(columns=[col])

        self.feature_names = X.columns.tolist()
        X = X.values

        logger.info(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
        logger.info(f"Target statistics: mean={np.mean(y):.2f}, std={np.std(y):.2f}, range=[{np.min(y):.2f}, {np.max(y):.2f}]")

        return X, y, self.feature_names

    def preprocess_features(
        self,
        X: np.ndarray,
        fit: bool = True
    ) -> np.ndarray:
        """
        Impute missing values and scale features.

        Args:
            X: Feature matrix (numpy array or DataFrame)
            fit: Whether to fit the imputer and scaler (True for training)

        Returns:
            Preprocessed feature matrix
        """
        logger.info(f"Preprocessing features (fit={fit})")

        # CRITICAL: Validate feature order when predicting (not training)
        if not fit and hasattr(X, 'columns') and self.feature_names:
            input_features = list(X.columns)
            expected_features = self.feature_names

            if input_features != expected_features:
                # Check if it's just order mismatch (can fix) vs missing features (error)
                missing = set(expected_features) - set(input_features)
                extra = set(input_features) - set(expected_features)

                if missing:
                    raise ValueError(
                        f"Missing features for prediction: {missing}. "
                        f"Expected {len(expected_features)} features: {expected_features}"
                    )
                if extra:
                    logger.warning(f"Dropping unexpected features: {extra}")
                    X = X[[col for col in input_features if col in expected_features]]

                # Reorder to match expected order
                logger.info("Reordering input features to match training order")
                X = X[expected_features]

        # CRITICAL: Convert DataFrame to numpy array to avoid feature name mismatch
        # Models were trained on numpy arrays without feature names
        if hasattr(X, 'values'):  # Check if X is a DataFrame
            logger.info("Converting DataFrame to numpy array (preserving feature order)")
            X = X.values

        # Verify X is now a numpy array
        if not isinstance(X, np.ndarray):
            raise TypeError(f"Expected numpy array after conversion, got {type(X)}")

        # Preprocess: impute missing values with median, then scale
        if fit:
            X_imputed = self.imputer.fit_transform(X)
            X_processed = self.scaler.fit_transform(X_imputed)
        else:
            X_imputed = self.imputer.transform(X)
            X_processed = self.scaler.transform(X_imputed)

        return X_processed

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42,
        n_jobs: int = -1
    ) -> Dict[str, Any]:
        """
        Train regressor with hyperparameter tuning.

        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion for test set
            val_size: Proportion for validation (from remaining after test split)
            random_state: Random seed
            n_jobs: Number of parallel jobs (-1 = all cores)

        Returns:
            Dictionary with training results and metrics
        """
        logger.info("=" * 80)
        logger.info("Starting regressor training")
        logger.info("=" * 80)

        # Split data: first separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Then split remaining into train and validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
        )

        logger.info(f"Data splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

        # Preprocess features
        X_train_processed = self.preprocess_features(X_train, fit=True)
        X_val_processed = self.preprocess_features(X_val, fit=False)
        X_test_processed = self.preprocess_features(X_test, fit=False)

        # Define hyperparameter grid
        if self.model_type == 'random_forest':
            base_model = RandomForestRegressor(random_state=random_state)
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif self.model_type == 'gradient_boosting':
            base_model = GradientBoostingRegressor(random_state=random_state)
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Grid search with cross-validation
        logger.info(f"Starting GridSearchCV with {len(param_grid)} parameters")
        logger.info(f"Parameter grid: {param_grid}")

        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
        self.grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring='r2',
            n_jobs=n_jobs,
            verbose=1
        )

        # Fit grid search
        logger.info("Fitting GridSearchCV...")
        self.grid_search.fit(X_train_processed, y_train)

        # Best model
        self.model = self.grid_search.best_estimator_
        logger.info(f"Best parameters: {self.grid_search.best_params_}")

        # Extract and store CV fold scores
        best_idx = self.grid_search.best_index_
        cv_scores = self.grid_search.cv_results_['split0_test_score'][best_idx:best_idx+1]
        cv_fold_scores = []
        for split_idx in range(5):  # 5-fold CV
            split_key = f'split{split_idx}_test_score'
            cv_fold_scores.append(self.grid_search.cv_results_[split_key][best_idx])

        cv_mean = float(np.mean(cv_fold_scores))
        cv_std = float(np.std(cv_fold_scores))

        # Store CV metrics
        self.metrics['cv_fold_scores'] = cv_fold_scores
        self.metrics['cv_mean'] = cv_mean
        self.metrics['cv_std'] = cv_std

        logger.info("\n" + "=" * 60)
        logger.info("CROSS-VALIDATION RESULTS (more realistic)")
        logger.info("=" * 60)
        logger.info(f"CV Mean R²:    {cv_mean:.4f}")
        logger.info(f"CV Std R²:     {cv_std:.4f}")
        logger.info(f"CV Fold scores: {[f'{s:.4f}' for s in cv_fold_scores]}")

        # Log target distribution (not class imbalance since this is regression)
        logger.info("\n" + "=" * 60)
        logger.info("TARGET DISTRIBUTION IN TRAINING DATA")
        logger.info("=" * 60)
        logger.info(f"Mean WQI:  {np.mean(y_train):.2f}")
        logger.info(f"Std WQI:   {np.std(y_train):.2f}")
        logger.info(f"Min WQI:   {np.min(y_train):.2f}")
        logger.info(f"Max WQI:   {np.max(y_train):.2f}")
        logger.info(f"Median WQI: {np.median(y_train):.2f}")

        # Check for extreme skewness in distribution
        q1 = np.percentile(y_train, 25)
        q3 = np.percentile(y_train, 75)
        logger.info(f"Q1 (25%):  {q1:.2f}")
        logger.info(f"Q3 (75%):  {q3:.2f}")

        # Evaluate on training set (in-sample)
        train_metrics = self.evaluate(X_train_processed, y_train, dataset_name="Train")

        # Evaluate on validation set
        val_metrics = self.evaluate(X_val_processed, y_val, dataset_name="Validation")

        # Evaluate on test set (with bootstrap CIs)
        test_metrics = self.evaluate(
            X_test_processed, y_test, dataset_name="Test", compute_intervals=True
        )

        # Report both in-sample and CV metrics clearly
        logger.info("\n" + "=" * 80)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"\nIn-Sample (may be optimistic):")
        logger.info(f"  Train R²:      {train_metrics['r2_score']:.4f}")
        logger.info(f"  Train RMSE:    {train_metrics['rmse']:.2f}")
        logger.info(f"\nCross-Validation (more realistic):")
        logger.info(f"  CV Mean R²:    {cv_mean:.4f} ± {cv_std:.4f}")
        logger.info(f"\nHeld-out Sets:")
        logger.info(f"  Val R²:        {val_metrics['r2_score']:.4f}")
        logger.info(f"  Val RMSE:      {val_metrics['rmse']:.2f}")
        logger.info(f"  Test R²:       {test_metrics['r2_score']:.4f}")
        logger.info(f"  Test RMSE:     {test_metrics['rmse']:.2f}")

        # Store results
        results = {
            'best_params': self.grid_search.best_params_,
            'best_cv_score': self.grid_search.best_score_,
            'cv_fold_scores': cv_fold_scores,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'feature_names': self.feature_names,
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'target_distribution': {
                'mean': float(np.mean(y_train)),
                'std': float(np.std(y_train)),
                'min': float(np.min(y_train)),
                'max': float(np.max(y_train)),
                'median': float(np.median(y_train)),
                'q1': float(q1),
                'q3': float(q3)
            }
        }

        logger.info("=" * 80)
        logger.info("Training complete!")
        logger.info("=" * 80)

        return results

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dataset_name: str = "Test",
        compute_intervals: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate regressor with comprehensive metrics.

        Args:
            X: Feature matrix (preprocessed)
            y: True values
            dataset_name: Name for logging

        Returns:
            Dictionary of metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Evaluating on {dataset_name} set ({len(X)} samples)")
        logger.info(f"{'=' * 60}")

        # Predictions
        y_pred = self.model.predict(X)

        # Calculate metrics
        metrics = {
            'r2_score': r2_score(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'explained_variance': explained_variance_score(y, y_pred)
        }

        # Additional statistics
        residuals = y - y_pred
        metrics.update({
            'mean_residual': float(np.mean(residuals)),
            'std_residual': float(np.std(residuals)),
            'min_residual': float(np.min(residuals)),
            'max_residual': float(np.max(residuals))
        })

        # Residual diagnostics
        # Normality test (Shapiro-Wilk, subsample for large n)
        sample_size = min(len(residuals), 5000)
        if sample_size < len(residuals):
            rng = np.random.RandomState(42)
            residual_sample = rng.choice(residuals, sample_size, replace=False)
        else:
            residual_sample = residuals
        _, normality_p = stats.shapiro(residual_sample)
        metrics['residual_normality_p'] = float(normality_p)
        metrics['residuals_normal'] = bool(normality_p > 0.05)

        # Heteroscedasticity check (correlation between |residuals| and predictions)
        het_corr, het_p = stats.spearmanr(np.abs(residuals), y_pred)
        metrics['heteroscedasticity_corr'] = float(het_corr)
        metrics['heteroscedasticity_p'] = float(het_p)
        metrics['heteroscedastic'] = bool(het_p < 0.05 and abs(het_corr) > 0.1)

        # Q-Q plot data (downsampled to 100 points for JSON size)
        n_qq_points = min(100, len(residuals))
        qq_indices = np.linspace(0, len(residuals) - 1, n_qq_points, dtype=int)
        sorted_residuals = np.sort(residuals)
        theoretical_quantiles = stats.norm.ppf(
            np.linspace(0.01, 0.99, n_qq_points)
        )
        metrics['qq_plot_data'] = {
            'theoretical': [float(x) for x in theoretical_quantiles],
            'observed': [float(sorted_residuals[i]) for i in qq_indices]
        }

        # Bootstrap confidence intervals (test set only, expensive)
        if compute_intervals:
            logger.info("Computing bootstrap confidence intervals (n=1000)...")

            for metric_name, metric_fn in [
                ('r2_score', r2_score),
                ('mae', mean_absolute_error),
                ('rmse', lambda yt, yp: np.sqrt(mean_squared_error(yt, yp))),
            ]:
                _, lower, upper = bootstrap_confidence_interval(y, y_pred, metric_fn)
                metrics[f'{metric_name}_ci_lower'] = lower
                metrics[f'{metric_name}_ci_upper'] = upper

        # Log results
        logger.info(f"\nMetrics:")
        logger.info(f"  R² Score:           {metrics['r2_score']:.4f}")
        logger.info(f"  Mean Absolute Error: {metrics['mae']:.4f}")
        logger.info(f"  Mean Squared Error:  {metrics['mse']:.4f}")
        logger.info(f"  Root Mean Squared Error: {metrics['rmse']:.4f}")
        logger.info(f"  Explained Variance:  {metrics['explained_variance']:.4f}")

        logger.info(f"\nResidual Statistics:")
        logger.info(f"  Mean:   {metrics['mean_residual']:.4f}")
        logger.info(f"  Std:    {metrics['std_residual']:.4f}")
        logger.info(f"  Range:  [{metrics['min_residual']:.4f}, {metrics['max_residual']:.4f}]")

        logger.info(f"\nResidual Diagnostics:")
        logger.info(f"  Normality (p={metrics['residual_normality_p']:.4f}): {'Normal' if metrics['residuals_normal'] else 'Non-normal'}")
        logger.info(f"  Heteroscedasticity (corr={metrics['heteroscedasticity_corr']:.3f}): {'Detected' if metrics['heteroscedastic'] else 'Not detected'}")

        if compute_intervals:
            logger.info(f"\n95% Confidence Intervals:")
            logger.info(f"  R² Score: [{metrics['r2_score_ci_lower']:.4f}, {metrics['r2_score_ci_upper']:.4f}]")
            logger.info(f"  MAE:      [{metrics['mae_ci_lower']:.4f}, {metrics['mae_ci_upper']:.4f}]")
            logger.info(f"  RMSE:     [{metrics['rmse_ci_lower']:.4f}, {metrics['rmse_ci_upper']:.4f}]")

        # Prediction quality by WQI range
        logger.info(f"\nPrediction Quality by WQI Range:")
        ranges = [(0, 25), (25, 50), (50, 70), (70, 90), (90, 100)]
        for low, high in ranges:
            mask = (y >= low) & (y < high)
            if np.sum(mask) > 0:
                mae_range = mean_absolute_error(y[mask], y_pred[mask])
                logger.info(f"  WQI [{low:3d}-{high:3d}): {np.sum(mask):4d} samples, MAE={mae_range:.2f}")

        # Store metrics
        self.metrics[dataset_name.lower()] = metrics

        return metrics

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from trained model.

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model does not support feature importance")

        # Get importances
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': [self.feature_names[i] for i in indices[:top_n]],
            'importance': importances[indices[:top_n]]
        })

        logger.info(f"\nTop {top_n} Feature Importances:")
        for idx, row in importance_df.iterrows():
            logger.info(f"  {idx+1:2d}. {row['feature']:40s} {row['importance']:.4f}")

        return importance_df

    def predict(self, X: np.ndarray, apply_calibration: bool = True) -> np.ndarray:
        """
        Predict WQI scores with optional domain calibration.

        Args:
            X: Feature matrix (will be preprocessed)
            apply_calibration: Whether to apply domain calibration if available

        Returns:
            Predicted WQI scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        X_processed = self.preprocess_features(X, fit=False)
        predictions = self.model.predict(X_processed)

        # Apply domain calibration if available (corrects EU → US domain shift)
        if apply_calibration and self.calibrator is not None:
            predictions = self.calibrator.calibrate(predictions)
            logger.debug(f"Applied domain calibration to {len(predictions)} predictions")

        # Clip predictions to valid WQI range [0, 100]
        return np.clip(predictions, 0, 100)

    def predict_trend(
        self,
        X: np.ndarray,
        current_year: int = 2024
    ) -> Dict[str, Any]:
        """
        Predict WQI trends over time.

        Args:
            X: Feature matrix with temporal features
            current_year: Current year for trend projection

        Returns:
            Dictionary with trend analysis
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        # Check if 'year' feature exists
        if 'year' not in self.feature_names:
            logger.warning("'year' feature not found, cannot analyze trends")
            return {'trend': 'unknown', 'message': 'Temporal features not available'}

        year_idx = self.feature_names.index('year')

        # Predict for current and future years
        predictions = {}
        for year_offset in [0, 1, 2, 5]:
            year = current_year + year_offset
            X_future = X.copy()
            X_future[:, year_idx] = year

            pred = self.predict(X_future)
            predictions[year] = float(np.mean(pred))

        # Analyze trend
        current_wqi = predictions[current_year]
        future_wqi = predictions[current_year + 5]
        wqi_change = future_wqi - current_wqi

        if wqi_change > 5:
            trend = 'improving'
        elif wqi_change < -5:
            trend = 'declining'
        else:
            trend = 'stable'

        return {
            'trend': trend,
            'current_wqi': current_wqi,
            'future_wqi': future_wqi,
            'wqi_change': wqi_change,
            'predictions_by_year': predictions
        }

    def predict_future_trend(
        self,
        X: np.ndarray,
        start_date: datetime,
        periods: int = 12,
        freq: str = 'M'
    ) -> Dict[str, Any]:
        """
        Predict WQI trends over future time periods for visualization.

        Args:
            X: Feature matrix with temporal features (single sample or multiple)
            start_date: Starting date for predictions (typically current date)
            periods: Number of future periods to predict (default: 12 months)
            freq: Frequency of predictions - 'M' for monthly, 'Y' for yearly

        Returns:
            Dictionary with:
                - dates: List of datetime objects for each prediction
                - predictions: List of WQI predictions
                - trend: Overall trend direction ('improving', 'stable', 'declining')
                - trend_slope: Rate of change per period
                - current_wqi: Starting WQI prediction
                - final_wqi: Ending WQI prediction
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        # Check if 'year' feature exists
        if 'year' not in self.feature_names:
            logger.warning("'year' feature not found, cannot analyze trends")
            return {
                'trend': 'unknown',
                'message': 'Temporal features not available',
                'dates': [],
                'predictions': []
            }

        year_idx = self.feature_names.index('year')

        # Generate date range based on frequency
        from dateutil.relativedelta import relativedelta

        dates = []
        predictions = []

        for i in range(periods):
            # Calculate future date
            if freq == 'M':
                future_date = start_date + relativedelta(months=i)
            elif freq == 'Y':
                future_date = start_date + relativedelta(years=i)
            else:
                raise ValueError(f"Unsupported frequency: {freq}. Use 'M' or 'Y'.")

            dates.append(future_date)

            # Calculate fractional year for the prediction
            # Use decimal year representation (e.g., 2024.5 for mid-2024)
            year_decimal = future_date.year + (future_date.month - 1) / 12.0

            # Create feature matrix for this time point
            X_future = X.copy()
            X_future[:, year_idx] = year_decimal

            # Make prediction
            pred = self.predict(X_future)
            predictions.append(float(np.mean(pred)))

        # Analyze overall trend
        current_wqi = predictions[0]
        final_wqi = predictions[-1]
        wqi_change = final_wqi - current_wqi

        # Calculate trend slope (change per period)
        trend_slope = wqi_change / periods if periods > 0 else 0

        # Determine trend direction
        if wqi_change > 5:
            trend = 'improving'
        elif wqi_change < -5:
            trend = 'declining'
        else:
            trend = 'stable'

        return {
            'dates': dates,
            'predictions': predictions,
            'trend': trend,
            'trend_slope': trend_slope,
            'current_wqi': current_wqi,
            'final_wqi': final_wqi,
            'wqi_change': wqi_change,
            'periods': periods,
            'frequency': freq
        }

    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save model and preprocessing components.

        Args:
            filepath: Path to save (auto-generated if None)

        Returns:
            Path where model was saved
        """
        if self.model is None:
            raise ValueError("No model to save. Train first.")

        # Generate filepath with timestamp (using absolute path for robustness)
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            models_dir = Path(__file__).parent.parent.parent / "data" / "models"
            filepath = str(models_dir / f"regressor_{timestamp}.joblib")

        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Save everything including CV metrics
        model_data = {
            'model': self.model,
            'imputer': self.imputer,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'metrics': self.metrics,  # Includes cv_fold_scores, cv_mean, cv_std, train/val/test metrics
            'best_params': self.grid_search.best_params_ if self.grid_search else None,
            'timestamp': datetime.now().isoformat()
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

        return filepath

    @classmethod
    def load(cls, filepath: str, load_calibration: bool = True) -> 'WQIPredictionRegressor':
        """
        Load a saved model with optional domain calibration.

        Args:
            filepath: Path to saved model
            load_calibration: Whether to auto-load calibration if available

        Returns:
            Loaded WQIPredictionRegressor instance
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # Security: validate path is within allowed directories
        _validate_model_path(filepath)

        logger.info(f"Loading model from {filepath}")
        model_data = joblib.load(filepath)

        # Create instance
        instance = cls(model_type=model_data['model_type'])
        instance.model = model_data['model']
        # Handle new format (imputer + scaler) and legacy format (preprocessor Pipeline)
        if 'imputer' in model_data and 'scaler' in model_data:
            # New format: separate imputer and scaler
            instance.imputer = model_data['imputer']
            instance.scaler = model_data['scaler']
        elif 'preprocessor' in model_data:
            # Legacy KNN Pipeline format: extract components
            logger.warning("Loading legacy model with KNN Pipeline - extracting components")
            pipeline = model_data['preprocessor']
            instance.scaler = pipeline.named_steps.get('scaler', StandardScaler())
            # Create a new SimpleImputer since we're removing KNN
            instance.imputer = SimpleImputer(strategy='median')
            logger.warning("Replaced KNN imputer with median imputer for legacy model")
        instance.feature_names = model_data['feature_names']
        instance.metrics = model_data['metrics']

        logger.info(f"Model loaded successfully")
        logger.info(f"  Type: {instance.model_type}")
        logger.info(f"  Features: {len(instance.feature_names)}")
        logger.info(f"  Saved: {model_data.get('timestamp', 'unknown')}")

        # Auto-load companion calibrator if exists
        if load_calibration:
            calibrator_path = str(filepath).replace('regressor_', 'calibrator_us_').replace('.joblib', '.joblib')
            if Path(calibrator_path).exists():
                try:
                    from src.models.domain_calibrator import DomainCalibrator
                    instance.calibrator = DomainCalibrator.load(calibrator_path)
                    logger.info(f"✓ Loaded domain calibration for US predictions")
                except Exception as e:
                    logger.warning(f"Failed to load calibration: {e}")
                    instance.calibrator = None
            else:
                logger.info(f"No calibration file found (looked for {Path(calibrator_path).name})")

        return instance


if __name__ == "__main__":
    # Example usage
    from src.preprocessing.feature_engineering import prepare_ml_dataset

    logger.info("=" * 80)
    logger.info("Training WQI Prediction Regressor")
    logger.info("=" * 80)

    # Load data
    logger.info("\nLoading and preparing dataset...")
    df = prepare_ml_dataset()

    # Initialize regressor
    regressor = WQIPredictionRegressor(model_type='random_forest')

    # Prepare data
    X, y, feature_names = regressor.prepare_data(df)

    # Train
    results = regressor.train(X, y)

    # Get feature importance
    importance = regressor.get_feature_importance(top_n=15)

    # Save model
    model_path = regressor.save()

    logger.info(f"\n{'=' * 80}")
    logger.info("TRAINING COMPLETE")
    logger.info(f"{'=' * 80}")
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Test R² Score: {results['test_metrics']['r2_score']:.4f}")
    logger.info(f"Test RMSE: {results['test_metrics']['rmse']:.4f}")
