"""
Water Quality Classification Model

Binary classifier to predict safe/unsafe water quality (WQI >= 70 threshold).
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

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight

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


class WaterQualityClassifier:
    """
    Binary classifier for water quality safety prediction.

    Predicts whether water is safe (WQI >= 70) or unsafe (WQI < 70)
    based on water quality parameters and contextual features.
    """

    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the classifier.

        Args:
            model_type: 'random_forest' or 'gradient_boosting'
        """
        self.model_type = model_type
        self.model = None
        # Single preprocessor pipeline: scale first (so KNN distances are meaningful),
        # then impute using KNN on scaled data. Result is already scaled.
        # This replaces separate scaler + SimpleImputer to avoid double scaling.
        self.preprocessor = Pipeline([
            ('scaler', StandardScaler()),
            ('knn_imputer', KNNImputer(n_neighbors=5, weights='distance')),
        ])
        self.feature_names = None
        self.metrics = {}
        self.grid_search = None

        logger.info(f"Initialized WaterQualityClassifier with {model_type}")

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'is_safe',
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
        logger.info("Preparing data for classification")

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
                # Turbidity is not available in the Kaggle training data and is
                # always NaN in the core feature pipeline. Exclude it here so the
                # trained model schema matches the 18-feature inference schema
                # produced by prepare_us_features_for_prediction().
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
        logger.info(f"Target distribution: Safe={np.sum(y)}, Unsafe={len(y)-np.sum(y)}")

        # Check for class imbalance
        safe_pct = np.mean(y) * 100
        logger.info(f"Class balance: {safe_pct:.1f}% safe, {100-safe_pct:.1f}% unsafe")

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

        # Preprocess: scale then impute using KNN (single pipeline, already scaled)
        if fit:
            X_processed = self.preprocessor.fit_transform(X)
        else:
            X_processed = self.preprocessor.transform(X)

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
        Train classifier with hyperparameter tuning.

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
        logger.info("Starting classifier training")
        logger.info("=" * 80)

        # Split data: first separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Then split remaining into train and validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted,
            random_state=random_state, stratify=y_temp
        )

        logger.info(f"Data splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

        # Preprocess features
        X_train_processed = self.preprocess_features(X_train, fit=True)
        X_val_processed = self.preprocess_features(X_val, fit=False)
        X_test_processed = self.preprocess_features(X_test, fit=False)

        # Define hyperparameter grid
        if self.model_type == 'random_forest':
            # Always use class_weight='balanced' to handle severe class imbalance (98.8% SAFE)
            # This is hardcoded on the base model, not tuned via grid search
            base_model = RandomForestClassifier(
                random_state=random_state,
                class_weight='balanced',
                n_jobs=-1
            )
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                # class_weight REMOVED from grid - always 'balanced' from base_model
            }
        elif self.model_type == 'gradient_boosting':
            base_model = GradientBoostingClassifier(random_state=random_state)
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

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        self.grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring='f1',
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
        logger.info(f"CV Mean F1:    {cv_mean:.4f}")
        logger.info(f"CV Std F1:     {cv_std:.4f}")
        logger.info(f"CV Fold scores: {[f'{s:.4f}' for s in cv_fold_scores]}")

        # Log class imbalance in training data
        y_train_safe_pct = np.mean(y_train) * 100
        y_train_unsafe_pct = 100 - y_train_safe_pct
        logger.info("\n" + "=" * 60)
        logger.info("CLASS DISTRIBUTION IN TRAINING DATA")
        logger.info("=" * 60)
        logger.info(f"Safe:   {np.sum(y_train):5d} samples ({y_train_safe_pct:5.1f}%)")
        logger.info(f"Unsafe: {len(y_train) - np.sum(y_train):5d} samples ({y_train_unsafe_pct:5.1f}%)")
        if y_train_safe_pct > 90 or y_train_unsafe_pct > 90:
            logger.warning(f"⚠ SEVERE CLASS IMBALANCE DETECTED (>90% one class)")
            logger.warning(f"  Using class_weight='balanced' to mitigate")

        # Compute and log effective balanced weights
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        weight_dict = dict(zip(classes, weights))
        logger.info(f"Effective class weights (balanced): {weight_dict}")

        # Evaluate on training set (in-sample)
        train_metrics = self.evaluate(X_train_processed, y_train, dataset_name="Train")

        # Evaluate on validation set
        val_metrics = self.evaluate(X_val_processed, y_val, dataset_name="Validation")

        # Evaluate on test set
        test_metrics = self.evaluate(X_test_processed, y_test, dataset_name="Test")

        # Report both in-sample and CV metrics clearly
        logger.info("\n" + "=" * 80)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"\nIn-Sample (may be optimistic):")
        logger.info(f"  Train F1:      {train_metrics['f1_score']:.4f}")
        logger.info(f"  Train Accuracy: {train_metrics['accuracy']:.4f}")
        logger.info(f"\nCross-Validation (more realistic):")
        logger.info(f"  CV Mean F1:    {cv_mean:.4f} ± {cv_std:.4f}")
        logger.info(f"\nHeld-out Sets:")
        logger.info(f"  Val F1:        {val_metrics['f1_score']:.4f}")
        logger.info(f"  Test F1:       {test_metrics['f1_score']:.4f}")

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
            'class_distribution': {
                'safe_count': int(np.sum(y_train)),
                'unsafe_count': int(len(y_train) - np.sum(y_train)),
                'safe_pct': float(y_train_safe_pct),
                'unsafe_pct': float(y_train_unsafe_pct)
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
        dataset_name: str = "Test"
    ) -> Dict[str, float]:
        """
        Evaluate classifier with comprehensive metrics.

        Args:
            X: Feature matrix (preprocessed)
            y: True labels
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
        y_pred_proba = self.model.predict_proba(X)[:, 1]

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1_score': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_pred_proba)
        }

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics.update({
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        })

        # Log results
        logger.info(f"\nMetrics:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")
        logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

        logger.info(f"\nConfusion Matrix:")
        logger.info(f"                 Predicted")
        logger.info(f"                 Unsafe  Safe")
        logger.info(f"  Actual Unsafe  {tn:5d}  {fp:5d}")
        logger.info(f"         Safe    {fn:5d}  {tp:5d}")

        # Classification report
        logger.info(f"\nClassification Report:")
        logger.info("\n" + classification_report(y, y_pred, target_names=['Unsafe', 'Safe']))

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

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Feature matrix (will be preprocessed)

        Returns:
            Predicted labels (0=Unsafe, 1=Safe)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        X_processed = self.preprocess_features(X, fit=False)
        return self.model.predict(X_processed)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix (will be preprocessed)

        Returns:
            Predicted probabilities [P(Unsafe), P(Safe)]
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        X_processed = self.preprocess_features(X, fit=False)
        return self.model.predict_proba(X_processed)

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
            filepath = str(models_dir / f"classifier_{timestamp}.joblib")

        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Save everything including CV metrics
        model_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,  # Single pipeline: scaler + KNN imputer
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
    def load(cls, filepath: str) -> 'WaterQualityClassifier':
        """
        Load a saved model.

        Args:
            filepath: Path to saved model

        Returns:
            Loaded WaterQualityClassifier instance
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
        # Handle both new format (preprocessor) and legacy format (scaler + imputer)
        if 'preprocessor' in model_data:
            instance.preprocessor = model_data['preprocessor']
        elif 'scaler' in model_data and 'imputer' in model_data:
            # Legacy model: create preprocessor from separate components
            # Note: This preserves backwards compatibility but won't have KNN imputation
            logger.warning("Loading legacy model with SimpleImputer (not KNN)")
            from sklearn.impute import SimpleImputer
            instance.preprocessor = Pipeline([
                ('scaler', model_data['scaler']),
                ('imputer', model_data['imputer']),
            ])
        instance.feature_names = model_data['feature_names']
        instance.metrics = model_data['metrics']

        logger.info(f"Model loaded successfully")
        logger.info(f"  Type: {instance.model_type}")
        logger.info(f"  Features: {len(instance.feature_names)}")
        logger.info(f"  Saved: {model_data.get('timestamp', 'unknown')}")

        return instance


if __name__ == "__main__":
    # Example usage
    from src.preprocessing.feature_engineering import prepare_ml_dataset
    import sys

    logger.info("=" * 80)
    logger.info("Training Water Quality Classifier")
    logger.info("=" * 80)

    # Load data
    logger.info("\nLoading and preparing dataset...")
    df = prepare_ml_dataset()

    # Initialize classifier
    classifier = WaterQualityClassifier(model_type='random_forest')

    # Prepare data
    X, y, feature_names = classifier.prepare_data(df)

    # Train
    results = classifier.train(X, y)

    # Get feature importance
    importance = classifier.get_feature_importance(top_n=15)

    # Save model
    model_path = classifier.save()

    logger.info(f"\n{'=' * 80}")
    logger.info("TRAINING COMPLETE")
    logger.info(f"{'=' * 80}")
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
    logger.info(f"Test F1 Score: {results['test_metrics']['f1_score']:.4f}")
