"""
Water Quality Classification Model

Binary classifier to predict safe/unsafe water quality (WQI >= 70 threshold).
Uses RandomForest and GradientBoosting with hyperparameter tuning.
"""

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
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
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

        # CRITICAL: Convert DataFrame to numpy array to avoid feature name mismatch
        # Models were trained on numpy arrays without feature names
        if hasattr(X, 'values'):  # Check if X is a DataFrame
            logger.info("Converting DataFrame to numpy array (preserving feature order)")
            X = X.values

        # Verify X is now a numpy array
        if not isinstance(X, np.ndarray):
            raise TypeError(f"Expected numpy array after conversion, got {type(X)}")

        # Impute missing values
        if fit:
            X_imputed = self.imputer.fit_transform(X)
        else:
            X_imputed = self.imputer.transform(X)

        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X_imputed)
        else:
            X_scaled = self.scaler.transform(X_imputed)

        return X_scaled

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
            base_model = RandomForestClassifier(random_state=random_state)
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced', None]
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
        logger.info(f"Best CV F1 score: {self.grid_search.best_score_:.4f}")

        # Evaluate on validation set
        val_metrics = self.evaluate(X_val_processed, y_val, dataset_name="Validation")

        # Evaluate on test set
        test_metrics = self.evaluate(X_test_processed, y_test, dataset_name="Test")

        # Store results
        results = {
            'best_params': self.grid_search.best_params_,
            'best_cv_score': self.grid_search.best_score_,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'feature_names': self.feature_names,
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test)
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

        # Generate filepath with timestamp
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"data/models/classifier_{timestamp}.joblib"

        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Save everything
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'metrics': self.metrics,
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

        logger.info(f"Loading model from {filepath}")
        model_data = joblib.load(filepath)

        # Create instance
        instance = cls(model_type=model_data['model_type'])
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.imputer = model_data['imputer']
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
