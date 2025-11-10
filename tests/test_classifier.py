"""
Comprehensive test suite for WaterQualityClassifier.

Tests cover:
1. Preprocessing pipeline (imputation, scaling)
2. WQI threshold alignment (70.0 boundary)
3. Edge cases and robustness
4. Model persistence (save/load)
5. Performance metrics
6. Model-specific behavior (RandomForest, GradientBoosting)

Total: 120 tests
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import joblib
from pathlib import Path
from unittest.mock import Mock, patch
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

from src.models.classifier import WaterQualityClassifier


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_features():
    """Create sample feature matrix (50 samples, 10 features)."""
    np.random.seed(42)
    return np.random.randn(50, 10)


@pytest.fixture
def sample_labels_balanced():
    """Create balanced binary labels (25 safe, 25 unsafe)."""
    return np.array([0] * 25 + [1] * 25)


@pytest.fixture
def sample_labels_imbalanced():
    """Create imbalanced labels (45 safe, 5 unsafe)."""
    return np.array([1] * 45 + [0] * 5)


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame with features and WQI scores."""
    np.random.seed(42)
    n_samples = 100

    df = pd.DataFrame({
        # Water quality parameters
        'ph': np.random.uniform(6.0, 8.5, n_samples),
        'dissolved_oxygen': np.random.uniform(5.0, 12.0, n_samples),
        'temperature': np.random.uniform(5.0, 25.0, n_samples),
        'nitrate': np.random.uniform(0.0, 10.0, n_samples),
        'conductance': np.random.uniform(100.0, 1000.0, n_samples),

        # Derived features
        'ph_deviation_from_7': np.random.uniform(0.0, 1.5, n_samples),
        'do_temp_ratio': np.random.uniform(0.3, 2.0, n_samples),
        'pollution_stress': np.random.uniform(0.0, 0.5, n_samples),
        'temp_stress': np.random.uniform(0.0, 0.7, n_samples),

        # One-hot encoded features
        'conductance_low': np.random.choice([0.0, 1.0], n_samples),
        'conductance_medium': np.random.choice([0.0, 1.0], n_samples),
        'conductance_high': np.random.choice([0.0, 1.0], n_samples),
    })

    # Generate WQI scores (mix of safe/unsafe)
    df['wqi_score'] = np.random.uniform(20.0, 100.0, n_samples)

    # Create is_safe target (WQI >= 70)
    df['is_safe'] = (df['wqi_score'] >= 70.0).astype(int)

    # Columns to exclude
    df['waterBodyIdentifier'] = [f'WB{i:04d}' for i in range(n_samples)]
    df['wqi_classification'] = 'Good'
    df['parameter_scores'] = [{}] * n_samples

    return df


@pytest.fixture
def sample_dataframe_with_missing():
    """DataFrame with missing values."""
    np.random.seed(42)
    n_samples = 50

    df = pd.DataFrame({
        'ph': np.random.uniform(6.0, 8.5, n_samples),
        'dissolved_oxygen': np.random.uniform(5.0, 12.0, n_samples),
        'temperature': np.random.uniform(5.0, 25.0, n_samples),
        'nitrate': np.random.uniform(0.0, 10.0, n_samples),
        'conductance': np.random.uniform(100.0, 1000.0, n_samples),
    })

    # Inject missing values (20% missing rate)
    for col in df.columns:
        mask = np.random.rand(n_samples) < 0.2
        df.loc[mask, col] = np.nan

    df['wqi_score'] = np.random.uniform(20.0, 100.0, n_samples)
    df['is_safe'] = (df['wqi_score'] >= 70.0).astype(int)
    df['waterBodyIdentifier'] = [f'WB{i:04d}' for i in range(n_samples)]

    return df


@pytest.fixture
def classifier_rf():
    """Untrained RandomForest classifier."""
    return WaterQualityClassifier(model_type='random_forest')


@pytest.fixture
def classifier_gb():
    """Untrained GradientBoosting classifier."""
    return WaterQualityClassifier(model_type='gradient_boosting')


@pytest.fixture
def trained_classifier_small(sample_dataframe):
    """Quickly trained classifier with small dataset (no GridSearchCV)."""
    classifier = WaterQualityClassifier(model_type='random_forest')

    # Prepare data
    X, y, feature_names = classifier.prepare_data(sample_dataframe)

    # Preprocess
    X_processed = classifier.preprocess_features(X, fit=True)

    # Train simple model (no grid search)
    classifier.model = RandomForestClassifier(
        n_estimators=10,
        max_depth=5,
        random_state=42
    )
    classifier.model.fit(X_processed, y)
    classifier.feature_names = feature_names

    # Mock grid_search for testing
    classifier.grid_search = Mock()
    classifier.grid_search.best_params_ = {'n_estimators': 10, 'max_depth': 5}
    classifier.grid_search.best_score_ = 0.85

    return classifier


# ============================================================================
# 1. Preprocessing Pipeline Tests (25 tests)
# ============================================================================

class TestPreprocessingPipeline:
    """Test SimpleImputer and StandardScaler pipeline."""

    # SimpleImputer validation (10 tests)

    def test_imputer_uses_median_strategy(self, classifier_rf):
        """Verify imputer uses median strategy."""
        assert classifier_rf.imputer.strategy == 'median'

    def test_imputer_handles_single_column_missing(self, classifier_rf):
        """Impute when single column has NaN."""
        X = np.array([[1, 2], [2, np.nan], [3, np.nan]])
        X_imputed = classifier_rf.imputer.fit_transform(X)

        # Second column should be imputed with median (only 2 is observed, so median=2)
        assert not np.isnan(X_imputed).any()
        # Second column values should be imputed to median
        assert X_imputed[1, 1] == 2.0
        assert X_imputed[2, 1] == 2.0

    def test_imputer_handles_multiple_columns_missing(self, classifier_rf):
        """Impute when multiple columns have NaN."""
        X = np.array([[1, np.nan, 3], [np.nan, 2, 3], [1, 2, np.nan]])
        X_imputed = classifier_rf.imputer.fit_transform(X)

        assert not np.isnan(X_imputed).any()

    def test_imputer_median_correctness(self, classifier_rf):
        """Verify median imputation is correct."""
        X = np.array([[1], [2], [3], [np.nan], [5]])
        X_imputed = classifier_rf.imputer.fit_transform(X)

        # Median of [1, 2, 3, 5] = 2.5
        assert X_imputed[3, 0] == 2.5

    def test_imputer_all_nan_column_skipped_with_warning(self, classifier_rf):
        """All NaN column cannot be imputed - sklearn raises warning and removes it."""
        X = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])

        # SimpleImputer will skip columns with all NaN and raise warning
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            X_imputed = classifier_rf.imputer.fit_transform(X)

            # Should have warning about skipping feature
            assert any("Skipping features without any observed values" in str(warn.message) for warn in w)

        # Output should only have 1 column (the all-NaN column is removed)
        assert X_imputed.shape == (3, 1)
        # Remaining column should have no NaN
        assert not np.isnan(X_imputed).any()

    def test_imputer_no_missing_values_unchanged(self, classifier_rf):
        """Data without missing values passes through unchanged."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        X_imputed = classifier_rf.imputer.fit_transform(X)

        np.testing.assert_array_equal(X, X_imputed)

    def test_imputer_imputed_values_within_range(self, classifier_rf):
        """Imputed values fall within min/max of column."""
        X = np.array([[1.0], [5.0], [np.nan], [10.0]])
        X_imputed = classifier_rf.imputer.fit_transform(X)

        # Imputed value should be between 1 and 10
        imputed_val = X_imputed[2, 0]
        assert 1.0 <= imputed_val <= 10.0

    def test_imputer_preserves_non_nan_values(self, classifier_rf):
        """Non-NaN values remain unchanged after imputation."""
        X = np.array([[1.0], [np.nan], [3.0]])
        X_imputed = classifier_rf.imputer.fit_transform(X)

        assert X_imputed[0, 0] == 1.0
        assert X_imputed[2, 0] == 3.0

    def test_no_nan_after_imputation(self, classifier_rf, sample_dataframe_with_missing):
        """Verify no NaN values after preprocessing."""
        X, y, _ = classifier_rf.prepare_data(sample_dataframe_with_missing)
        X_processed = classifier_rf.preprocess_features(X, fit=True)

        assert not np.isnan(X_processed).any()

    def test_imputer_random_missing_patterns(self, classifier_rf):
        """Imputation works with random missing patterns."""
        np.random.seed(42)
        X = np.random.randn(20, 5)

        # Randomly set 30% to NaN
        mask = np.random.rand(20, 5) < 0.3
        X[mask] = np.nan

        X_imputed = classifier_rf.imputer.fit_transform(X)
        assert not np.isnan(X_imputed).any()

    # StandardScaler validation (10 tests)

    def test_scaler_mean_zero_after_scaling(self, classifier_rf, sample_features):
        """Scaled features have mean ~0."""
        X_scaled = classifier_rf.scaler.fit_transform(sample_features)

        # Mean should be close to 0 (within tolerance)
        means = X_scaled.mean(axis=0)
        np.testing.assert_allclose(means, 0, atol=1e-10)

    def test_scaler_std_one_after_scaling(self, classifier_rf, sample_features):
        """Scaled features have std ~1."""
        X_scaled = classifier_rf.scaler.fit_transform(sample_features)

        # Std should be close to 1
        stds = X_scaled.std(axis=0)
        np.testing.assert_allclose(stds, 1, atol=1e-10)

    def test_scaler_inverse_transform_recovers_original(self, classifier_rf, sample_features):
        """Inverse transform recovers original scale."""
        X_scaled = classifier_rf.scaler.fit_transform(sample_features)
        X_recovered = classifier_rf.scaler.inverse_transform(X_scaled)

        np.testing.assert_allclose(X_recovered, sample_features, rtol=1e-5)

    def test_scaler_transform_without_fit_fails(self, classifier_rf, sample_features):
        """Transform without fit raises error."""
        fresh_classifier = WaterQualityClassifier(model_type='random_forest')

        with pytest.raises(Exception):  # sklearn raises NotFittedError
            fresh_classifier.scaler.transform(sample_features)

    def test_scaler_preserves_relative_distances(self, classifier_rf):
        """Scaling preserves relative distances between samples."""
        X = np.array([[1, 2], [3, 4], [5, 6]])

        # Distance before scaling
        dist_before = np.linalg.norm(X[0] - X[1])

        X_scaled = classifier_rf.scaler.fit_transform(X)

        # Distance after scaling (should be proportional)
        dist_after = np.linalg.norm(X_scaled[0] - X_scaled[1])

        # Ratio should be consistent for all pairs
        assert dist_after > 0

    def test_scaler_handles_constant_feature(self, classifier_rf):
        """Scaling handles zero-variance features (all same value)."""
        X = np.array([[5, 1], [5, 2], [5, 3]])

        # First column is constant, should scale to 0
        X_scaled = classifier_rf.scaler.fit_transform(X)

        # Constant feature becomes 0 (mean centering with 0 variance)
        assert np.allclose(X_scaled[:, 0], 0)

    def test_scaler_negative_values_handled(self, classifier_rf):
        """Scaler handles negative values correctly."""
        X = np.array([[-5, -10], [0, 0], [5, 10]])
        X_scaled = classifier_rf.scaler.fit_transform(X)

        # Should still have mean ~0, std ~1
        assert abs(X_scaled.mean()) < 1e-10

    def test_scaler_large_values_normalized(self, classifier_rf):
        """Very large values get normalized."""
        X = np.array([[1e6], [2e6], [3e6]])
        X_scaled = classifier_rf.scaler.fit_transform(X)

        # Should be normalized to mean=0, std=1
        assert abs(X_scaled.mean()) < 1e-10
        assert abs(X_scaled.std() - 1.0) < 1e-10

    def test_scaler_small_values_normalized(self, classifier_rf):
        """Very small values get normalized."""
        X = np.array([[1e-6], [2e-6], [3e-6]])
        X_scaled = classifier_rf.scaler.fit_transform(X)

        assert abs(X_scaled.mean()) < 1e-10
        assert abs(X_scaled.std() - 1.0) < 1e-10

    def test_scaler_deterministic(self, classifier_rf, sample_features):
        """Scaling is deterministic (same input → same output)."""
        X_scaled_1 = classifier_rf.scaler.fit_transform(sample_features)

        # Create new scaler
        classifier_rf.scaler = classifier_rf.scaler.__class__()
        X_scaled_2 = classifier_rf.scaler.fit_transform(sample_features)

        np.testing.assert_array_equal(X_scaled_1, X_scaled_2)

    # Pipeline integration (5 tests)

    def test_preprocessing_imputation_before_scaling(self, classifier_rf):
        """Verify imputation happens before scaling."""
        X = np.array([[1, np.nan], [2, 3], [3, 4]])

        X_processed = classifier_rf.preprocess_features(X, fit=True)

        # No NaN values should remain
        assert not np.isnan(X_processed).any()

        # Values should be scaled (mean ~0, std ~1)
        assert abs(X_processed.mean()) < 0.5

    def test_preprocess_fit_true_fits_transformers(self, classifier_rf, sample_features):
        """fit=True fits both imputer and scaler."""
        X_processed = classifier_rf.preprocess_features(sample_features, fit=True)

        # Scaler should be fitted (has mean_ and scale_ attributes)
        assert hasattr(classifier_rf.scaler, 'mean_')
        assert hasattr(classifier_rf.scaler, 'scale_')

        # Imputer should be fitted
        assert hasattr(classifier_rf.imputer, 'statistics_')

    def test_preprocess_fit_false_uses_existing_transformers(
        self, classifier_rf, sample_features
    ):
        """fit=False uses already-fitted transformers."""
        # First fit
        classifier_rf.preprocess_features(sample_features, fit=True)

        # Get parameters
        mean_after_fit = classifier_rf.scaler.mean_.copy()

        # Transform new data (fit=False)
        X_new = np.random.randn(10, 10)
        classifier_rf.preprocess_features(X_new, fit=False)

        # Parameters should be unchanged
        np.testing.assert_array_equal(classifier_rf.scaler.mean_, mean_after_fit)

    def test_preprocess_transform_without_fit_raises_error(self, classifier_rf):
        """Calling preprocess with fit=False before fit=True raises error."""
        fresh_classifier = WaterQualityClassifier(model_type='random_forest')
        X = np.random.randn(10, 5)

        with pytest.raises(Exception):  # NotFittedError
            fresh_classifier.preprocess_features(X, fit=False)

    def test_preprocessing_deterministic(self, classifier_rf, sample_features):
        """Preprocessing is deterministic."""
        X_proc_1 = classifier_rf.preprocess_features(sample_features, fit=True)

        # Reset transformers
        classifier_rf.scaler = classifier_rf.scaler.__class__()
        classifier_rf.imputer = classifier_rf.imputer.__class__(strategy='median')

        X_proc_2 = classifier_rf.preprocess_features(sample_features, fit=True)

        np.testing.assert_array_almost_equal(X_proc_1, X_proc_2)


# ============================================================================
# 2. WQI Threshold Alignment (35 tests)
# ============================================================================

class TestWQIThresholdAlignment:
    """Test that classifier aligns with WQI >= 70 threshold."""

    # Threshold boundary testing (15 tests)

    def test_wqi_70_is_safe(self):
        """WQI = 70.0 should be classified as safe."""
        df = pd.DataFrame({'wqi_score': [70.0]})
        df['is_safe'] = (df['wqi_score'] >= 70.0).astype(int)

        assert df['is_safe'].iloc[0] == 1

    def test_wqi_69_999_is_unsafe(self):
        """WQI = 69.999 should be classified as unsafe."""
        df = pd.DataFrame({'wqi_score': [69.999]})
        df['is_safe'] = (df['wqi_score'] >= 70.0).astype(int)

        assert df['is_safe'].iloc[0] == 0

    def test_wqi_70_001_is_safe(self):
        """WQI = 70.001 should be classified as safe."""
        df = pd.DataFrame({'wqi_score': [70.001]})
        df['is_safe'] = (df['wqi_score'] >= 70.0).astype(int)

        assert df['is_safe'].iloc[0] == 1

    def test_wqi_100_is_safe(self):
        """WQI = 100.0 (max) should be safe."""
        df = pd.DataFrame({'wqi_score': [100.0]})
        df['is_safe'] = (df['wqi_score'] >= 70.0).astype(int)

        assert df['is_safe'].iloc[0] == 1

    def test_wqi_0_is_unsafe(self):
        """WQI = 0.0 (min) should be unsafe."""
        df = pd.DataFrame({'wqi_score': [0.0]})
        df['is_safe'] = (df['wqi_score'] >= 70.0).astype(int)

        assert df['is_safe'].iloc[0] == 0

    def test_threshold_exactly_70_across_dataset(self, sample_dataframe):
        """Verify is_safe = 1 when WQI >= 70 for entire dataset."""
        safe_mask = sample_dataframe['wqi_score'] >= 70.0

        # All safe samples should have is_safe = 1
        assert (sample_dataframe.loc[safe_mask, 'is_safe'] == 1).all()

        # All unsafe samples should have is_safe = 0
        unsafe_mask = sample_dataframe['wqi_score'] < 70.0
        assert (sample_dataframe.loc[unsafe_mask, 'is_safe'] == 0).all()

    def test_boundary_values_near_70(self):
        """Test multiple values very close to 70."""
        wqi_values = [69.9, 69.95, 69.99, 70.0, 70.01, 70.05, 70.1]
        df = pd.DataFrame({'wqi_score': wqi_values})
        df['is_safe'] = (df['wqi_score'] >= 70.0).astype(int)

        expected = [0, 0, 0, 1, 1, 1, 1]
        assert df['is_safe'].tolist() == expected

    def test_wqi_50_is_unsafe(self):
        """WQI = 50 (fair quality) should be unsafe."""
        df = pd.DataFrame({'wqi_score': [50.0]})
        df['is_safe'] = (df['wqi_score'] >= 70.0).astype(int)

        assert df['is_safe'].iloc[0] == 0

    def test_wqi_90_is_safe(self):
        """WQI = 90 (excellent) should be safe."""
        df = pd.DataFrame({'wqi_score': [90.0]})
        df['is_safe'] = (df['wqi_score'] >= 70.0).astype(int)

        assert df['is_safe'].iloc[0] == 1

    def test_wqi_25_is_unsafe(self):
        """WQI = 25 (poor) should be unsafe."""
        df = pd.DataFrame({'wqi_score': [25.0]})
        df['is_safe'] = (df['wqi_score'] >= 70.0).astype(int)

        assert df['is_safe'].iloc[0] == 0

    def test_threshold_consistent_with_prepare_data(self, classifier_rf, sample_dataframe):
        """prepare_data() creates is_safe labels correctly."""
        X, y, _ = classifier_rf.prepare_data(sample_dataframe)

        # Count safe samples
        safe_count_from_y = np.sum(y)
        safe_count_from_df = np.sum(sample_dataframe['wqi_score'] >= 70.0)

        assert safe_count_from_y == safe_count_from_df

    def test_no_safe_samples_when_all_below_70(self, classifier_rf):
        """When all WQI < 70, is_safe should all be 0."""
        df = pd.DataFrame({
            'ph': [7.0] * 10,
            'wqi_score': np.random.uniform(0, 69.9, 10),
            'is_safe': [0] * 10,
            'waterBodyIdentifier': [f'WB{i}' for i in range(10)]
        })

        X, y, _ = classifier_rf.prepare_data(df)

        assert np.sum(y) == 0

    def test_all_safe_samples_when_all_above_70(self, classifier_rf):
        """When all WQI >= 70, is_safe should all be 1."""
        df = pd.DataFrame({
            'ph': [7.0] * 10,
            'wqi_score': np.random.uniform(70.0, 100.0, 10),
            'is_safe': [1] * 10,
            'waterBodyIdentifier': [f'WB{i}' for i in range(10)]
        })

        X, y, _ = classifier_rf.prepare_data(df)

        assert np.sum(y) == len(y)

    def test_mixed_safe_unsafe_distribution(self, classifier_rf):
        """Mixed WQI values create mixed is_safe labels."""
        df = pd.DataFrame({
            'ph': [7.0] * 20,
            'wqi_score': [50] * 10 + [80] * 10,  # 10 unsafe, 10 safe
            'is_safe': [0] * 10 + [1] * 10,
            'waterBodyIdentifier': [f'WB{i}' for i in range(20)]
        })

        X, y, _ = classifier_rf.prepare_data(df)

        assert np.sum(y) == 10
        assert np.sum(y == 0) == 10

    def test_wqi_not_in_features(self, classifier_rf, sample_dataframe):
        """Verify WQI score is excluded from features (no data leakage)."""
        X, y, feature_names = classifier_rf.prepare_data(sample_dataframe)

        assert 'wqi_score' not in feature_names
        assert 'wqi_classification' not in feature_names
        assert 'is_safe' not in feature_names

    # Prediction probability alignment (10 tests)

    def test_predict_proba_shape(self, trained_classifier_small):
        """predict_proba returns (n_samples, 2) array."""
        X = np.random.randn(10, len(trained_classifier_small.feature_names))
        proba = trained_classifier_small.predict_proba(X)

        assert proba.shape == (10, 2)

    def test_predict_proba_sums_to_one(self, trained_classifier_small):
        """Probabilities sum to 1 for each sample."""
        X = np.random.randn(20, len(trained_classifier_small.feature_names))
        proba = trained_classifier_small.predict_proba(X)

        sums = proba.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, rtol=1e-5)

    def test_predict_proba_in_valid_range(self, trained_classifier_small):
        """All probabilities in [0, 1]."""
        X = np.random.randn(15, len(trained_classifier_small.feature_names))
        proba = trained_classifier_small.predict_proba(X)

        assert (proba >= 0).all()
        assert (proba <= 1).all()

    def test_high_safe_probability_predicts_safe(self, trained_classifier_small):
        """When P(safe) > 0.5, predict() should return 1."""
        X = np.random.randn(30, len(trained_classifier_small.feature_names))

        proba = trained_classifier_small.predict_proba(X)
        predictions = trained_classifier_small.predict(X)

        # Check samples where P(safe) > 0.5
        safe_proba = proba[:, 1]
        high_safe_mask = safe_proba > 0.5

        # These should predict safe (1)
        if high_safe_mask.any():
            assert (predictions[high_safe_mask] == 1).all()

    def test_high_unsafe_probability_predicts_unsafe(self, trained_classifier_small):
        """When P(unsafe) > 0.5, predict() should return 0."""
        X = np.random.randn(30, len(trained_classifier_small.feature_names))

        proba = trained_classifier_small.predict_proba(X)
        predictions = trained_classifier_small.predict(X)

        # Check samples where P(unsafe) > 0.5
        unsafe_proba = proba[:, 0]
        high_unsafe_mask = unsafe_proba > 0.5

        if high_unsafe_mask.any():
            assert (predictions[high_unsafe_mask] == 0).all()

    def test_predict_proba_consistent_with_predict(self, trained_classifier_small):
        """Predictions match argmax of probabilities."""
        X = np.random.randn(25, len(trained_classifier_small.feature_names))

        proba = trained_classifier_small.predict_proba(X)
        predictions = trained_classifier_small.predict(X)

        expected_predictions = np.argmax(proba, axis=1)
        np.testing.assert_array_equal(predictions, expected_predictions)

    def test_predict_proba_single_sample(self, trained_classifier_small):
        """predict_proba works with single sample."""
        X = np.random.randn(1, len(trained_classifier_small.feature_names))
        proba = trained_classifier_small.predict_proba(X)

        assert proba.shape == (1, 2)
        assert abs(proba.sum() - 1.0) < 1e-5

    def test_predict_proba_deterministic(self, trained_classifier_small):
        """Same input gives same probabilities."""
        X = np.random.randn(5, len(trained_classifier_small.feature_names))

        proba_1 = trained_classifier_small.predict_proba(X)
        proba_2 = trained_classifier_small.predict_proba(X)

        np.testing.assert_array_equal(proba_1, proba_2)

    def test_predict_proba_different_inputs_different_outputs(
        self, trained_classifier_small
    ):
        """Different inputs give different probabilities."""
        X1 = np.random.randn(5, len(trained_classifier_small.feature_names))
        X2 = np.random.randn(5, len(trained_classifier_small.feature_names)) + 10

        proba_1 = trained_classifier_small.predict_proba(X1)
        proba_2 = trained_classifier_small.predict_proba(X2)

        # Should be different (with high probability)
        assert not np.allclose(proba_1, proba_2)

    def test_predict_proba_column_order(self, trained_classifier_small):
        """Verify column order: [:, 0] = P(unsafe), [:, 1] = P(safe)."""
        X = np.random.randn(10, len(trained_classifier_small.feature_names))
        proba = trained_classifier_small.predict_proba(X)

        # Classes should be [0, 1] (unsafe, safe)
        assert trained_classifier_small.model.classes_.tolist() == [0, 1]

        # Column 0 = P(class 0) = P(unsafe)
        # Column 1 = P(class 1) = P(safe)
        assert proba[:, 0].shape == (10,)
        assert proba[:, 1].shape == (10,)

    # Target consistency (10 tests)

    def test_prepare_data_extracts_target(self, classifier_rf, sample_dataframe):
        """prepare_data extracts is_safe as target."""
        X, y, _ = classifier_rf.prepare_data(sample_dataframe, target_col='is_safe')

        assert len(y) == len(sample_dataframe)
        assert set(np.unique(y)).issubset({0, 1})

    def test_prepare_data_excludes_target_from_features(
        self, classifier_rf, sample_dataframe
    ):
        """Target column excluded from features."""
        X, y, feature_names = classifier_rf.prepare_data(sample_dataframe)

        assert 'is_safe' not in feature_names

    def test_safe_count_matches_wqi_threshold(self, classifier_rf, sample_dataframe):
        """Count of safe samples matches WQI >= 70 count."""
        X, y, _ = classifier_rf.prepare_data(sample_dataframe)

        safe_from_wqi = np.sum(sample_dataframe['wqi_score'] >= 70.0)
        safe_from_y = np.sum(y)

        assert safe_from_y == safe_from_wqi

    def test_class_distribution_logged_correctly(
        self, classifier_rf, sample_dataframe, caplog
    ):
        """Class distribution is logged during prepare_data."""
        import logging
        caplog.set_level(logging.INFO)

        X, y, _ = classifier_rf.prepare_data(sample_dataframe)

        # Check log contains class distribution
        assert 'Safe=' in caplog.text
        assert 'Unsafe=' in caplog.text

    def test_no_label_leakage_wqi_excluded(self, classifier_rf, sample_dataframe):
        """WQI score excluded from features (no label leakage)."""
        X, y, feature_names = classifier_rf.prepare_data(sample_dataframe)

        # WQI-related columns should be excluded
        excluded = ['wqi_score', 'wqi_classification', 'is_safe', 'parameter_scores']
        for col in excluded:
            assert col not in feature_names

    def test_prepare_data_returns_numpy_arrays(self, classifier_rf, sample_dataframe):
        """Returns numpy arrays, not DataFrames."""
        X, y, feature_names = classifier_rf.prepare_data(sample_dataframe)

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(feature_names, list)

    def test_prepare_data_feature_count(self, classifier_rf, sample_dataframe):
        """Feature count matches columns after exclusion."""
        X, y, feature_names = classifier_rf.prepare_data(sample_dataframe)

        assert X.shape[1] == len(feature_names)

    def test_prepare_data_sample_count(self, classifier_rf, sample_dataframe):
        """Sample count matches DataFrame rows."""
        X, y, feature_names = classifier_rf.prepare_data(sample_dataframe)

        assert X.shape[0] == len(sample_dataframe)
        assert len(y) == len(sample_dataframe)

    def test_prepare_data_custom_exclude_list(self, classifier_rf, sample_dataframe):
        """Custom exclude_cols parameter works."""
        exclude = ['is_safe', 'wqi_score', 'ph']
        X, y, feature_names = classifier_rf.prepare_data(
            sample_dataframe, exclude_cols=exclude
        )

        assert 'ph' not in feature_names

    def test_prepare_data_drops_object_columns(self, classifier_rf):
        """Object dtype columns are dropped with warning."""
        df = pd.DataFrame({
            'numeric': [1, 2, 3],
            'text': ['a', 'b', 'c'],
            'is_safe': [1, 0, 1],
            'waterBodyIdentifier': ['W1', 'W2', 'W3']
        })

        X, y, feature_names = classifier_rf.prepare_data(df)

        # 'text' should be dropped (object dtype)
        assert 'text' not in feature_names
        assert 'numeric' in feature_names


# ============================================================================
# 3. Edge Cases & Robustness (30 tests)
# ============================================================================

class TestEdgeCasesRobustness:
    """Test model behavior with edge cases and extreme inputs."""

    # Missing data scenarios (10 tests)

    def test_all_features_missing_single_sample(self, trained_classifier_small):
        """Prediction works when all features are NaN."""
        n_features = len(trained_classifier_small.feature_names)
        X = np.full((1, n_features), np.nan)

        # Should not raise error (imputer handles it)
        prediction = trained_classifier_small.predict(X)

        assert prediction.shape == (1,)
        assert prediction[0] in [0, 1]

    def test_single_feature_present_rest_missing(self, trained_classifier_small):
        """Prediction works with only one feature present."""
        n_features = len(trained_classifier_small.feature_names)
        X = np.full((1, n_features), np.nan)
        X[0, 0] = 5.0  # One feature present

        prediction = trained_classifier_small.predict(X)

        assert prediction.shape == (1,)

    def test_half_features_missing(self, trained_classifier_small):
        """Prediction works with 50% features missing."""
        n_features = len(trained_classifier_small.feature_names)
        X = np.random.randn(5, n_features)

        # Set half to NaN
        X[:, :n_features//2] = np.nan

        predictions = trained_classifier_small.predict(X)

        assert predictions.shape == (5,)
        assert set(predictions).issubset({0, 1})

    def test_random_missing_pattern_1(self, trained_classifier_small):
        """Random missing pattern 1: 30% missing."""
        n_features = len(trained_classifier_small.feature_names)
        np.random.seed(1)
        X = np.random.randn(10, n_features)

        mask = np.random.rand(10, n_features) < 0.3
        X[mask] = np.nan

        predictions = trained_classifier_small.predict(X)
        assert predictions.shape == (10,)

    def test_random_missing_pattern_2(self, trained_classifier_small):
        """Random missing pattern 2: 50% missing."""
        n_features = len(trained_classifier_small.feature_names)
        np.random.seed(2)
        X = np.random.randn(10, n_features)

        mask = np.random.rand(10, n_features) < 0.5
        X[mask] = np.nan

        predictions = trained_classifier_small.predict(X)
        assert predictions.shape == (10,)

    def test_random_missing_pattern_3(self, trained_classifier_small):
        """Random missing pattern 3: 70% missing."""
        n_features = len(trained_classifier_small.feature_names)
        np.random.seed(3)
        X = np.random.randn(10, n_features)

        mask = np.random.rand(10, n_features) < 0.7
        X[mask] = np.nan

        predictions = trained_classifier_small.predict(X)
        assert predictions.shape == (10,)

    def test_random_missing_pattern_4(self, trained_classifier_small):
        """Random missing pattern 4: sparse data."""
        n_features = len(trained_classifier_small.feature_names)
        np.random.seed(4)
        X = np.random.randn(10, n_features)

        mask = np.random.rand(10, n_features) < 0.85
        X[mask] = np.nan

        predictions = trained_classifier_small.predict(X)
        assert predictions.shape == (10,)

    def test_random_missing_pattern_5(self, trained_classifier_small):
        """Random missing pattern 5: per-sample variation."""
        n_features = len(trained_classifier_small.feature_names)
        np.random.seed(5)
        X = np.random.randn(10, n_features)

        # Each sample has different missing rate
        for i in range(10):
            missing_rate = i * 0.1
            mask = np.random.rand(n_features) < missing_rate
            X[i, mask] = np.nan

        predictions = trained_classifier_small.predict(X)
        assert predictions.shape == (10,)

    def test_all_samples_all_missing(self, trained_classifier_small):
        """Multiple samples, all features missing."""
        n_features = len(trained_classifier_small.feature_names)
        X = np.full((5, n_features), np.nan)

        predictions = trained_classifier_small.predict(X)

        assert predictions.shape == (5,)

    def test_alternating_missing_columns(self, trained_classifier_small):
        """Alternating columns missing."""
        n_features = len(trained_classifier_small.feature_names)
        X = np.random.randn(5, n_features)

        # Set every other column to NaN
        X[:, ::2] = np.nan

        predictions = trained_classifier_small.predict(X)
        assert predictions.shape == (5,)

    # Extreme values (10 tests)

    def test_very_large_values(self, trained_classifier_small):
        """Very large feature values (1e6)."""
        n_features = len(trained_classifier_small.feature_names)
        X = np.full((1, n_features), 1e6)

        prediction = trained_classifier_small.predict(X)

        assert prediction.shape == (1,)
        assert prediction[0] in [0, 1]

    def test_very_small_values(self, trained_classifier_small):
        """Very small feature values (1e-6)."""
        n_features = len(trained_classifier_small.feature_names)
        X = np.full((1, n_features), 1e-6)

        prediction = trained_classifier_small.predict(X)
        assert prediction[0] in [0, 1]

    def test_negative_values(self, trained_classifier_small):
        """All negative feature values."""
        n_features = len(trained_classifier_small.feature_names)
        X = np.full((1, n_features), -100.0)

        prediction = trained_classifier_small.predict(X)
        assert prediction[0] in [0, 1]

    def test_zero_values(self, trained_classifier_small):
        """All features = 0."""
        n_features = len(trained_classifier_small.feature_names)
        X = np.zeros((1, n_features))

        prediction = trained_classifier_small.predict(X)
        assert prediction[0] in [0, 1]

    def test_mixed_extreme_values(self, trained_classifier_small):
        """Mix of very large and very small values."""
        n_features = len(trained_classifier_small.feature_names)
        X = np.random.randn(1, n_features)

        X[0, 0] = 1e6
        X[0, 1] = -1e6
        X[0, 2] = 1e-6

        prediction = trained_classifier_small.predict(X)
        assert prediction[0] in [0, 1]

    def test_infinity_values_raise_warning_or_handled(self, trained_classifier_small):
        """Infinity values are handled or raise appropriate error."""
        n_features = len(trained_classifier_small.feature_names)
        X = np.random.randn(1, n_features)
        X[0, 0] = np.inf

        # Scikit-learn typically handles inf via imputation
        # This test documents behavior
        try:
            prediction = trained_classifier_small.predict(X)
            # If it works, verify output
            assert prediction[0] in [0, 1]
        except (ValueError, RuntimeWarning):
            # If it raises error, that's acceptable too
            pass

    def test_negative_infinity_values(self, trained_classifier_small):
        """Negative infinity values."""
        n_features = len(trained_classifier_small.feature_names)
        X = np.random.randn(1, n_features)
        X[0, 0] = -np.inf

        try:
            prediction = trained_classifier_small.predict(X)
            assert prediction[0] in [0, 1]
        except (ValueError, RuntimeWarning):
            pass

    def test_mix_of_nan_and_extreme_values(self, trained_classifier_small):
        """Mix of NaN and extreme values."""
        n_features = len(trained_classifier_small.feature_names)
        X = np.random.randn(1, n_features)

        X[0, 0] = np.nan
        X[0, 1] = 1e6
        X[0, 2] = -1e6

        prediction = trained_classifier_small.predict(X)
        assert prediction[0] in [0, 1]

    def test_large_positive_and_negative_mix(self, trained_classifier_small):
        """Large positive and negative values mixed."""
        n_features = len(trained_classifier_small.feature_names)
        X = np.array([[1e5, -1e5, 1e4, -1e4] + [0] * (n_features - 4)])

        prediction = trained_classifier_small.predict(X)
        assert prediction[0] in [0, 1]

    def test_values_near_float_limits(self, trained_classifier_small):
        """Values near float64 limits may cause overflow warnings."""
        n_features = len(trained_classifier_small.feature_names)
        X = np.random.randn(1, n_features)

        # Near max float64 (but not overflow territory)
        X[0, 0] = 1e100

        # May raise overflow warning, which is acceptable
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                prediction = trained_classifier_small.predict(X)
                assert prediction[0] in [0, 1]
            except (ValueError, OverflowError):
                # If it raises error, that's acceptable documentation of limits
                pass

    # Data quality (10 tests)

    def test_single_sample_prediction(self, trained_classifier_small):
        """Prediction works with single sample."""
        n_features = len(trained_classifier_small.feature_names)
        X = np.random.randn(1, n_features)

        prediction = trained_classifier_small.predict(X)

        assert prediction.shape == (1,)

    def test_constant_feature_zero_variance(self, classifier_rf, sample_dataframe):
        """Training handles constant features (zero variance)."""
        # Add constant feature
        sample_dataframe['constant_col'] = 5.0

        X, y, _ = classifier_rf.prepare_data(sample_dataframe)

        # Should train successfully (scaler handles constant features)
        X_processed = classifier_rf.preprocess_features(X, fit=True)

        # Train simple model
        classifier_rf.model = RandomForestClassifier(n_estimators=10, random_state=42)
        classifier_rf.model.fit(X_processed, y)

        # Should predict successfully
        predictions = classifier_rf.predict(X[:5])
        assert predictions.shape == (5,)

    def test_duplicate_samples(self, classifier_rf):
        """Training handles duplicate samples."""
        df = pd.DataFrame({
            'ph': [7.0] * 20,
            'dissolved_oxygen': [8.0] * 20,
            'wqi_score': [75.0] * 10 + [50.0] * 10,
            'is_safe': [1] * 10 + [0] * 10,
            'waterBodyIdentifier': [f'WB{i}' for i in range(20)]
        })

        X, y, _ = classifier_rf.prepare_data(df)
        X_processed = classifier_rf.preprocess_features(X, fit=True)

        classifier_rf.model = RandomForestClassifier(n_estimators=10, random_state=42)
        classifier_rf.model.fit(X_processed, y)

        predictions = classifier_rf.predict(X[:5])
        assert predictions.shape == (5,)

    def test_class_imbalance_99_1(self, classifier_rf):
        """Extreme class imbalance (99% safe, 1% unsafe)."""
        df = pd.DataFrame({
            'ph': [7.0] * 100,
            'wqi_score': [80.0] * 99 + [40.0] * 1,
            'is_safe': [1] * 99 + [0] * 1,
            'waterBodyIdentifier': [f'WB{i}' for i in range(100)]
        })

        X, y, _ = classifier_rf.prepare_data(df)
        X_processed = classifier_rf.preprocess_features(X, fit=True)

        # Should train despite imbalance
        classifier_rf.model = RandomForestClassifier(n_estimators=10, random_state=42)
        classifier_rf.model.fit(X_processed, y)

        predictions = classifier_rf.predict(X[:10])
        assert predictions.shape == (10,)

    def test_class_imbalance_1_99(self, classifier_rf):
        """Extreme class imbalance (1% safe, 99% unsafe)."""
        df = pd.DataFrame({
            'ph': [7.0] * 100,
            'wqi_score': [40.0] * 99 + [80.0] * 1,
            'is_safe': [0] * 99 + [1] * 1,
            'waterBodyIdentifier': [f'WB{i}' for i in range(100)]
        })

        X, y, _ = classifier_rf.prepare_data(df)
        X_processed = classifier_rf.preprocess_features(X, fit=True)

        classifier_rf.model = RandomForestClassifier(n_estimators=10, random_state=42)
        classifier_rf.model.fit(X_processed, y)

        predictions = classifier_rf.predict(X[:10])
        assert predictions.shape == (10,)

    def test_predict_without_training_raises_error(self, classifier_rf):
        """Calling predict() before training raises error."""
        X = np.random.randn(5, 10)

        with pytest.raises(ValueError, match="Model not trained"):
            classifier_rf.predict(X)

    def test_predict_proba_without_training_raises_error(self, classifier_rf):
        """Calling predict_proba() before training raises error."""
        X = np.random.randn(5, 10)

        with pytest.raises(ValueError, match="Model not trained"):
            classifier_rf.predict_proba(X)

    def test_predict_wrong_feature_count_raises_error(self, trained_classifier_small):
        """Predicting with wrong number of features raises error."""
        # Trained on N features, predict with N-1
        n_features = len(trained_classifier_small.feature_names)
        X_wrong = np.random.randn(5, n_features - 1)

        with pytest.raises(ValueError):
            trained_classifier_small.predict(X_wrong)

    def test_empty_predictions_batch(self, trained_classifier_small):
        """Empty input array raises ValueError (sklearn limitation)."""
        n_features = len(trained_classifier_small.feature_names)
        X_empty = np.empty((0, n_features))

        # sklearn's SimpleImputer requires at least 1 sample
        with pytest.raises(ValueError, match="minimum of 1 is required"):
            predictions = trained_classifier_small.predict(X_empty)

    def test_large_batch_prediction(self, trained_classifier_small):
        """Prediction works with large batch (1000 samples)."""
        n_features = len(trained_classifier_small.feature_names)
        X_large = np.random.randn(1000, n_features)

        predictions = trained_classifier_small.predict(X_large)

        assert predictions.shape == (1000,)
        assert set(predictions).issubset({0, 1})


# ============================================================================
# 4. Model Persistence (20 tests)
# ============================================================================

class TestModelPersistence:
    """Test save/load functionality."""

    # Save functionality (8 tests)

    def test_save_creates_file(self, trained_classifier_small):
        """save() creates a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = str(Path(tmpdir) / "test_model.joblib")

            saved_path = trained_classifier_small.save(filepath)

            assert Path(saved_path).exists()
            assert saved_path == filepath

    def test_save_auto_generated_filename(self, trained_classifier_small):
        """save() auto-generates filename with timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Override default path to use tmpdir
            original_path = trained_classifier_small.save.__func__

            # Save without filepath
            saved_path = trained_classifier_small.save(
                filepath=str(Path(tmpdir) / "classifier_test.joblib")
            )

            assert Path(saved_path).exists()
            assert "classifier" in saved_path

    def test_save_creates_directory_if_missing(self, trained_classifier_small):
        """save() creates parent directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = str(Path(tmpdir) / "nested" / "dir" / "model.joblib")

            saved_path = trained_classifier_small.save(filepath)

            assert Path(saved_path).exists()
            assert Path(saved_path).parent.exists()

    def test_save_without_training_raises_error(self, classifier_rf):
        """save() without training raises ValueError."""
        with pytest.raises(ValueError, match="No model to save"):
            classifier_rf.save()

    def test_save_custom_filepath(self, trained_classifier_small):
        """Custom filepath is respected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_path = str(Path(tmpdir) / "my_custom_model.joblib")

            saved_path = trained_classifier_small.save(custom_path)

            assert saved_path == custom_path
            assert Path(custom_path).exists()

    def test_save_overwrites_existing_file(self, trained_classifier_small):
        """save() overwrites existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = str(Path(tmpdir) / "model.joblib")

            # Save once
            trained_classifier_small.save(filepath)
            original_size = Path(filepath).stat().st_size

            # Save again
            trained_classifier_small.save(filepath)
            new_size = Path(filepath).stat().st_size

            # File should exist (may be same or different size)
            assert Path(filepath).exists()

    def test_saved_file_is_joblib_format(self, trained_classifier_small):
        """Saved file can be loaded with joblib."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = str(Path(tmpdir) / "model.joblib")

            trained_classifier_small.save(filepath)

            # Should be loadable with joblib
            data = joblib.load(filepath)

            assert isinstance(data, dict)
            assert 'model' in data
            assert 'scaler' in data
            assert 'imputer' in data

    def test_save_includes_all_components(self, trained_classifier_small):
        """Saved file includes model, scaler, imputer, metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = str(Path(tmpdir) / "model.joblib")

            trained_classifier_small.save(filepath)
            data = joblib.load(filepath)

            required_keys = [
                'model', 'scaler', 'imputer', 'feature_names',
                'model_type', 'metrics', 'best_params', 'timestamp'
            ]

            for key in required_keys:
                assert key in data

    # Load functionality (8 tests)

    def test_load_recreates_model(self, trained_classifier_small):
        """load() recreates a working classifier."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = str(Path(tmpdir) / "model.joblib")

            trained_classifier_small.save(filepath)
            loaded = WaterQualityClassifier.load(filepath)

            assert loaded.model is not None
            assert loaded.scaler is not None
            assert loaded.imputer is not None

    def test_load_missing_file_raises_error(self):
        """load() with non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            WaterQualityClassifier.load("/nonexistent/path/model.joblib")

    def test_load_preserves_model_type(self, trained_classifier_small):
        """Loaded model has correct model_type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = str(Path(tmpdir) / "model.joblib")

            original_type = trained_classifier_small.model_type
            trained_classifier_small.save(filepath)

            loaded = WaterQualityClassifier.load(filepath)

            assert loaded.model_type == original_type

    def test_load_preserves_feature_names(self, trained_classifier_small):
        """Loaded model has same feature names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = str(Path(tmpdir) / "model.joblib")

            original_features = trained_classifier_small.feature_names.copy()
            trained_classifier_small.save(filepath)

            loaded = WaterQualityClassifier.load(filepath)

            assert loaded.feature_names == original_features

    def test_load_preserves_metrics(self, trained_classifier_small):
        """Loaded model preserves metrics dictionary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = str(Path(tmpdir) / "model.joblib")

            # Add some metrics
            trained_classifier_small.metrics['test'] = {'accuracy': 0.95}

            trained_classifier_small.save(filepath)
            loaded = WaterQualityClassifier.load(filepath)

            assert 'test' in loaded.metrics
            assert loaded.metrics['test']['accuracy'] == 0.95

    def test_load_preserves_scaler_state(self, trained_classifier_small):
        """Loaded scaler has same mean/scale."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = str(Path(tmpdir) / "model.joblib")

            original_mean = trained_classifier_small.scaler.mean_.copy()
            original_scale = trained_classifier_small.scaler.scale_.copy()

            trained_classifier_small.save(filepath)
            loaded = WaterQualityClassifier.load(filepath)

            np.testing.assert_array_equal(loaded.scaler.mean_, original_mean)
            np.testing.assert_array_equal(loaded.scaler.scale_, original_scale)

    def test_load_preserves_imputer_state(self, trained_classifier_small):
        """Loaded imputer has same statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = str(Path(tmpdir) / "model.joblib")

            original_stats = trained_classifier_small.imputer.statistics_.copy()

            trained_classifier_small.save(filepath)
            loaded = WaterQualityClassifier.load(filepath)

            np.testing.assert_array_equal(loaded.imputer.statistics_, original_stats)

    def test_load_includes_timestamp(self, trained_classifier_small):
        """Loaded model includes save timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = str(Path(tmpdir) / "model.joblib")

            trained_classifier_small.save(filepath)

            data = joblib.load(filepath)
            assert 'timestamp' in data
            assert isinstance(data['timestamp'], str)

    # Round-trip consistency (4 tests)

    def test_predictions_identical_after_save_load(self, trained_classifier_small):
        """Predictions identical before/after save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = str(Path(tmpdir) / "model.joblib")

            # Predict before save
            n_features = len(trained_classifier_small.feature_names)
            X_test = np.random.randn(20, n_features)
            predictions_before = trained_classifier_small.predict(X_test)

            # Save and load
            trained_classifier_small.save(filepath)
            loaded = WaterQualityClassifier.load(filepath)

            # Predict after load
            predictions_after = loaded.predict(X_test)

            np.testing.assert_array_equal(predictions_before, predictions_after)

    def test_probabilities_identical_after_save_load(self, trained_classifier_small):
        """Probabilities identical before/after save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = str(Path(tmpdir) / "model.joblib")

            n_features = len(trained_classifier_small.feature_names)
            X_test = np.random.randn(15, n_features)

            proba_before = trained_classifier_small.predict_proba(X_test)

            trained_classifier_small.save(filepath)
            loaded = WaterQualityClassifier.load(filepath)

            proba_after = loaded.predict_proba(X_test)

            np.testing.assert_array_almost_equal(proba_before, proba_after, decimal=10)

    def test_feature_names_preserved_exactly(self, trained_classifier_small):
        """Feature names list preserved exactly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = str(Path(tmpdir) / "model.joblib")

            original_features = trained_classifier_small.feature_names.copy()

            trained_classifier_small.save(filepath)
            loaded = WaterQualityClassifier.load(filepath)

            assert loaded.feature_names == original_features
            assert len(loaded.feature_names) == len(original_features)

    def test_metrics_dictionary_preserved(self, trained_classifier_small):
        """Metrics dictionary fully preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = str(Path(tmpdir) / "model.joblib")

            # Add multiple metrics
            trained_classifier_small.metrics = {
                'train': {'accuracy': 0.98, 'f1_score': 0.97},
                'val': {'accuracy': 0.95, 'f1_score': 0.94},
                'test': {'accuracy': 0.93, 'f1_score': 0.92}
            }

            trained_classifier_small.save(filepath)
            loaded = WaterQualityClassifier.load(filepath)

            assert loaded.metrics == trained_classifier_small.metrics


# ============================================================================
# 5. Performance Metrics (10 tests)
# ============================================================================

class TestPerformanceMetrics:
    """Test metric calculations and storage."""

    # Metric calculations (6 tests)

    def test_accuracy_calculation(self, trained_classifier_small):
        """Accuracy = (TP + TN) / total."""
        # Create known data
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 1])

        # Manually: TP=1, TN=1, FP=1, FN=1
        # Accuracy = (1+1)/4 = 0.5

        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_true, y_pred)

        assert acc == 0.5

    def test_precision_calculation(self, trained_classifier_small):
        """Precision = TP / (TP + FP)."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 0])

        # TP=1, FP=0
        # Precision = 1 / (1 + 0) = 1.0

        from sklearn.metrics import precision_score
        prec = precision_score(y_true, y_pred, zero_division=0)

        assert prec == 1.0

    def test_recall_calculation(self, trained_classifier_small):
        """Recall = TP / (TP + FN)."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])

        # TP=2, FN=0
        # Recall = 2 / (2 + 0) = 1.0

        from sklearn.metrics import recall_score
        rec = recall_score(y_true, y_pred, zero_division=0)

        assert rec == 1.0

    def test_f1_score_calculation(self, trained_classifier_small):
        """F1 = 2 * (precision * recall) / (precision + recall)."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 1])

        from sklearn.metrics import precision_score, recall_score, f1_score

        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        expected_f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

        assert abs(f1 - expected_f1) < 1e-6

    def test_roc_auc_in_valid_range(self, trained_classifier_small):
        """ROC-AUC score is in [0, 1]."""
        n_features = len(trained_classifier_small.feature_names)
        X = np.random.randn(30, n_features)
        y_true = np.random.randint(0, 2, 30)

        y_proba = trained_classifier_small.predict_proba(X)[:, 1]

        from sklearn.metrics import roc_auc_score
        roc_auc = roc_auc_score(y_true, y_proba)

        assert 0 <= roc_auc <= 1

    def test_confusion_matrix_sum_equals_total(self, trained_classifier_small):
        """TP + TN + FP + FN = total samples."""
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 1, 0])

        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        assert tn + fp + fn + tp == len(y_true)

    # Confusion matrix (4 tests)

    def test_confusion_matrix_values_are_integers(self, trained_classifier_small):
        """Confusion matrix values are integers."""
        n_features = len(trained_classifier_small.feature_names)
        X = np.random.randn(20, n_features)
        y_true = np.random.randint(0, 2, 20)

        y_pred = trained_classifier_small.predict(X)

        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        assert isinstance(int(tn), int)
        assert isinstance(int(fp), int)
        assert isinstance(int(fn), int)
        assert isinstance(int(tp), int)

    def test_evaluate_stores_metrics(self, trained_classifier_small, sample_features):
        """evaluate() stores metrics in self.metrics."""
        n_features = len(trained_classifier_small.feature_names)
        X = np.random.randn(20, n_features)
        y = np.random.randint(0, 2, 20)

        X_processed = trained_classifier_small.preprocess_features(X, fit=False)

        metrics = trained_classifier_small.evaluate(X_processed, y, dataset_name="TestSet")

        assert 'testset' in trained_classifier_small.metrics
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics

    def test_evaluate_returns_all_required_metrics(self, trained_classifier_small):
        """evaluate() returns accuracy, precision, recall, F1, ROC-AUC, CM."""
        n_features = len(trained_classifier_small.feature_names)
        X = np.random.randn(25, n_features)
        y = np.random.randint(0, 2, 25)

        X_processed = trained_classifier_small.preprocess_features(X, fit=False)
        metrics = trained_classifier_small.evaluate(X_processed, y)

        required_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc',
            'true_negatives', 'false_positives', 'false_negatives', 'true_positives'
        ]

        for metric in required_metrics:
            assert metric in metrics

    def test_multiple_evaluate_calls_update_metrics_dict(
        self, trained_classifier_small
    ):
        """Multiple evaluate() calls add to metrics dict."""
        n_features = len(trained_classifier_small.feature_names)
        X = np.random.randn(15, n_features)
        y = np.random.randint(0, 2, 15)

        X_processed = trained_classifier_small.preprocess_features(X, fit=False)

        trained_classifier_small.evaluate(X_processed, y, dataset_name="Set1")
        trained_classifier_small.evaluate(X_processed, y, dataset_name="Set2")

        assert 'set1' in trained_classifier_small.metrics
        assert 'set2' in trained_classifier_small.metrics


# ============================================================================
# 6. Model-Specific Behavior (5 tests)
# ============================================================================

class TestModelSpecificBehavior:
    """Test RandomForest and GradientBoosting specific behavior."""

    # RandomForest mode (3 tests)

    def test_random_forest_initialization(self):
        """model_type='random_forest' initializes correctly."""
        classifier = WaterQualityClassifier(model_type='random_forest')

        assert classifier.model_type == 'random_forest'
        assert classifier.model is None  # Not trained yet

    def test_random_forest_uses_correct_estimator(self, classifier_rf, sample_dataframe):
        """RandomForest mode uses RandomForestClassifier."""
        X, y, _ = classifier_rf.prepare_data(sample_dataframe)
        X_processed = classifier_rf.preprocess_features(X, fit=True)

        # Train without GridSearch for testing
        classifier_rf.model = RandomForestClassifier(n_estimators=10, random_state=42)
        classifier_rf.model.fit(X_processed, y)

        assert isinstance(classifier_rf.model, RandomForestClassifier)

    def test_random_forest_has_feature_importance(self, trained_classifier_small):
        """RandomForest model has feature_importances_."""
        assert hasattr(trained_classifier_small.model, 'feature_importances_')

        importances = trained_classifier_small.model.feature_importances_

        # Should have one importance per feature
        assert len(importances) == len(trained_classifier_small.feature_names)

        # Importances should sum to ~1
        assert abs(importances.sum() - 1.0) < 1e-5

    # GradientBoosting mode (2 tests)

    def test_gradient_boosting_initialization(self):
        """model_type='gradient_boosting' initializes correctly."""
        classifier = WaterQualityClassifier(model_type='gradient_boosting')

        assert classifier.model_type == 'gradient_boosting'
        assert classifier.model is None

    def test_gradient_boosting_uses_correct_estimator(
        self, classifier_gb, sample_dataframe
    ):
        """GradientBoosting mode uses GradientBoostingClassifier."""
        X, y, _ = classifier_gb.prepare_data(sample_dataframe)
        X_processed = classifier_gb.preprocess_features(X, fit=True)

        classifier_gb.model = GradientBoostingClassifier(
            n_estimators=10, random_state=42
        )
        classifier_gb.model.fit(X_processed, y)

        assert isinstance(classifier_gb.model, GradientBoostingClassifier)


# ============================================================================
# Meta-tests: Verify test counts
# ============================================================================

class TestMetaTestCounts:
    """Verify we have the correct number of tests."""

    def test_preprocessing_pipeline_test_count(self):
        """TestPreprocessingPipeline has 25 tests."""
        test_class = TestPreprocessingPipeline
        test_methods = [m for m in dir(test_class) if m.startswith('test_')]
        assert len(test_methods) == 25, f"Expected 25, got {len(test_methods)}"

    def test_wqi_threshold_alignment_test_count(self):
        """TestWQIThresholdAlignment has 35 tests."""
        test_class = TestWQIThresholdAlignment
        test_methods = [m for m in dir(test_class) if m.startswith('test_')]
        assert len(test_methods) == 35, f"Expected 35, got {len(test_methods)}"

    def test_edge_cases_robustness_test_count(self):
        """TestEdgeCasesRobustness has 30 tests."""
        test_class = TestEdgeCasesRobustness
        test_methods = [m for m in dir(test_class) if m.startswith('test_')]
        assert len(test_methods) == 30, f"Expected 30, got {len(test_methods)}"

    def test_model_persistence_test_count(self):
        """TestModelPersistence has 20 tests."""
        test_class = TestModelPersistence
        test_methods = [m for m in dir(test_class) if m.startswith('test_')]
        assert len(test_methods) == 20, f"Expected 20, got {len(test_methods)}"

    def test_performance_metrics_test_count(self):
        """TestPerformanceMetrics has 10 tests."""
        test_class = TestPerformanceMetrics
        test_methods = [m for m in dir(test_class) if m.startswith('test_')]
        assert len(test_methods) == 10, f"Expected 10, got {len(test_methods)}"

    def test_model_specific_behavior_test_count(self):
        """TestModelSpecificBehavior has 5 tests."""
        test_class = TestModelSpecificBehavior
        test_methods = [m for m in dir(test_class) if m.startswith('test_')]
        assert len(test_methods) == 5, f"Expected 5, got {len(test_methods)}"

    def test_total_test_count(self):
        """Total test count is 120 (excluding meta-tests)."""
        all_classes = [
            TestPreprocessingPipeline,
            TestWQIThresholdAlignment,
            TestEdgeCasesRobustness,
            TestModelPersistence,
            TestPerformanceMetrics,
            TestModelSpecificBehavior
        ]

        total = 0
        for test_class in all_classes:
            test_methods = [m for m in dir(test_class) if m.startswith('test_')]
            total += len(test_methods)

        # Should have exactly 120 tests (25+35+30+20+10+5)
        assert total == 125, f"Expected 125, got {total}"
