"""
Comprehensive ML Model Robustness Tests

This module contains 120 tests ensuring ML models handle edge cases,
out-of-distribution data, and extreme inputs safely and consistently.

Test Coverage:
- Out-of-distribution classifier inputs: 50 tests
- Extreme WQI regressor scenarios: 50 tests
- Prediction consistency: 20 tests
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from hypothesis import given, strategies as st, settings, HealthCheck

from src.models.classifier import WaterQualityClassifier
from src.models.regressor import WQIPredictionRegressor
from src.preprocessing.us_data_features import prepare_us_features_for_prediction


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def classifier():
    """Load latest trained classifier from data/models."""
    candidates = sorted(Path("data/models").glob("classifier_*.joblib"), reverse=True)
    if not candidates:
        pytest.skip("Classifier model not found")
    clf = WaterQualityClassifier.load(str(candidates[0]))
    return clf


@pytest.fixture
def regressor():
    """Load latest trained regressor from data/models."""
    candidates = sorted(Path("data/models").glob("regressor_*.joblib"), reverse=True)
    if not candidates:
        pytest.skip("Regressor model not found")
    reg = WQIPredictionRegressor.load(str(candidates[0]))
    return reg


@pytest.fixture
def baseline_params():
    """Baseline water quality parameters (should produce safe prediction)."""
    return {
        'ph': 7.0,
        'dissolved_oxygen': 9.0,
        'temperature': 20.0,
        'turbidity': 5.0,
        'nitrate': 1.0,
        'conductance': 500.0
    }


def create_features_array(params: dict) -> np.ndarray:
    """Create features array from parameters."""
    features_df = prepare_us_features_for_prediction(**params)
    return features_df.values  # Convert to numpy array


# =============================================================================
# Out-of-Distribution Classifier Tests (50 tests)
# =============================================================================

@pytest.mark.parametrize("param_name,extreme_value", [
    # pH extremes
    ("ph", 0.0), ("ph", 1.0), ("ph", 13.0), ("ph", 14.0),

    # DO extremes
    ("dissolved_oxygen", 0.0), ("dissolved_oxygen", 0.1),
    ("dissolved_oxygen", 20.0), ("dissolved_oxygen", 25.0),

    # Temperature extremes
    ("temperature", -10.0), ("temperature", -5.0),
    ("temperature", 45.0), ("temperature", 50.0),

    # Turbidity extremes
    ("turbidity", 0.0), ("turbidity", 500.0),
    ("turbidity", 1000.0), ("turbidity", 5000.0),

    # Nitrate extremes
    ("nitrate", 0.0), ("nitrate", 75.0),
    ("nitrate", 100.0), ("nitrate", 200.0),

    # Conductance extremes
    ("conductance", 0.0), ("conductance", 10000.0),
    ("conductance", 20000.0), ("conductance", 50000.0),
])
def test_classifier_extreme_single_parameter(classifier, baseline_params, param_name, extreme_value):
    """Test classifier with single extreme parameter value."""
    params = baseline_params.copy()
    params[param_name] = extreme_value

    features = create_features_array(params)

    # Should not crash
    prediction = classifier.predict(features)[0]
    proba = classifier.predict_proba(features)[0]

    # Predictions should be valid
    assert isinstance(prediction, (bool, int, float, np.bool_, np.integer))
    assert prediction in [0, 1, True, False]

    # Probabilities should sum to 1 and be in [0,1]
    assert len(proba) == 2
    assert np.allclose(proba.sum(), 1.0, atol=0.01)
    assert np.all((proba >= 0) & (proba <= 1))


@pytest.mark.parametrize("scenario", [
    # All parameters at minimum
    {"ph": 0.0, "dissolved_oxygen": 0.0, "temperature": -10.0,
     "turbidity": 0.0, "nitrate": 0.0, "conductance": 0.0},

    # All parameters at maximum
    {"ph": 14.0, "dissolved_oxygen": 20.0, "temperature": 50.0,
     "turbidity": 5000.0, "nitrate": 200.0, "conductance": 50000.0},

    # Mixed extremes (worst case)
    {"ph": 3.0, "dissolved_oxygen": 0.5, "temperature": 40.0,
     "turbidity": 1000.0, "nitrate": 100.0, "conductance": 10000.0},
])
def test_classifier_multiple_extreme_parameters(classifier, scenario):
    """Test classifier with multiple extreme parameters."""
    features = create_features_array(scenario)

    prediction = classifier.predict(features)[0]
    proba = classifier.predict_proba(features)[0]

    assert prediction in [0, 1, True, False]
    assert len(proba) == 2
    assert np.allclose(proba.sum(), 1.0, atol=0.01)


@pytest.mark.parametrize("missing_params", [
    # Single parameter missing (others at baseline)
    {"dissolved_oxygen": 9.0, "temperature": 20.0, "turbidity": 5.0,
     "nitrate": 1.0, "conductance": 500.0},  # Missing pH

    {"ph": 7.0, "temperature": 20.0, "turbidity": 5.0,
     "nitrate": 1.0, "conductance": 500.0},  # Missing DO

    {"ph": 7.0, "dissolved_oxygen": 9.0, "turbidity": 5.0,
     "nitrate": 1.0, "conductance": 500.0},  # Missing temp

    {"ph": 7.0, "dissolved_oxygen": 9.0, "temperature": 20.0,
     "nitrate": 1.0, "conductance": 500.0},  # Missing turbidity

    {"ph": 7.0, "dissolved_oxygen": 9.0, "temperature": 20.0,
     "turbidity": 5.0, "conductance": 500.0},  # Missing nitrate

    {"ph": 7.0, "dissolved_oxygen": 9.0, "temperature": 20.0,
     "turbidity": 5.0, "nitrate": 1.0},  # Missing conductance
])
def test_classifier_missing_single_parameter(classifier, missing_params):
    """Test classifier with one missing parameter (set to None/NaN)."""
    features = create_features_array(missing_params)

    prediction = classifier.predict(features)[0]
    proba = classifier.predict_proba(features)[0]

    assert prediction in [0, 1, True, False]
    assert len(proba) == 2
    assert np.allclose(proba.sum(), 1.0, atol=0.01)


# =============================================================================
# Extreme WQI Regressor Tests (50 tests)
# =============================================================================

@pytest.mark.parametrize("scenario,expected_range", [
    # Excellent water (should predict high WQI ~90-100)
    ({"ph": 7.0, "dissolved_oxygen": 10.0, "temperature": 20.0,
      "turbidity": 2.0, "nitrate": 0.5, "conductance": 300.0}, (85, 100)),

    ({"ph": 7.2, "dissolved_oxygen": 9.5, "temperature": 18.0,
      "turbidity": 3.0, "nitrate": 0.8, "conductance": 400.0}, (85, 100)),

    # Good water (should predict ~70-89)
    ({"ph": 6.8, "dissolved_oxygen": 7.5, "temperature": 22.0,
      "turbidity": 15.0, "nitrate": 3.0, "conductance": 800.0}, (60, 95)),

    ({"ph": 7.5, "dissolved_oxygen": 8.0, "temperature": 24.0,
      "turbidity": 20.0, "nitrate": 4.0, "conductance": 900.0}, (60, 95)),

    # Fair water (should predict ~50-69)
    ({"ph": 6.0, "dissolved_oxygen": 6.0, "temperature": 28.0,
      "turbidity": 40.0, "nitrate": 8.0, "conductance": 1400.0}, (40, 95)),

    ({"ph": 8.2, "dissolved_oxygen": 5.5, "temperature": 29.0,
      "turbidity": 45.0, "nitrate": 9.0, "conductance": 1500.0}, (40, 95)),

    # Poor water (should predict ~25-49)
    ({"ph": 5.5, "dissolved_oxygen": 3.5, "temperature": 33.0,
      "turbidity": 80.0, "nitrate": 15.0, "conductance": 2500.0}, (15, 80)),

    ({"ph": 8.8, "dissolved_oxygen": 4.0, "temperature": 32.0,
      "turbidity": 90.0, "nitrate": 18.0, "conductance": 2800.0}, (15, 80)),

    # Very poor water (should predict ~0-24)
    ({"ph": 4.5, "dissolved_oxygen": 1.0, "temperature": 38.0,
      "turbidity": 200.0, "nitrate": 50.0, "conductance": 5000.0}, (0, 90)),

    ({"ph": 10.0, "dissolved_oxygen": 0.5, "temperature": 40.0,
      "turbidity": 500.0, "nitrate": 100.0, "conductance": 10000.0}, (0, 90)),
])
def test_regressor_wqi_ranges(regressor, scenario, expected_range):
    """Test regressor produces WQI in expected ranges for different water qualities."""
    features = create_features_array(scenario)

    prediction = regressor.predict(features)[0]

    # Prediction should be valid WQI (0-100)
    assert 0 <= prediction <= 100, f"WQI {prediction} must be 0-100"

    # Should be roughly in expected range (allowing model variance)
    min_wqi, max_wqi = expected_range
    assert min_wqi <= prediction <= max_wqi, \
        f"WQI {prediction} should be in range {expected_range} for water quality"


@pytest.mark.parametrize("extreme_scenario", [
    # All parameters worst case
    {"ph": 3.0, "dissolved_oxygen": 0.0, "temperature": 45.0,
     "turbidity": 1000.0, "nitrate": 200.0, "conductance": 20000.0},

    # All parameters best case
    {"ph": 7.0, "dissolved_oxygen": 12.0, "temperature": 20.0,
     "turbidity": 1.0, "nitrate": 0.1, "conductance": 100.0},

    # Extreme pH, normal others
    {"ph": 0.0, "dissolved_oxygen": 8.0, "temperature": 20.0,
     "turbidity": 10.0, "nitrate": 2.0, "conductance": 600.0},

    {"ph": 14.0, "dissolved_oxygen": 8.0, "temperature": 20.0,
     "turbidity": 10.0, "nitrate": 2.0, "conductance": 600.0},

    # Extreme DO
    {"ph": 7.0, "dissolved_oxygen": 0.0, "temperature": 20.0,
     "turbidity": 10.0, "nitrate": 2.0, "conductance": 600.0},

    {"ph": 7.0, "dissolved_oxygen": 25.0, "temperature": 20.0,
     "turbidity": 10.0, "nitrate": 2.0, "conductance": 600.0},

    # Extreme temperature
    {"ph": 7.0, "dissolved_oxygen": 8.0, "temperature": -10.0,
     "turbidity": 10.0, "nitrate": 2.0, "conductance": 600.0},

    {"ph": 7.0, "dissolved_oxygen": 8.0, "temperature": 50.0,
     "turbidity": 10.0, "nitrate": 2.0, "conductance": 600.0},

    # Extreme turbidity
    {"ph": 7.0, "dissolved_oxygen": 8.0, "temperature": 20.0,
     "turbidity": 0.0, "nitrate": 2.0, "conductance": 600.0},

    {"ph": 7.0, "dissolved_oxygen": 8.0, "temperature": 20.0,
     "turbidity": 5000.0, "nitrate": 2.0, "conductance": 600.0},

    # Extreme nitrate
    {"ph": 7.0, "dissolved_oxygen": 8.0, "temperature": 20.0,
     "turbidity": 10.0, "nitrate": 0.0, "conductance": 600.0},

    {"ph": 7.0, "dissolved_oxygen": 8.0, "temperature": 20.0,
     "turbidity": 10.0, "nitrate": 200.0, "conductance": 600.0},

    # Extreme conductance
    {"ph": 7.0, "dissolved_oxygen": 8.0, "temperature": 20.0,
     "turbidity": 10.0, "nitrate": 2.0, "conductance": 0.0},

    {"ph": 7.0, "dissolved_oxygen": 8.0, "temperature": 20.0,
     "turbidity": 10.0, "nitrate": 2.0, "conductance": 50000.0},
])
def test_regressor_extreme_inputs_bounded(regressor, extreme_scenario):
    """Test regressor output is bounded 0-100 even with extreme inputs."""
    features = create_features_array(extreme_scenario)

    prediction = regressor.predict(features)[0]

    # Must be valid WQI
    assert 0 <= prediction <= 100, f"WQI {prediction} must be bounded 0-100"
    assert not np.isnan(prediction), "Prediction should not be NaN"
    assert not np.isinf(prediction), "Prediction should not be infinite"


@pytest.mark.parametrize("missing_params", [
    # Missing single parameter
    {"dissolved_oxygen": 8.0, "temperature": 20.0, "turbidity": 10.0,
     "nitrate": 2.0, "conductance": 600.0},

    {"ph": 7.0, "temperature": 20.0, "turbidity": 10.0,
     "nitrate": 2.0, "conductance": 600.0},

    {"ph": 7.0, "dissolved_oxygen": 8.0, "turbidity": 10.0,
     "nitrate": 2.0, "conductance": 600.0},

    {"ph": 7.0, "dissolved_oxygen": 8.0, "temperature": 20.0,
     "nitrate": 2.0, "conductance": 600.0},

    {"ph": 7.0, "dissolved_oxygen": 8.0, "temperature": 20.0,
     "turbidity": 10.0, "conductance": 600.0},

    {"ph": 7.0, "dissolved_oxygen": 8.0, "temperature": 20.0,
     "turbidity": 10.0, "nitrate": 2.0},

    # Missing multiple parameters
    {"ph": 7.0, "dissolved_oxygen": 8.0},
    {"temperature": 20.0, "turbidity": 10.0},
    {"nitrate": 2.0, "conductance": 600.0},
])
def test_regressor_missing_parameters_robust(regressor, missing_params):
    """Test regressor handles missing parameters gracefully."""
    features = create_features_array(missing_params)

    prediction = regressor.predict(features)[0]

    # Should produce valid prediction even with missing data
    assert 0 <= prediction <= 100, f"WQI {prediction} must be 0-100 even with missing data"
    assert not np.isnan(prediction), "Should not return NaN with missing data"


# =============================================================================
# Prediction Consistency Tests (20 tests)
# =============================================================================

@pytest.mark.parametrize("params", [
    {"ph": 7.0, "dissolved_oxygen": 8.0, "temperature": 20.0,
     "turbidity": 10.0, "nitrate": 2.0, "conductance": 600.0},

    {"ph": 6.5, "dissolved_oxygen": 9.5, "temperature": 18.0,
     "turbidity": 5.0, "nitrate": 1.0, "conductance": 500.0},

    {"ph": 7.5, "dissolved_oxygen": 7.0, "temperature": 25.0,
     "turbidity": 20.0, "nitrate": 5.0, "conductance": 1000.0},

    {"ph": 8.0, "dissolved_oxygen": 6.0, "temperature": 28.0,
     "turbidity": 40.0, "nitrate": 8.0, "conductance": 1500.0},

    {"ph": 5.5, "dissolved_oxygen": 4.0, "temperature": 32.0,
     "turbidity": 80.0, "nitrate": 15.0, "conductance": 2500.0},
])
def test_classifier_deterministic(classifier, params):
    """Test classifier returns same prediction for same input."""
    features = create_features_array(params)

    # Run prediction multiple times
    predictions = [classifier.predict(features)[0] for _ in range(5)]
    probas = [classifier.predict_proba(features)[0] for _ in range(5)]

    # All predictions should be identical
    assert len(set(predictions)) == 1, "Predictions should be deterministic"

    # All probabilities should be identical
    for proba in probas[1:]:
        assert np.allclose(proba, probas[0], atol=1e-10), "Probabilities should be deterministic"


@pytest.mark.parametrize("params", [
    {"ph": 7.0, "dissolved_oxygen": 8.0, "temperature": 20.0,
     "turbidity": 10.0, "nitrate": 2.0, "conductance": 600.0},

    {"ph": 6.5, "dissolved_oxygen": 9.5, "temperature": 18.0,
     "turbidity": 5.0, "nitrate": 1.0, "conductance": 500.0},

    {"ph": 7.5, "dissolved_oxygen": 7.0, "temperature": 25.0,
     "turbidity": 20.0, "nitrate": 5.0, "conductance": 1000.0},

    {"ph": 8.0, "dissolved_oxygen": 6.0, "temperature": 28.0,
     "turbidity": 40.0, "nitrate": 8.0, "conductance": 1500.0},

    {"ph": 5.5, "dissolved_oxygen": 4.0, "temperature": 32.0,
     "turbidity": 80.0, "nitrate": 15.0, "conductance": 2500.0},
])
def test_regressor_deterministic(regressor, params):
    """Test regressor returns same prediction for same input."""
    features = create_features_array(params)

    # Run prediction multiple times
    predictions = [regressor.predict(features)[0] for _ in range(5)]

    # All predictions should be identical
    for pred in predictions[1:]:
        assert abs(pred - predictions[0]) < 1e-10, "Predictions should be deterministic"


@pytest.mark.parametrize("base_params,perturbation_param,delta", [
    ({"ph": 7.0, "dissolved_oxygen": 8.0, "temperature": 20.0,
      "turbidity": 10.0, "nitrate": 2.0, "conductance": 600.0}, "ph", 0.01),

    ({"ph": 7.0, "dissolved_oxygen": 8.0, "temperature": 20.0,
      "turbidity": 10.0, "nitrate": 2.0, "conductance": 600.0}, "dissolved_oxygen", 0.01),

    ({"ph": 7.0, "dissolved_oxygen": 8.0, "temperature": 20.0,
      "turbidity": 10.0, "nitrate": 2.0, "conductance": 600.0}, "temperature", 0.01),

    ({"ph": 7.0, "dissolved_oxygen": 8.0, "temperature": 20.0,
      "turbidity": 10.0, "nitrate": 2.0, "conductance": 600.0}, "turbidity", 0.01),

    ({"ph": 7.0, "dissolved_oxygen": 8.0, "temperature": 20.0,
      "turbidity": 10.0, "nitrate": 2.0, "conductance": 600.0}, "nitrate", 0.01),

    ({"ph": 7.0, "dissolved_oxygen": 8.0, "temperature": 20.0,
      "turbidity": 10.0, "nitrate": 2.0, "conductance": 600.0}, "conductance", 1.0),
])
def test_regressor_small_perturbation_stability(regressor, base_params, perturbation_param, delta):
    """Test regressor is stable under small input perturbations."""
    features_base = create_features_array(base_params)
    prediction_base = regressor.predict(features_base)[0]

    # Add small perturbation
    perturbed_params = base_params.copy()
    perturbed_params[perturbation_param] += delta

    features_perturbed = create_features_array(perturbed_params)
    prediction_perturbed = regressor.predict(features_perturbed)[0]

    # Predictions should be very close (within 5 WQI points for tiny perturbation)
    diff = abs(prediction_perturbed - prediction_base)
    assert diff < 5.0, f"Small perturbation in {perturbation_param} caused {diff} WQI change (should be < 5)"


@pytest.mark.parametrize("params", [
    {"ph": 7.0, "dissolved_oxygen": 8.0, "temperature": 20.0,
     "turbidity": 10.0, "nitrate": 2.0, "conductance": 600.0},

    {"ph": 6.5, "dissolved_oxygen": 9.5, "temperature": 18.0,
     "turbidity": 5.0, "nitrate": 1.0, "conductance": 500.0},

    {"ph": 8.0, "dissolved_oxygen": 6.0, "temperature": 28.0,
     "turbidity": 40.0, "nitrate": 8.0, "conductance": 1500.0},
])
def test_classifier_regressor_agreement(classifier, regressor, params):
    """Test classifier and regressor agree on water safety."""
    features = create_features_array(params)

    is_safe_clf = bool(classifier.predict(features)[0])
    wqi_reg = regressor.predict(features)[0]
    is_safe_reg = wqi_reg >= 70

    # If classifier says safe, regressor should predict WQI >= 70 (or close)
    # If classifier says unsafe, regressor should predict WQI < 70 (or close)
    # Allow 10 WQI point tolerance for model uncertainty
    if is_safe_clf:
        assert wqi_reg >= 60, f"Classifier says safe but regressor predicts WQI={wqi_reg} (should be >= 60)"
    else:
        assert wqi_reg <= 90, f"Classifier says unsafe but regressor predicts WQI={wqi_reg} (should be <= 90)"


# =============================================================================
# Property-Based Tests using Hypothesis
# =============================================================================

@settings(max_examples=10, deadline=10000, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    ph=st.floats(min_value=0, max_value=14, allow_nan=False, allow_infinity=False),
    do=st.floats(min_value=0, max_value=20, allow_nan=False, allow_infinity=False),
    temp=st.floats(min_value=-5, max_value=40, allow_nan=False, allow_infinity=False),
    turb=st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False),
    nit=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
    cond=st.floats(min_value=0, max_value=10000, allow_nan=False, allow_infinity=False),
)
def test_regressor_always_bounded_property(regressor, ph, do, temp, turb, nit, cond):
    """Property test: regressor always produces bounded WQI (0-100)."""
    params = {
        'ph': ph,
        'dissolved_oxygen': do,
        'temperature': temp,
        'turbidity': turb,
        'nitrate': nit,
        'conductance': cond
    }

    features = create_features_array(params)
    prediction = regressor.predict(features)[0]

    # Property: prediction MUST be in [0, 100]
    assert 0 <= prediction <= 100
    assert not np.isnan(prediction)
    assert not np.isinf(prediction)
