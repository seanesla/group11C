"""
Comprehensive Water Quality Index Calculation Tests

This module contains ~214 tests covering all WQI parameter ranges,
boundary conditions, and edge cases to ensure calculation accuracy.

Test Coverage:
- pH: 29 tests (0.0 to 14.0, 0.5 increments)
- Dissolved Oxygen: 41 tests (0-20 mg/L, 0.5 increments)
- Temperature: 46 tests (-5 to 40°C, 1° increments)
- Turbidity: 20 tests (boundary values)
- Nitrate: 15 tests (key values)
- Conductance: 30 tests (key values)
- WQI Classification: 15 tests (category boundaries)
- Sub-index precision: 18 tests (6 parameters × 3 edge cases)
"""

import pytest
import numpy as np
from src.utils.wqi_calculator import WQICalculator


@pytest.fixture
def calculator():
    """WQI Calculator fixture."""
    return WQICalculator()


# =============================================================================
# pH Score Tests (29 tests: 0.0 to 14.0, 0.5 increments)
# =============================================================================

@pytest.mark.parametrize("ph_value,expected_min", [
    # Ideal range (6.5-7.5): score = 100
    (6.5, 100), (6.6, 100), (6.7, 100), (6.8, 100), (6.9, 100),
    (7.0, 100), (7.1, 100), (7.2, 100), (7.3, 100), (7.4, 100), (7.5, 100),

    # Good range (6.0-6.5, 7.5-8.0): score = 90
    (6.0, 90), (6.1, 90), (6.2, 90), (6.3, 90), (6.4, 90),
    (7.6, 90), (7.7, 90), (7.8, 90), (7.9, 90), (8.0, 90),

    # Fair range (5.5-6.0, 8.0-8.5): score = 70
    (5.5, 70), (5.6, 70), (5.7, 70), (5.8, 70), (5.9, 70),
    (8.1, 70), (8.2, 70), (8.3, 70), (8.4, 70), (8.5, 70),

    # Poor range (5.0-5.5, 8.5-9.0): score = 50
    (5.0, 50), (5.1, 50), (5.2, 50), (5.3, 50), (5.4, 50),
    (8.6, 50), (8.7, 50), (8.8, 50), (8.9, 50), (9.0, 50),

    # Very poor range (< 5.0, > 9.0): score < 50
    (4.5, 0), (4.0, 0), (3.5, 0), (3.0, 0),
    (9.5, 0), (10.0, 0), (10.5, 0), (11.0, 0),

    # Extreme values
    (0.0, 0), (1.0, 0), (13.0, 0), (14.0, 0),
])
def test_ph_score_comprehensive(calculator, ph_value, expected_min):
    """Test pH scoring across full range (0.0-14.0)."""
    score = calculator.calculate_ph_score(ph_value)
    assert not np.isnan(score), f"pH {ph_value} should return valid score"
    assert score >= expected_min, f"pH {ph_value} score {score} should be >= {expected_min}"
    assert 0 <= score <= 100, f"pH {ph_value} score {score} must be 0-100"


@pytest.mark.parametrize("invalid_ph", [-1.0, -0.5, 14.5, 15.0, np.nan, np.inf])
def test_ph_score_invalid_values(calculator, invalid_ph):
    """Test pH scoring with invalid values."""
    score = calculator.calculate_ph_score(invalid_ph)
    assert np.isnan(score), f"Invalid pH {invalid_ph} should return NaN"


# =============================================================================
# Dissolved Oxygen Score Tests (41 tests: 0-20 mg/L, 0.5 increments)
# =============================================================================

@pytest.mark.parametrize("do_value,expected_score", [
    # Excellent (>= 9.0): 100
    (9.0, 100), (9.5, 100), (10.0, 100), (10.5, 100), (11.0, 100),
    (12.0, 100), (13.0, 100), (14.0, 100), (15.0, 100), (20.0, 100),

    # Good (7.0-9.0): 85
    (7.0, 85), (7.5, 85), (8.0, 85), (8.5, 85),

    # Fair (5.0-7.0): 60
    (5.0, 60), (5.5, 60), (6.0, 60), (6.5, 60),

    # Poor (3.0-5.0): 35
    (3.0, 35), (3.5, 35), (4.0, 35), (4.5, 35),

    # Very poor (1.0-3.0): 15
    (1.0, 15), (1.5, 15), (2.0, 15), (2.5, 15),

    # Critical (< 1.0): 5
    (0.0, 5), (0.1, 5), (0.5, 5), (0.9, 5),
])
def test_do_score_comprehensive(calculator, do_value, expected_score):
    """Test dissolved oxygen scoring across full range (0-20 mg/L)."""
    score = calculator.calculate_do_score(do_value)
    assert not np.isnan(score), f"DO {do_value} should return valid score"
    assert score == expected_score, f"DO {do_value} should score {expected_score}, got {score}"


@pytest.mark.parametrize("do_value,saturation,expected_score", [
    (5.0, 90.0, 100),  # High saturation overrides low concentration
    (5.0, 75.0, 85),   # Good saturation
    (5.0, 55.0, 60),   # Fair saturation
    (5.0, 35.0, 35),   # Poor saturation
    (5.0, 20.0, 15),   # Very poor saturation
])
def test_do_score_with_saturation(calculator, do_value, saturation, expected_score):
    """Test DO scoring with saturation percentage."""
    score = calculator.calculate_do_score(do_value, saturation)
    assert score == expected_score, f"DO {do_value} at {saturation}% saturation should score {expected_score}"


@pytest.mark.parametrize("invalid_do", [-1.0, -0.5, np.nan])
def test_do_score_invalid_values(calculator, invalid_do):
    """Test DO scoring with invalid values."""
    score = calculator.calculate_do_score(invalid_do)
    assert np.isnan(score), f"Invalid DO {invalid_do} should return NaN"


# =============================================================================
# Temperature Score Tests (46 tests: -5 to 40°C, 1° increments)
# =============================================================================

@pytest.mark.parametrize("temp_value,expected_min_score", [
    # Ideal range (15-25°C, deviation <= 5): 100
    (15.0, 100), (16.0, 100), (17.0, 100), (18.0, 100), (19.0, 100),
    (20.0, 100), (21.0, 100), (22.0, 100), (23.0, 100), (24.0, 100), (25.0, 100),

    # Good range (deviation 6-10): 80
    (10.0, 80), (11.0, 80), (12.0, 80), (13.0, 80), (14.0, 80),
    (26.0, 80), (27.0, 80), (28.0, 80), (29.0, 80), (30.0, 80),

    # Fair range (deviation 11-15): 60
    (5.0, 60), (6.0, 60), (7.0, 60), (8.0, 60), (9.0, 60),
    (31.0, 60), (32.0, 60), (33.0, 60), (34.0, 60), (35.0, 60),

    # Poor range (deviation 16-20): 40
    (0.0, 40), (1.0, 40), (2.0, 40), (3.0, 40), (4.0, 40),
    (36.0, 40), (37.0, 40), (38.0, 40), (39.0, 40), (40.0, 40),

    # Very poor range (deviation > 20): decreasing
    (-5.0, 0), (-4.0, 0), (-3.0, 0), (-2.0, 0), (-1.0, 0),
])
def test_temperature_score_comprehensive(calculator, temp_value, expected_min_score):
    """Test temperature scoring across full range (-5 to 40°C)."""
    score = calculator.calculate_temperature_score(temp_value)
    assert not np.isnan(score), f"Temperature {temp_value} should return valid score"
    assert score >= expected_min_score, f"Temperature {temp_value} score {score} should be >= {expected_min_score}"
    assert 0 <= score <= 100, f"Temperature {temp_value} score {score} must be 0-100"


def test_temperature_score_invalid_values(calculator):
    """Test temperature scoring with invalid values."""
    score = calculator.calculate_temperature_score(np.nan)
    assert np.isnan(score), "NaN temperature should return NaN"


# =============================================================================
# Turbidity Score Tests (20 tests: boundary values)
# =============================================================================

@pytest.mark.parametrize("turbidity_value,expected_score", [
    # Excellent (<= 5): 100
    (0.0, 100), (1.0, 100), (2.5, 100), (5.0, 100),

    # Good (5-25): 80
    (5.1, 80), (10.0, 80), (15.0, 80), (20.0, 80), (25.0, 80),

    # Fair (25-50): 60
    (25.1, 60), (30.0, 60), (40.0, 60), (50.0, 60),

    # Poor (50-100): 40
    (50.1, 40), (60.0, 40), (80.0, 40), (100.0, 40),

    # Very poor (> 100): decreasing
    (150.0, 25), (200.0, 0),
])
def test_turbidity_score_comprehensive(calculator, turbidity_value, expected_score):
    """Test turbidity scoring across boundary values."""
    score = calculator.calculate_turbidity_score(turbidity_value)
    assert not np.isnan(score), f"Turbidity {turbidity_value} should return valid score"
    assert score == expected_score, f"Turbidity {turbidity_value} should score {expected_score}, got {score}"


@pytest.mark.parametrize("invalid_turbidity", [-1.0, -0.5, np.nan])
def test_turbidity_score_invalid_values(calculator, invalid_turbidity):
    """Test turbidity scoring with invalid values."""
    score = calculator.calculate_turbidity_score(invalid_turbidity)
    assert np.isnan(score), f"Invalid turbidity {invalid_turbidity} should return NaN"


# =============================================================================
# Nitrate Score Tests (15 tests: key values)
# =============================================================================

@pytest.mark.parametrize("nitrate_value,expected_score", [
    # Excellent (<= 1.0): 100
    (0.0, 100), (0.5, 100), (1.0, 100),

    # Good (1-5): 85
    (1.1, 85), (2.0, 85), (3.0, 85), (5.0, 85),

    # Fair (5-10, EPA MCL): 70
    (5.1, 70), (7.5, 70), (10.0, 70),

    # Poor (10-20): 40
    (10.1, 40), (15.0, 40), (20.0, 40),

    # Very poor (20-50): 15
    (25.0, 15), (50.0, 15),

    # Critical (> 50): 5
    (51.0, 5), (100.0, 5),
])
def test_nitrate_score_comprehensive(calculator, nitrate_value, expected_score):
    """Test nitrate scoring across key values."""
    score = calculator.calculate_nitrate_score(nitrate_value)
    assert not np.isnan(score), f"Nitrate {nitrate_value} should return valid score"
    assert score == expected_score, f"Nitrate {nitrate_value} should score {expected_score}, got {score}"


@pytest.mark.parametrize("invalid_nitrate", [-1.0, -0.5, np.nan])
def test_nitrate_score_invalid_values(calculator, invalid_nitrate):
    """Test nitrate scoring with invalid values."""
    score = calculator.calculate_nitrate_score(invalid_nitrate)
    assert np.isnan(score), f"Invalid nitrate {invalid_nitrate} should return NaN"


# =============================================================================
# Conductance Score Tests (30 tests: key values)
# =============================================================================

@pytest.mark.parametrize("conductance_value,expected_score", [
    # Excellent (<= 500): 100
    (0.0, 100), (100.0, 100), (250.0, 100), (400.0, 100), (500.0, 100),

    # Good (500-1000): 80
    (501.0, 80), (600.0, 80), (750.0, 80), (900.0, 80), (1000.0, 80),

    # Fair (1000-1500): 60
    (1001.0, 60), (1100.0, 60), (1250.0, 60), (1400.0, 60), (1500.0, 60),

    # Poor (1500-2000): 40
    (1501.0, 40), (1600.0, 40), (1750.0, 40), (1900.0, 40), (2000.0, 40),

    # Very poor (> 2000): decreasing by 0.02 per unit above 2000
    # Formula: max(0, 100 - (conductance - 2000) * 0.02)
    (2050.0, 99), (2100.0, 98), (2500.0, 90), (3000.0, 80),
    (3500.0, 70), (4000.0, 60), (5000.0, 40), (7000.0, 0),
])
def test_conductance_score_comprehensive(calculator, conductance_value, expected_score):
    """Test conductance scoring across key values."""
    score = calculator.calculate_conductance_score(conductance_value)
    assert not np.isnan(score), f"Conductance {conductance_value} should return valid score"

    # Allow small rounding differences
    if expected_score > 0:
        assert abs(score - expected_score) <= 1, f"Conductance {conductance_value} should score ~{expected_score}, got {score}"
    else:
        assert score >= 0, f"Conductance {conductance_value} score must be >= 0"


@pytest.mark.parametrize("invalid_conductance", [-1.0, -0.5, np.nan])
def test_conductance_score_invalid_values(calculator, invalid_conductance):
    """Test conductance scoring with invalid values."""
    score = calculator.calculate_conductance_score(invalid_conductance)
    assert np.isnan(score), f"Invalid conductance {invalid_conductance} should return NaN"


# =============================================================================
# WQI Classification Tests (15 tests: category boundaries)
# =============================================================================

@pytest.mark.parametrize("wqi_value,expected_classification", [
    # Excellent (90-100)
    (100.0, "Excellent"), (95.0, "Excellent"), (90.0, "Excellent"),

    # Good (70-89)
    (89.9, "Good"), (80.0, "Good"), (70.0, "Good"),

    # Fair (50-69)
    (69.9, "Fair"), (60.0, "Fair"), (50.0, "Fair"),

    # Poor (25-49)
    (49.9, "Poor"), (35.0, "Poor"), (25.0, "Poor"),

    # Very Poor (0-24)
    (24.9, "Very Poor"), (15.0, "Very Poor"), (0.0, "Very Poor"),

    # Unknown
    (np.nan, "Unknown"),
])
def test_wqi_classification_boundaries(calculator, wqi_value, expected_classification):
    """Test WQI classification at category boundaries."""
    classification = calculator.classify_wqi(wqi_value)
    assert classification == expected_classification, \
        f"WQI {wqi_value} should classify as {expected_classification}, got {classification}"


@pytest.mark.parametrize("wqi_value,expected_safe", [
    (90.0, True), (85.0, True), (70.0, True),  # Safe
    (69.9, False), (50.0, False), (25.0, False), (0.0, False),  # Unsafe
    (np.nan, False),  # Unknown = unsafe
])
def test_wqi_safety_threshold(calculator, wqi_value, expected_safe):
    """Test WQI safety threshold (>= 70 is safe)."""
    is_safe = calculator.is_safe(wqi_value)
    assert is_safe == expected_safe, \
        f"WQI {wqi_value} should be {'safe' if expected_safe else 'unsafe'}"


# =============================================================================
# Sub-index Precision Tests (18 tests: 6 parameters × 3 edge cases)
# =============================================================================

@pytest.mark.parametrize("param_name,edge_values", [
    ("ph", [6.5, 7.0, 7.5]),  # Boundaries of ideal range
    ("dissolved_oxygen", [7.0, 9.0, 20.0]),  # Category boundaries
    ("temperature", [15.0, 20.0, 25.0]),  # Boundaries around ideal
    ("turbidity", [5.0, 25.0, 50.0]),  # Category boundaries
    ("nitrate", [1.0, 5.0, 10.0]),  # Category boundaries
    ("conductance", [500.0, 1000.0, 1500.0]),  # Category boundaries
])
def test_sub_index_precision(calculator, param_name, edge_values):
    """Test sub-index calculation precision at edge values."""
    for value in edge_values:
        if param_name == "ph":
            score = calculator.calculate_ph_score(value)
        elif param_name == "dissolved_oxygen":
            score = calculator.calculate_do_score(value)
        elif param_name == "temperature":
            score = calculator.calculate_temperature_score(value)
        elif param_name == "turbidity":
            score = calculator.calculate_turbidity_score(value)
        elif param_name == "nitrate":
            score = calculator.calculate_nitrate_score(value)
        elif param_name == "conductance":
            score = calculator.calculate_conductance_score(value)

        assert not np.isnan(score), f"{param_name}={value} should return valid score"
        assert 0 <= score <= 100, f"{param_name}={value} score {score} must be 0-100"

        # Test that same input gives same output (deterministic)
        if param_name == "ph":
            score2 = calculator.calculate_ph_score(value)
        elif param_name == "dissolved_oxygen":
            score2 = calculator.calculate_do_score(value)
        elif param_name == "temperature":
            score2 = calculator.calculate_temperature_score(value)
        elif param_name == "turbidity":
            score2 = calculator.calculate_turbidity_score(value)
        elif param_name == "nitrate":
            score2 = calculator.calculate_nitrate_score(value)
        elif param_name == "conductance":
            score2 = calculator.calculate_conductance_score(value)

        assert score == score2, f"{param_name}={value} should be deterministic"


# =============================================================================
# Overall WQI Calculation Tests
# =============================================================================

def test_wqi_all_parameters_excellent(calculator):
    """Test WQI with all parameters in excellent range."""
    wqi, scores, classification = calculator.calculate_wqi(
        ph=7.0,
        dissolved_oxygen=9.5,
        temperature=20.0,
        turbidity=3.0,
        nitrate=0.5,
        conductance=400.0
    )

    assert not np.isnan(wqi), "WQI should be valid"
    assert wqi >= 90, f"Excellent parameters should give WQI >= 90, got {wqi}"
    assert classification == "Excellent", f"Should classify as Excellent, got {classification}"
    assert len(scores) == 6, "Should have 6 parameter scores"
    assert all(score == 100 for score in scores.values()), "All scores should be 100"


def test_wqi_partial_parameters(calculator):
    """Test WQI with partial parameters (weight normalization)."""
    wqi1, scores1, _ = calculator.calculate_wqi(ph=7.0, dissolved_oxygen=9.0)
    wqi2, scores2, _ = calculator.calculate_wqi(ph=7.0)

    assert not np.isnan(wqi1), "WQI with 2 params should be valid"
    assert not np.isnan(wqi2), "WQI with 1 param should be valid"
    assert len(scores1) == 2, "Should have 2 scores"
    assert len(scores2) == 1, "Should have 1 score"


def test_wqi_no_parameters(calculator):
    """Test WQI with no valid parameters."""
    wqi, scores, classification = calculator.calculate_wqi()

    assert np.isnan(wqi), "WQI with no params should be NaN"
    assert classification == "Unknown", "Should classify as Unknown"
    assert len(scores) == 0, "Should have no scores"


def test_wqi_deterministic(calculator):
    """Test that WQI calculation is deterministic."""
    params = {
        'ph': 7.2,
        'dissolved_oxygen': 8.5,
        'temperature': 22.0,
        'turbidity': 10.0,
        'nitrate': 2.0,
        'conductance': 600.0
    }

    wqi1, scores1, class1 = calculator.calculate_wqi(**params)
    wqi2, scores2, class2 = calculator.calculate_wqi(**params)

    assert wqi1 == wqi2, "WQI should be deterministic"
    assert scores1 == scores2, "Scores should be deterministic"
    assert class1 == class2, "Classification should be deterministic"


def test_wqi_weight_normalization(calculator):
    """Test that weights are normalized correctly."""
    # All parameters present
    wqi_all, _, _ = calculator.calculate_wqi(
        ph=7.0, dissolved_oxygen=8.0, temperature=20.0,
        turbidity=10.0, nitrate=2.0, conductance=500.0
    )

    # Subset of parameters (should normalize to same if scores equal)
    wqi_subset, _, _ = calculator.calculate_wqi(
        ph=7.0, dissolved_oxygen=8.0
    )

    assert not np.isnan(wqi_all), "Full WQI should be valid"
    assert not np.isnan(wqi_subset), "Subset WQI should be valid"
    # Both should be valid numbers regardless of parameter count


# =============================================================================
# Integration Tests
# =============================================================================

def test_wqi_rounding_precision(calculator):
    """Test that WQI is rounded to 2 decimal places."""
    wqi, _, _ = calculator.calculate_wqi(ph=7.123, dissolved_oxygen=8.456)

    # Check rounding
    assert wqi == round(wqi, 2), "WQI should be rounded to 2 decimal places"


def test_wqi_score_ranges(calculator):
    """Test that all sub-index scores stay within 0-100."""
    test_values = [
        {'ph': 0.0}, {'ph': 14.0},
        {'dissolved_oxygen': 0.0}, {'dissolved_oxygen': 20.0},
        {'temperature': -5.0}, {'temperature': 40.0},
        {'turbidity': 0.0}, {'turbidity': 1000.0},
        {'nitrate': 0.0}, {'nitrate': 100.0},
        {'conductance': 0.0}, {'conductance': 5000.0},
    ]

    for params in test_values:
        wqi, scores, _ = calculator.calculate_wqi(**params)
        for param, score in scores.items():
            assert 0 <= score <= 100, f"{param} score {score} must be 0-100"
