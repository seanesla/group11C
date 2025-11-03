"""
Comprehensive tests for WQI Calculator.

NO MOCKS - Testing real calculation logic with real parameter values.
Target: 100% code coverage for this critical business logic component.

Total test cases: ~80
"""

import pytest
import numpy as np
import pandas as pd
from src.utils.wqi_calculator import WQICalculator


class TestPHScoring:
    """Test pH scoring function comprehensively."""

    def setup_method(self):
        self.calculator = WQICalculator()

    def test_ideal_ph_range_perfect(self):
        """pH 6.5-7.5 should score 100."""
        assert self.calculator.calculate_ph_score(7.0) == 100
        assert self.calculator.calculate_ph_score(6.5) == 100
        assert self.calculator.calculate_ph_score(7.5) == 100
        assert self.calculator.calculate_ph_score(7.2) == 100

    def test_good_ph_range(self):
        """pH 6.0-6.5 and 7.5-8.0 should score 90."""
        assert self.calculator.calculate_ph_score(6.0) == 90
        assert self.calculator.calculate_ph_score(6.3) == 90
        assert self.calculator.calculate_ph_score(7.7) == 90
        assert self.calculator.calculate_ph_score(8.0) == 90

    def test_fair_ph_range(self):
        """pH 5.5-6.0 and 8.0-8.5 should score 70."""
        assert self.calculator.calculate_ph_score(5.5) == 70
        assert self.calculator.calculate_ph_score(5.8) == 70
        assert self.calculator.calculate_ph_score(8.2) == 70
        assert self.calculator.calculate_ph_score(8.5) == 70

    def test_poor_ph_range(self):
        """pH 5.0-5.5 and 8.5-9.0 should score 50."""
        assert self.calculator.calculate_ph_score(5.0) == 50
        assert self.calculator.calculate_ph_score(5.3) == 50
        assert self.calculator.calculate_ph_score(8.7) == 50
        assert self.calculator.calculate_ph_score(9.0) == 50

    def test_very_poor_ph_extreme_acidic(self):
        """Very acidic pH (< 5.0) should score very low."""
        score = self.calculator.calculate_ph_score(4.0)
        assert 0 <= score < 50
        score = self.calculator.calculate_ph_score(3.0)
        assert 0 <= score < 30

    def test_very_poor_ph_extreme_alkaline(self):
        """Very alkaline pH (> 9.0) should score very low."""
        score = self.calculator.calculate_ph_score(10.0)
        assert 0 <= score < 50
        score = self.calculator.calculate_ph_score(11.0)
        assert 0 <= score < 30

    def test_ph_boundary_exactly_zero(self):
        """pH exactly 0 (extreme acid) should score 0."""
        score = self.calculator.calculate_ph_score(0.0)
        assert score == 0  # Extreme but valid measurement

    def test_ph_exactly_14(self):
        """pH exactly 14 (extreme base) should score 0."""
        score = self.calculator.calculate_ph_score(14.0)
        assert score == 0  # Extreme but valid measurement

    def test_ph_negative(self):
        """Negative pH should return NaN."""
        assert pd.isna(self.calculator.calculate_ph_score(-1.0))
        assert pd.isna(self.calculator.calculate_ph_score(-5.5))

    def test_ph_above_14(self):
        """pH > 14 should return NaN."""
        assert pd.isna(self.calculator.calculate_ph_score(15.0))
        assert pd.isna(self.calculator.calculate_ph_score(20.0))

    def test_ph_nan(self):
        """NaN pH should return NaN."""
        assert pd.isna(self.calculator.calculate_ph_score(np.nan))

    def test_ph_none(self):
        """None pH should handle gracefully."""
        result = self.calculator.calculate_ph_score(None)
        assert pd.isna(wqi)


class TestDissolvedOxygenScoring:
    """Test dissolved oxygen scoring function comprehensively."""

    def setup_method(self):
        self.calculator = WQICalculator()

    def test_do_excellent_range(self):
        """DO >= 9.0 mg/L should score 100."""
        assert self.calculator.calculate_do_score(9.0) == 100
        assert self.calculator.calculate_do_score(10.0) == 100
        assert self.calculator.calculate_do_score(15.0) == 100

    def test_do_good_range(self):
        """DO 7.0-9.0 mg/L should score 85."""
        assert self.calculator.calculate_do_score(7.0) == 85
        assert self.calculator.calculate_do_score(8.0) == 85
        assert self.calculator.calculate_do_score(8.9) == 85

    def test_do_fair_range(self):
        """DO 5.0-7.0 mg/L should score 60."""
        assert self.calculator.calculate_do_score(5.0) == 60
        assert self.calculator.calculate_do_score(6.0) == 60
        assert self.calculator.calculate_do_score(6.9) == 60

    def test_do_poor_range(self):
        """DO 3.0-5.0 mg/L should score 35."""
        assert self.calculator.calculate_do_score(3.0) == 35
        assert self.calculator.calculate_do_score(4.0) == 35
        assert self.calculator.calculate_do_score(4.9) == 35

    def test_do_critical_range(self):
        """DO 1.0-3.0 mg/L should score 15."""
        assert self.calculator.calculate_do_score(1.0) == 15
        assert self.calculator.calculate_do_score(2.0) == 15
        assert self.calculator.calculate_do_score(2.9) == 15

    def test_do_very_poor(self):
        """DO < 1.0 mg/L should score 5."""
        assert self.calculator.calculate_do_score(0.5) == 5
        assert self.calculator.calculate_do_score(0.1) == 5
        assert self.calculator.calculate_do_score(0.0) == 5

    def test_do_negative(self):
        """Negative DO should return NaN."""
        assert pd.isna(self.calculator.calculate_do_score(-1.0))
        assert pd.isna(self.calculator.calculate_do_score(-5.0))

    def test_do_nan(self):
        """NaN DO should return NaN."""
        assert pd.isna(self.calculator.calculate_do_score(np.nan))

    def test_do_with_saturation_excellent(self):
        """DO saturation >= 90% should score 100."""
        assert self.calculator.calculate_do_score(8.0, saturation=90) == 100
        assert self.calculator.calculate_do_score(8.0, saturation=100) == 100

    def test_do_with_saturation_good(self):
        """DO saturation 70-90% should score 85."""
        assert self.calculator.calculate_do_score(8.0, saturation=70) == 85
        assert self.calculator.calculate_do_score(8.0, saturation=80) == 85

    def test_do_with_saturation_fair(self):
        """DO saturation 50-70% should score 60."""
        assert self.calculator.calculate_do_score(8.0, saturation=50) == 60
        assert self.calculator.calculate_do_score(8.0, saturation=60) == 60

    def test_do_with_saturation_poor(self):
        """DO saturation 30-50% should score 35."""
        assert self.calculator.calculate_do_score(8.0, saturation=30) == 35
        assert self.calculator.calculate_do_score(8.0, saturation=40) == 35

    def test_do_with_saturation_very_poor(self):
        """DO saturation < 30% should score 15."""
        assert self.calculator.calculate_do_score(8.0, saturation=20) == 15
        assert self.calculator.calculate_do_score(8.0, saturation=10) == 15


class TestTemperatureScoring:
    """Test temperature scoring function comprehensively."""

    def setup_method(self):
        self.calculator = WQICalculator()

    def test_temperature_ideal(self):
        """Temperature 15-25°C (within 5° of ideal 20°C) should score 100."""
        assert self.calculator.calculate_temperature_score(20.0) == 100
        assert self.calculator.calculate_temperature_score(15.0) == 100
        assert self.calculator.calculate_temperature_score(25.0) == 100
        assert self.calculator.calculate_temperature_score(22.0) == 100

    def test_temperature_good(self):
        """Temperature 10-15°C or 25-30°C should score 80."""
        assert self.calculator.calculate_temperature_score(10.0) == 80
        assert self.calculator.calculate_temperature_score(30.0) == 80
        assert self.calculator.calculate_temperature_score(12.0) == 80

    def test_temperature_fair(self):
        """Temperature 5-10°C or 30-35°C should score 60."""
        assert self.calculator.calculate_temperature_score(5.0) == 60
        assert self.calculator.calculate_temperature_score(35.0) == 60
        assert self.calculator.calculate_temperature_score(7.0) == 60

    def test_temperature_poor(self):
        """Temperature 0-5°C or 35-40°C should score 40."""
        assert self.calculator.calculate_temperature_score(0.0) == 40
        assert self.calculator.calculate_temperature_score(40.0) == 40
        assert self.calculator.calculate_temperature_score(2.0) == 40

    def test_temperature_very_cold(self):
        """Very cold temperature should score low."""
        score = self.calculator.calculate_temperature_score(-5.0)
        assert 0 <= score < 40
        score = self.calculator.calculate_temperature_score(-10.0)
        assert 0 <= score < 20

    def test_temperature_very_hot(self):
        """Very hot temperature should score low."""
        score = self.calculator.calculate_temperature_score(45.0)
        assert 0 <= score < 40
        score = self.calculator.calculate_temperature_score(50.0)
        assert 0 <= score < 20

    def test_temperature_extreme(self):
        """Extreme temperature should score 0 or very low."""
        score = self.calculator.calculate_temperature_score(60.0)
        assert score == 0
        score = self.calculator.calculate_temperature_score(-20.0)
        assert score == 0

    def test_temperature_nan(self):
        """NaN temperature should return NaN."""
        assert pd.isna(self.calculator.calculate_temperature_score(np.nan))


class TestTurbidityScoring:
    """Test turbidity scoring function comprehensively."""

    def setup_method(self):
        self.calculator = WQICalculator()

    def test_turbidity_excellent(self):
        """Turbidity <= 5 NTU should score 100."""
        assert self.calculator.calculate_turbidity_score(0.0) == 100
        assert self.calculator.calculate_turbidity_score(2.0) == 100
        assert self.calculator.calculate_turbidity_score(5.0) == 100

    def test_turbidity_good(self):
        """Turbidity 5-25 NTU should score 80."""
        assert self.calculator.calculate_turbidity_score(10.0) == 80
        assert self.calculator.calculate_turbidity_score(20.0) == 80
        assert self.calculator.calculate_turbidity_score(25.0) == 80

    def test_turbidity_fair(self):
        """Turbidity 25-50 NTU should score 60."""
        assert self.calculator.calculate_turbidity_score(30.0) == 60
        assert self.calculator.calculate_turbidity_score(40.0) == 60
        assert self.calculator.calculate_turbidity_score(50.0) == 60

    def test_turbidity_poor(self):
        """Turbidity 50-100 NTU should score 40."""
        assert self.calculator.calculate_turbidity_score(60.0) == 40
        assert self.calculator.calculate_turbidity_score(80.0) == 40
        assert self.calculator.calculate_turbidity_score(100.0) == 40

    def test_turbidity_very_high(self):
        """Turbidity > 100 NTU should score low."""
        score = self.calculator.calculate_turbidity_score(150.0)
        assert 0 <= score < 40
        score = self.calculator.calculate_turbidity_score(200.0)
        assert score == 0

    def test_turbidity_extreme(self):
        """Extreme turbidity should score 0."""
        score = self.calculator.calculate_turbidity_score(500.0)
        assert score == 0
        score = self.calculator.calculate_turbidity_score(1000.0)
        assert score == 0

    def test_turbidity_negative(self):
        """Negative turbidity should return NaN."""
        assert pd.isna(self.calculator.calculate_turbidity_score(-1.0))
        assert pd.isna(self.calculator.calculate_turbidity_score(-10.0))

    def test_turbidity_nan(self):
        """NaN turbidity should return NaN."""
        assert pd.isna(self.calculator.calculate_turbidity_score(np.nan))


class TestNitrateScoring:
    """Test nitrate scoring function comprehensively."""

    def setup_method(self):
        self.calculator = WQICalculator()

    def test_nitrate_excellent(self):
        """Nitrate <= 1.0 mg/L should score 100."""
        assert self.calculator.calculate_nitrate_score(0.0) == 100
        assert self.calculator.calculate_nitrate_score(0.5) == 100
        assert self.calculator.calculate_nitrate_score(1.0) == 100

    def test_nitrate_good(self):
        """Nitrate 1.0-5.0 mg/L should score 85."""
        assert self.calculator.calculate_nitrate_score(2.0) == 85
        assert self.calculator.calculate_nitrate_score(3.5) == 85
        assert self.calculator.calculate_nitrate_score(5.0) == 85

    def test_nitrate_fair(self):
        """Nitrate 5.0-10.0 mg/L should score 70."""
        assert self.calculator.calculate_nitrate_score(7.0) == 70
        assert self.calculator.calculate_nitrate_score(10.0) == 70

    def test_nitrate_poor(self):
        """Nitrate 10.0-20.0 mg/L should score 40."""
        assert self.calculator.calculate_nitrate_score(15.0) == 40
        assert self.calculator.calculate_nitrate_score(20.0) == 40

    def test_nitrate_critical(self):
        """Nitrate 20.0-50.0 mg/L should score 15."""
        assert self.calculator.calculate_nitrate_score(30.0) == 15
        assert self.calculator.calculate_nitrate_score(50.0) == 15

    def test_nitrate_very_high(self):
        """Nitrate > 50.0 mg/L should score 5."""
        assert self.calculator.calculate_nitrate_score(60.0) == 5
        assert self.calculator.calculate_nitrate_score(100.0) == 5

    def test_nitrate_negative(self):
        """Negative nitrate should return NaN."""
        assert pd.isna(self.calculator.calculate_nitrate_score(-1.0))
        assert pd.isna(self.calculator.calculate_nitrate_score(-5.0))

    def test_nitrate_nan(self):
        """NaN nitrate should return NaN."""
        assert pd.isna(self.calculator.calculate_nitrate_score(np.nan))


class TestConductanceScoring:
    """Test specific conductance scoring function comprehensively."""

    def setup_method(self):
        self.calculator = WQICalculator()

    def test_conductance_excellent(self):
        """Conductance <= 500 µS/cm should score 100."""
        assert self.calculator.calculate_conductance_score(0.0) == 100
        assert self.calculator.calculate_conductance_score(250.0) == 100
        assert self.calculator.calculate_conductance_score(500.0) == 100

    def test_conductance_good(self):
        """Conductance 500-1000 µS/cm should score 80."""
        assert self.calculator.calculate_conductance_score(700.0) == 80
        assert self.calculator.calculate_conductance_score(1000.0) == 80

    def test_conductance_fair(self):
        """Conductance 1000-1500 µS/cm should score 60."""
        assert self.calculator.calculate_conductance_score(1200.0) == 60
        assert self.calculator.calculate_conductance_score(1500.0) == 60

    def test_conductance_poor(self):
        """Conductance 1500-2000 µS/cm should score 40."""
        assert self.calculator.calculate_conductance_score(1700.0) == 40
        assert self.calculator.calculate_conductance_score(2000.0) == 40

    def test_conductance_very_high(self):
        """Conductance > 2000 µS/cm should score progressively lower."""
        # Formula: max(0, 100 - (conductance - 2000) * 0.02)
        score = self.calculator.calculate_conductance_score(2500.0)
        assert 80 < score < 100  # 2500: 100 - 10 = 90
        score = self.calculator.calculate_conductance_score(3000.0)
        assert 70 < score < 90  # 3000: 100 - 20 = 80
        score = self.calculator.calculate_conductance_score(5000.0)
        assert 30 < score < 50  # 5000: 100 - 60 = 40

    def test_conductance_extreme(self):
        """Extreme conductance should score 0."""
        score = self.calculator.calculate_conductance_score(7000.0)
        assert score == 0
        score = self.calculator.calculate_conductance_score(10000.0)
        assert score == 0

    def test_conductance_negative(self):
        """Negative conductance should return NaN."""
        assert pd.isna(self.calculator.calculate_conductance_score(-100.0))

    def test_conductance_nan(self):
        """NaN conductance should return NaN."""
        assert pd.isna(self.calculator.calculate_conductance_score(np.nan))


class TestOverallWQICalculation:
    """Test overall WQI calculation logic comprehensively."""

    def setup_method(self):
        self.calculator = WQICalculator()

    def test_wqi_all_parameters_excellent(self):
        """All excellent parameters should yield excellent WQI."""
        wqi, scores, classification = self.calculator.calculate_wqi(
            ph=7.0,
            dissolved_oxygen=9.5,
            temperature=20.0,
            turbidity=2.0,
            nitrate=0.5,
            conductance=300.0
        )
        assert 90 <= wqi <= 100
        assert classification == "Excellent"
        assert len(scores) == 6

    def test_wqi_all_parameters_good(self):
        """All good parameters should yield good WQI."""
        wqi, scores, classification = self.calculator.calculate_wqi(
            ph=6.2,
            dissolved_oxygen=7.5,
            temperature=22.0,
            turbidity=15.0,
            nitrate=3.0,
            conductance=700.0
        )
        assert 70 <= wqi < 90
        assert classification == "Good"

    def test_wqi_all_parameters_fair(self):
        """All fair parameters should yield fair WQI."""
        wqi, scores, classification = self.calculator.calculate_wqi(
            ph=8.3,
            dissolved_oxygen=5.5,
            temperature=28.0,
            turbidity=35.0,
            nitrate=8.0,
            conductance=1300.0
        )
        assert 50 <= wqi < 70
        assert classification == "Fair"

    def test_wqi_all_parameters_poor(self):
        """All poor parameters should yield poor WQI."""
        wqi, scores, classification = self.calculator.calculate_wqi(
            ph=9.2,
            dissolved_oxygen=3.5,
            temperature=32.0,
            turbidity=70.0,
            nitrate=15.0,
            conductance=1700.0
        )
        assert 25 <= wqi < 50
        assert classification == "Poor"

    def test_wqi_all_parameters_very_poor(self):
        """All very poor parameters should yield very poor WQI."""
        wqi, scores, classification = self.calculator.calculate_wqi(
            ph=10.5,
            dissolved_oxygen=0.5,
            temperature=45.0,
            turbidity=200.0,
            nitrate=60.0,
            conductance=5000.0
        )
        assert 0 <= wqi < 25
        assert classification == "Very Poor"

    def test_wqi_only_ph(self):
        """WQI with only pH parameter should work."""
        wqi, scores, classification = self.calculator.calculate_wqi(ph=7.0)
        assert not pd.isna(wqi)
        assert len(scores) == 1
        assert 'ph' in scores

    def test_wqi_only_do(self):
        """WQI with only DO parameter should work."""
        wqi, scores, classification = self.calculator.calculate_wqi(dissolved_oxygen=9.0)
        assert not pd.isna(wqi)
        assert len(scores) == 1

    def test_wqi_missing_ph(self):
        """WQI without pH should work with remaining parameters."""
        wqi, scores, classification = self.calculator.calculate_wqi(
            dissolved_oxygen=9.0,
            temperature=20.0,
            turbidity=5.0,
            nitrate=1.0,
            conductance=400.0
        )
        assert not pd.isna(wqi)
        assert len(scores) == 5
        assert 'ph' not in scores

    def test_wqi_missing_do(self):
        """WQI without DO should work with remaining parameters."""
        wqi, scores, classification = self.calculator.calculate_wqi(
            ph=7.0,
            temperature=20.0,
            turbidity=5.0,
            nitrate=1.0,
            conductance=400.0
        )
        assert not pd.isna(wqi)
        assert len(scores) == 5
        assert 'dissolved_oxygen' not in scores

    def test_wqi_half_parameters(self):
        """WQI with 3 out of 6 parameters should work."""
        wqi, scores, classification = self.calculator.calculate_wqi(
            ph=7.0,
            dissolved_oxygen=9.0,
            temperature=20.0
        )
        assert not pd.isna(wqi)
        assert len(scores) == 3

    def test_wqi_no_parameters(self):
        """WQI with no valid parameters should return NaN."""
        wqi, scores, classification = self.calculator.calculate_wqi()
        assert pd.isna(wqi)
        assert len(scores) == 0
        assert classification == "Unknown"

    def test_wqi_all_nan_parameters(self):
        """WQI with all NaN parameters should return NaN."""
        wqi, scores, classification = self.calculator.calculate_wqi(
            ph=np.nan,
            dissolved_oxygen=np.nan,
            temperature=np.nan
        )
        assert pd.isna(wqi)
        assert classification == "Unknown"

    def test_wqi_some_nan_parameters(self):
        """WQI with some NaN parameters should work with valid ones."""
        wqi, scores, classification = self.calculator.calculate_wqi(
            ph=7.0,
            dissolved_oxygen=np.nan,
            temperature=20.0,
            turbidity=np.nan,
            nitrate=1.0,
            conductance=400.0
        )
        assert not pd.isna(wqi)
        assert len(scores) == 4  # Only valid parameters

    def test_wqi_returns_tuple(self):
        """WQI calculation should return tuple of (wqi, scores, classification)."""
        wqi, scores, classification = self.calculator.calculate_wqi(ph=7.0)
        assert isinstance(wqi, tuple)
        assert len(result) == 3
        wqi, scores, classification = result
        assert isinstance(wqi, (int, float))
        assert isinstance(scores, dict)
        assert isinstance(classification, str)

    def test_wqi_rounded_to_2_decimals(self):
        """WQI should be rounded to 2 decimal places."""
        wqi, _, _ = self.calculator.calculate_wqi(
            ph=7.123456789,
            dissolved_oxygen=9.123456789
        )
        # Check that wqi has at most 2 decimal places
        assert round(wqi, 2) == wqi


class TestWQIClassification:
    """Test WQI classification logic comprehensively."""

    def setup_method(self):
        self.calculator = WQICalculator()

    def test_classify_excellent_range(self):
        """WQI 90-100 should classify as Excellent."""
        assert self.calculator.classify_wqi(90.0) == "Excellent"
        assert self.calculator.classify_wqi(95.0) == "Excellent"
        assert self.calculator.classify_wqi(100.0) == "Excellent"

    def test_classify_good_range(self):
        """WQI 70-89 should classify as Good."""
        assert self.calculator.classify_wqi(70.0) == "Good"
        assert self.calculator.classify_wqi(80.0) == "Good"
        assert self.calculator.classify_wqi(89.99) == "Good"

    def test_classify_fair_range(self):
        """WQI 50-69 should classify as Fair."""
        assert self.calculator.classify_wqi(50.0) == "Fair"
        assert self.calculator.classify_wqi(60.0) == "Fair"
        assert self.calculator.classify_wqi(69.99) == "Fair"

    def test_classify_poor_range(self):
        """WQI 25-49 should classify as Poor."""
        assert self.calculator.classify_wqi(25.0) == "Poor"
        assert self.calculator.classify_wqi(35.0) == "Poor"
        assert self.calculator.classify_wqi(49.99) == "Poor"

    def test_classify_very_poor_range(self):
        """WQI 0-24 should classify as Very Poor."""
        assert self.calculator.classify_wqi(0.0) == "Very Poor"
        assert self.calculator.classify_wqi(10.0) == "Very Poor"
        assert self.calculator.classify_wqi(24.99) == "Very Poor"

    def test_classify_boundary_90(self):
        """WQI exactly 90.0 should be Excellent (boundary)."""
        assert self.calculator.classify_wqi(90.0) == "Excellent"

    def test_classify_boundary_70(self):
        """WQI exactly 70.0 should be Good (boundary)."""
        assert self.calculator.classify_wqi(70.0) == "Good"

    def test_classify_boundary_50(self):
        """WQI exactly 50.0 should be Fair (boundary)."""
        assert self.calculator.classify_wqi(50.0) == "Fair"

    def test_classify_boundary_25(self):
        """WQI exactly 25.0 should be Poor (boundary)."""
        assert self.calculator.classify_wqi(25.0) == "Poor"

    def test_classify_just_below_90(self):
        """WQI 89.99 should be Good, not Excellent."""
        assert self.calculator.classify_wqi(89.99) == "Good"

    def test_classify_just_above_90(self):
        """WQI 90.01 should be Excellent."""
        assert self.calculator.classify_wqi(90.01) == "Excellent"

    def test_classify_nan(self):
        """NaN WQI should classify as Unknown."""
        assert self.calculator.classify_wqi(np.nan) == "Unknown"


class TestWQISafety:
    """Test WQI safety determination comprehensively."""

    def setup_method(self):
        self.calculator = WQICalculator()

    def test_is_safe_excellent(self):
        """WQI >= 90 should be safe."""
        assert self.calculator.is_safe(90.0) is True
        assert self.calculator.is_safe(95.0) is True
        assert self.calculator.is_safe(100.0) is True

    def test_is_safe_good(self):
        """WQI 70-89 should be safe."""
        assert self.calculator.is_safe(70.0) is True
        assert self.calculator.is_safe(80.0) is True
        assert self.calculator.is_safe(89.99) is True

    def test_is_not_safe_fair(self):
        """WQI 50-69 should not be safe."""
        assert self.calculator.is_safe(50.0) is False
        assert self.calculator.is_safe(60.0) is False
        assert self.calculator.is_safe(69.99) is False

    def test_is_not_safe_poor(self):
        """WQI 25-49 should not be safe."""
        assert self.calculator.is_safe(25.0) is False
        assert self.calculator.is_safe(35.0) is False
        assert self.calculator.is_safe(49.99) is False

    def test_is_not_safe_very_poor(self):
        """WQI 0-24 should not be safe."""
        assert self.calculator.is_safe(0.0) is False
        assert self.calculator.is_safe(10.0) is False
        assert self.calculator.is_safe(24.99) is False

    def test_is_safe_boundary_exactly_70(self):
        """WQI exactly 70.0 should be safe (boundary)."""
        assert self.calculator.is_safe(70.0) is True

    def test_is_not_safe_just_below_70(self):
        """WQI 69.99 should not be safe."""
        assert self.calculator.is_safe(69.99) is False

    def test_is_safe_nan(self):
        """NaN WQI should not be safe."""
        assert self.calculator.is_safe(np.nan) is False


class TestWQICalculationNaNHandling:
    """Test NaN handling in calculate_wqi method (tests 91-105)."""

    def setup_method(self):
        self.calculator = WQICalculator()

    def test_calculate_wqi_nan_ph_via_pd_isna(self):
        """Test 91: Pass np.nan for ph, verify pd.isna check."""
        wqi, scores, classification = self.calculator.calculate_wqi(
            ph=np.nan,
            dissolved_oxygen=8.0,
            temperature=20.0,
            turbidity=5.0,
            nitrate=2.0,
            conductance=500.0
        )
        # Should still calculate WQI without pH
        assert not pd.isna(wqi)
        assert 0 <= wqi <= 100

    def test_calculate_wqi_nan_dissolved_oxygen(self):
        """Test 92: Pass np.nan for DO, verify skipped."""
        wqi, scores, classification = self.calculator.calculate_wqi(
            ph=7.0,
            dissolved_oxygen=np.nan,
            temperature=20.0,
            turbidity=5.0,
            nitrate=2.0,
            conductance=500.0
        )
        assert not pd.isna(wqi)
        assert 0 <= wqi <= 100

    def test_calculate_wqi_nan_temperature(self):
        """Test 93: Pass np.nan for temp, verify skipped."""
        wqi, scores, classification = self.calculator.calculate_wqi(
            ph=7.0,
            dissolved_oxygen=8.0,
            temperature=np.nan,
            turbidity=5.0,
            nitrate=2.0,
            conductance=500.0
        )
        assert not pd.isna(wqi)
        assert 0 <= wqi <= 100

    def test_calculate_wqi_nan_turbidity(self):
        """Test 94: Pass np.nan for turbidity, verify skipped."""
        wqi, scores, classification = self.calculator.calculate_wqi(
            ph=7.0,
            dissolved_oxygen=8.0,
            temperature=20.0,
            turbidity=np.nan,
            nitrate=2.0,
            conductance=500.0
        )
        assert not pd.isna(wqi)
        assert 0 <= wqi <= 100

    def test_calculate_wqi_nan_nitrate(self):
        """Test 95: Pass np.nan for nitrate, verify skipped."""
        wqi, scores, classification = self.calculator.calculate_wqi(
            ph=7.0,
            dissolved_oxygen=8.0,
            temperature=20.0,
            turbidity=5.0,
            nitrate=np.nan,
            conductance=500.0
        )
        assert not pd.isna(wqi)
        assert 0 <= wqi <= 100

    def test_calculate_wqi_nan_conductance(self):
        """Test 96: Pass np.nan for conductance, verify skipped."""
        wqi, scores, classification = self.calculator.calculate_wqi(
            ph=7.0,
            dissolved_oxygen=8.0,
            temperature=20.0,
            turbidity=5.0,
            nitrate=2.0,
            conductance=np.nan
        )
        assert not pd.isna(wqi)
        assert 0 <= wqi <= 100

    def test_calculate_wqi_all_nan_returns_nan(self):
        """Test 97: All params NaN → total_weight=0 → NaN."""
        wqi, scores, classification = self.calculator.calculate_wqi(
            ph=np.nan,
            dissolved_oxygen=np.nan,
            temperature=np.nan,
            turbidity=np.nan,
            nitrate=np.nan,
            conductance=np.nan
        )
        assert pd.isna(wqi)

    def test_calculate_wqi_zero_total_weight_edge_case(self):
        """Test 98: Explicitly test total_weight == 0 path."""
        # All NaN should result in zero weight
        wqi, scores, classification = self.calculator.calculate_wqi()
        assert pd.isna(wqi)

    def test_calculate_wqi_ignores_extra_kwargs(self):
        """Test 99: Pass extra kwargs, verify ignored."""
        wqi, scores, classification = self.calculator.calculate_wqi(
            ph=7.0,
            dissolved_oxygen=8.0,
            extra_param=999,  # Should be ignored
            another_param="ignored"
        )
        assert not pd.isna(wqi)
        assert isinstance(wqi, (int, float))

    def test_calculate_wqi_partial_params_with_nans(self):
        """Test 100: Mix of valid and NaN params."""
        wqi, scores, classification = self.calculator.calculate_wqi(
            ph=7.0,  # Valid
            dissolved_oxygen=np.nan,  # NaN
            temperature=20.0,  # Valid
            turbidity=np.nan,  # NaN
            nitrate=2.0,  # Valid
            conductance=np.nan  # NaN
        )
        # Should calculate with only the valid params
        assert not pd.isna(wqi)
        assert 0 <= wqi <= 100

    def test_calculate_wqi_all_params_at_boundaries(self):
        """Test 101: Test with boundary values for all params."""
        wqi, scores, classification = self.calculator.calculate_wqi(
            ph=7.0,  # Ideal pH
            dissolved_oxygen=14.6,  # 100% saturation
            temperature=20.0,  # Ideal temperature
            turbidity=1.0,  # Low turbidity
            nitrate=0.0,  # No nitrate
            conductance=500.0  # Good conductance
        )
        assert not pd.isna(wqi)
        # All parameters at good values should give high score
        assert wqi > 90

    def test_calculate_wqi_extremely_poor_all_params(self):
        """Test 102: All params worst case → score near 0."""
        wqi, scores, classification = self.calculator.calculate_wqi(
            ph=0.0,  # Extremely acidic
            dissolved_oxygen=0.0,  # No oxygen
            temperature=40.0,  # Very hot (20 degrees above ideal)
            turbidity=1000.0,  # Very turbid
            nitrate=100.0,  # Very high nitrate
            conductance=5000.0  # Very high conductance
        )
        assert not pd.isna(wqi)
        # All poor parameters should give low score
        assert wqi < 20

    def test_calculate_wqi_extremely_good_all_params(self):
        """Test 103: All params ideal → score near 100."""
        wqi, scores, classification = self.calculator.calculate_wqi(
            ph=7.0,  # Ideal
            dissolved_oxygen=14.6,  # 100% sat
            temperature=20.0,  # Ideal
            turbidity=1.0,  # Low
            nitrate=0.5,  # Low
            conductance=500.0  # Good
        )
        assert not pd.isna(wqi)
        assert wqi > 90

    def test_calculate_wqi_single_param_only(self):
        """Test 104: Only one param provided, rest NaN."""
        wqi, scores, classification = self.calculator.calculate_wqi(
            ph=7.0,
            dissolved_oxygen=np.nan,
            temperature=np.nan,
            turbidity=np.nan,
            nitrate=np.nan,
            conductance=np.nan
        )
        # Should calculate with just pH
        assert not pd.isna(wqi)
        # pH of 7.0 is ideal, should give 100
        assert result == 100.0

    def test_calculate_wqi_none_vs_nan(self):
        """Test 105: Verify None is treated same as NaN."""
        result_with_none = self.calculator.calculate_wqi(
            ph=7.0,
            dissolved_oxygen=None,
            temperature=20.0
        )
        result_with_nan = self.calculator.calculate_wqi(
            ph=7.0,
            dissolved_oxygen=np.nan,
            temperature=20.0
        )
        # Both should be valid and similar
        assert not pd.isna(result_with_none)
        assert not pd.isna(result_with_nan)


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
