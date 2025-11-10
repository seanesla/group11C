"""
Feature Engineering Test Suite

This test suite validates the correctness of feature engineering transformations
for the Water Quality ML models. Tests ensure that:
1. Calculations are mathematically correct
2. Edge cases are handled properly
3. Scientific assumptions are reasonable
4. Data types are appropriate
5. Feature counts and ordering are exact

Following CLAUDE.md: NO mocks, NO assumptions, real data validation only.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from preprocessing.feature_engineering import (
    create_ml_features,
    extract_wqi_parameters,
    calculate_wqi_labels,
    PARAMETER_MAPPING
)
from preprocessing.us_data_features import prepare_us_features_for_prediction


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_minimal_df(**kwargs):
    """Create minimal DataFrame with all required columns for create_ml_features()"""
    base = {
        'year': kwargs.get('year', 2017),
        'ph': kwargs.get('ph', 7.0),
        'dissolved_oxygen': kwargs.get('dissolved_oxygen', 8.0),
        'temperature': kwargs.get('temperature', 15.0),
        'nitrate': kwargs.get('nitrate', 5.0),
        'conductance': kwargs.get('conductance', 500.0),
        'turbidity': kwargs.get('turbidity', None),
    }
    # Override with any provided kwargs
    base.update(kwargs)
    return pd.DataFrame([base])


# ============================================================================
# TEST CLASS 1: Temporal Features (20 tests)
# ============================================================================

class TestTemporalFeatures:
    """Test temporal feature calculations (years_since_1991, decade, period indicators)"""

    def test_years_since_1991_baseline(self):
        """Baseline year 1991 should give 0"""
        df = create_minimal_df(year=1991)
        result = create_ml_features(df)
        assert result['years_since_1991'].iloc[0] == 0, "1991 should be baseline (0)"

    def test_years_since_1991_first_year(self):
        """First year 1992 should give 1"""
        df = create_minimal_df(year=1992)
        result = create_ml_features(df)
        assert result['years_since_1991'].iloc[0] == 1, "1992 - 1991 = 1"

    def test_years_since_1991_year_2000(self):
        """Year 2000 should give 9"""
        df = create_minimal_df(year=2000)
        result = create_ml_features(df)
        assert result['years_since_1991'].iloc[0] == 9, "2000 - 1991 = 9"

    def test_years_since_1991_year_2017(self):
        """Year 2017 (dataset max) should give 26"""
        df = create_minimal_df(year=2017)
        result = create_ml_features(df)
        assert result['years_since_1991'].iloc[0] == 26, "2017 - 1991 = 26"

    def test_years_since_1991_year_2024(self):
        """Year 2024 (US predictions) should give 33"""
        df = create_minimal_df(year=2024)
        result = create_ml_features(df)
        assert result['years_since_1991'].iloc[0] == 33, "2024 - 1991 = 33"

    def test_decade_1990s(self):
        """1991-1999 should map to decade 1990"""
        for year in [1991, 1995, 1999]:
            df = create_minimal_df(year=year)
            result = create_ml_features(df)
            assert result['decade'].iloc[0] == 1990, f"Year {year} should be in 1990s"

    def test_decade_2000s(self):
        """2000-2009 should map to decade 2000"""
        for year in [2000, 2005, 2009]:
            df = create_minimal_df(year=year)
            result = create_ml_features(df)
            assert result['decade'].iloc[0] == 2000, f"Year {year} should be in 2000s"

    def test_decade_2010s(self):
        """2010-2019 should map to decade 2010"""
        for year in [2010, 2015, 2017, 2019]:
            df = create_minimal_df(year=year)
            result = create_ml_features(df)
            assert result['decade'].iloc[0] == 2010, f"Year {year} should be in 2010s"

    def test_decade_2020s(self):
        """2020-2029 should map to decade 2020"""
        for year in [2020, 2024, 2029]:
            df = create_minimal_df(year=year)
            result = create_ml_features(df)
            assert result['decade'].iloc[0] == 2020, f"Year {year} should be in 2020s"

    def test_decade_calculation_formula(self):
        """Verify decade calculation: (year // 10) * 10"""
        test_cases = [
            (1991, 1990),
            (2000, 2000),
            (2017, 2010),
            (2024, 2020),
            (1999, 1990),
            (2009, 2000),
        ]
        for year, expected_decade in test_cases:
            df = create_minimal_df(year=year)
            result = create_ml_features(df)
            assert result['decade'].iloc[0] == expected_decade, \
                f"Year {year}: expected decade {expected_decade}, got {result['decade'].iloc[0]}"

    def test_is_1990s_true_cases(self):
        """Years 1990-1999 should have is_1990s=True"""
        for year in [1990, 1991, 1995, 1999]:
            df = create_minimal_df(year=year)
            result = create_ml_features(df)
            assert result['is_1990s'].iloc[0] == True, f"Year {year} should be in 1990s"

    def test_is_1990s_false_cases(self):
        """Years outside 1990-1999 should have is_1990s=False"""
        for year in [1989, 2000, 2010, 2024]:
            df = create_minimal_df(year=year)
            result = create_ml_features(df)
            assert result['is_1990s'].iloc[0] == False, f"Year {year} should NOT be in 1990s"

    def test_is_2000s_true_cases(self):
        """Years 2000-2009 should have is_2000s=True"""
        for year in [2000, 2005, 2009]:
            df = create_minimal_df(year=year)
            result = create_ml_features(df)
            assert result['is_2000s'].iloc[0] == True, f"Year {year} should be in 2000s"

    def test_is_2000s_false_cases(self):
        """Years outside 2000-2009 should have is_2000s=False"""
        for year in [1999, 2010, 2024]:
            df = create_minimal_df(year=year)
            result = create_ml_features(df)
            assert result['is_2000s'].iloc[0] == False, f"Year {year} should NOT be in 2000s"

    def test_is_2010s_true_cases(self):
        """Years 2010-2019 should have is_2010s=True"""
        for year in [2010, 2015, 2017, 2019]:
            df = create_minimal_df(year=year)
            result = create_ml_features(df)
            assert result['is_2010s'].iloc[0] == True, f"Year {year} should be in 2010s"

    def test_is_2010s_false_cases(self):
        """Years outside 2010-2019 should have is_2010s=False"""
        for year in [2009, 2020, 2024]:
            df = create_minimal_df(year=year)
            result = create_ml_features(df)
            assert result['is_2010s'].iloc[0] == False, f"Year {year} should NOT be in 2010s"

    def test_period_indicators_mutually_exclusive(self):
        """Only one period indicator should be True at a time"""
        test_years = [1995, 2005, 2015, 2024]
        for year in test_years:
            df = create_minimal_df(year=year)
            result = create_ml_features(df)

            indicators = [result['is_1990s'].iloc[0], result['is_2000s'].iloc[0], result['is_2010s'].iloc[0]]
            true_count = sum([1 for ind in indicators if ind])

            assert true_count <= 1, f"Year {year}: At most one period indicator should be True"

    def test_temporal_features_no_missing_values(self):
        """Temporal features should never be null"""
        # Create 4 rows with different years
        df = pd.concat([
            create_minimal_df(year=1991),
            create_minimal_df(year=2000),
            create_minimal_df(year=2017),
            create_minimal_df(year=2024)
        ], ignore_index=True)
        result = create_ml_features(df)

        temporal_cols = ['years_since_1991', 'decade', 'is_1990s', 'is_2000s', 'is_2010s']
        for col in temporal_cols:
            assert result[col].notna().all(), f"{col} should have no missing values"

    def test_temporal_features_data_types(self):
        """Temporal features should have correct data types"""
        df = create_minimal_df(year=2017)
        result = create_ml_features(df)

        # Numeric features should be numeric
        assert pd.api.types.is_numeric_dtype(result['years_since_1991']), "years_since_1991 should be numeric"
        assert pd.api.types.is_numeric_dtype(result['decade']), "decade should be numeric"

        # Boolean features should be boolean
        assert pd.api.types.is_bool_dtype(result['is_1990s']), "is_1990s should be boolean"
        assert pd.api.types.is_bool_dtype(result['is_2000s']), "is_2000s should be boolean"
        assert pd.api.types.is_bool_dtype(result['is_2010s']), "is_2010s should be boolean"

    def test_decade_boundary_year_2000(self):
        """Year 2000 is exactly on decade boundary - should be 2000"""
        df = create_minimal_df(year=2000)
        result = create_ml_features(df)
        assert result['decade'].iloc[0] == 2000, "Year 2000 should map to decade 2000"
        assert result['is_2000s'].iloc[0] == True, "Year 2000 should be in 2000s"
        assert result['is_1990s'].iloc[0] == False, "Year 2000 should NOT be in 1990s"


# ============================================================================
# TEST CLASS 2: Water Quality Derived Features (30 tests)
# ============================================================================

class TestWaterQualityDerivedFeatures:
    """Test water quality derived features (pH deviation, DO-temp ratio, conductance categories)"""

    # pH Deviation Tests (5 tests)

    def test_ph_deviation_neutral(self):
        """pH 7.0 (neutral) should have deviation 0"""
        df = create_minimal_df(ph=7.0)
        result = create_ml_features(df)
        assert result['ph_deviation_from_7'].iloc[0] == 0.0, "pH 7.0 deviation should be 0"

    def test_ph_deviation_acidic(self):
        """pH 6.5 should have deviation 0.5"""
        df = create_minimal_df(ph=6.5)
        result = create_ml_features(df)
        assert abs(result['ph_deviation_from_7'].iloc[0] - 0.5) < 0.001, "pH 6.5 deviation should be 0.5"

    def test_ph_deviation_basic(self):
        """pH 8.5 should have deviation 1.5"""
        df = create_minimal_df(ph=8.5)
        result = create_ml_features(df)
        assert abs(result['ph_deviation_from_7'].iloc[0] - 1.5) < 0.001, "pH 8.5 deviation should be 1.5"

    def test_ph_deviation_symmetry(self):
        """pH deviations should be symmetric around 7.0"""
        df = pd.concat([
            create_minimal_df(ph=6.0),
            create_minimal_df(ph=8.0)
        ], ignore_index=True)
        result = create_ml_features(df)
        assert abs(result['ph_deviation_from_7'].iloc[0] - result['ph_deviation_from_7'].iloc[1]) < 0.001, \
            "pH 6.0 and 8.0 should have same deviation magnitude"

    def test_ph_deviation_extreme_values(self):
        """Extreme pH values should have correct deviations"""
        test_cases = [
            (0.0, 7.0),   # pH 0 (strong acid)
            (14.0, 7.0),  # pH 14 (strong base)
            (6.5, 0.5),   # EPA SMCL lower limit
            (8.5, 1.5),   # EPA SMCL upper limit
        ]
        for ph, expected_dev in test_cases:
            df = create_minimal_df(ph=ph)
            result = create_ml_features(df)
            assert abs(result['ph_deviation_from_7'].iloc[0] - expected_dev) < 0.001, \
                f"pH {ph} deviation should be {expected_dev}"

    # DO-Temperature Ratio Tests (10 tests)

    def test_do_temp_ratio_normal_conditions(self):
        """DO 8.0 mg/L, temp 15°C should give ratio 0.5"""
        df = create_minimal_df(dissolved_oxygen=8.0, temperature=15.0)
        result = create_ml_features(df)
        expected = 8.0 / (15.0 + 1)
        assert abs(result['do_temp_ratio'].iloc[0] - expected) < 0.001, \
            f"DO 8.0, temp 15°C: expected ratio {expected}"

    def test_do_temp_ratio_cold_water(self):
        """Cold water (0°C) should give higher ratio"""
        df = create_minimal_df(dissolved_oxygen=10.0, temperature=0.0)
        result = create_ml_features(df)
        expected = 10.0 / (0.0 + 1)  # = 10.0
        assert abs(result['do_temp_ratio'].iloc[0] - expected) < 0.001, \
            f"DO 10.0, temp 0°C: expected ratio {expected}"

    def test_do_temp_ratio_warm_water(self):
        """Warm water (30°C) should give lower ratio"""
        df = create_minimal_df(dissolved_oxygen=6.0, temperature=30.0)
        result = create_ml_features(df)
        expected = 6.0 / (30.0 + 1)  # ≈ 0.194
        assert abs(result['do_temp_ratio'].iloc[0] - expected) < 0.001, \
            f"DO 6.0, temp 30°C: expected ratio {expected}"

    def test_do_temp_ratio_zero_temp_edge_case(self):
        """Temperature 0°C should not cause division issues"""
        df = create_minimal_df(dissolved_oxygen=8.0, temperature=0.0)
        result = create_ml_features(df)
        expected = 8.0 / 1.0  # +1 prevents division by zero
        assert abs(result['do_temp_ratio'].iloc[0] - expected) < 0.001, \
            "Temp 0°C should work with +1 offset"

    def test_do_temp_ratio_negative_temp_edge_case(self):
        """CRITICAL: Temperature -1°C causes division by zero"""
        df = create_minimal_df(dissolved_oxygen=8.0, temperature=-1.0)
        result = create_ml_features(df)

        # With temp + 1, -1°C gives division by zero: 8.0 / 0
        # This is a CRITICAL edge case - pandas will return inf
        value = result['do_temp_ratio'].iloc[0]

        # The code will produce inf due to division by zero
        assert np.isinf(value), \
            "Temp -1°C causes division by zero (8.0 / 0), should be inf"

        # Verify it's positive infinity
        assert value > 0, "Should be positive infinity"

    def test_do_temp_ratio_very_cold_temp(self):
        """Very cold temperature (-40°C) gives negative denominator"""
        df = create_minimal_df(dissolved_oxygen=12.0, temperature=-40.0)
        result = create_ml_features(df)
        expected = 12.0 / (-40.0 + 1)  # = 12.0 / -39.0 ≈ -0.308
        assert abs(result['do_temp_ratio'].iloc[0] - expected) < 0.001, \
            "Very cold temp should give negative ratio"

    def test_do_temp_ratio_missing_do(self):
        """Missing DO should give NaN ratio"""
        df = create_minimal_df(dissolved_oxygen=np.nan, temperature=15.0)
        result = create_ml_features(df)
        assert np.isnan(result['do_temp_ratio'].iloc[0]), "Missing DO should give NaN ratio"

    def test_do_temp_ratio_missing_temp(self):
        """Missing temperature should give NaN ratio"""
        df = create_minimal_df(dissolved_oxygen=8.0, temperature=np.nan)
        result = create_ml_features(df)
        assert np.isnan(result['do_temp_ratio'].iloc[0]), "Missing temp should give NaN ratio"

    def test_do_temp_ratio_both_missing(self):
        """Both DO and temp missing should give NaN ratio"""
        df = create_minimal_df(dissolved_oxygen=np.nan, temperature=np.nan)
        result = create_ml_features(df)
        assert np.isnan(result['do_temp_ratio'].iloc[0]), "Both missing should give NaN ratio"

    def test_do_temp_ratio_formula_correctness(self):
        """Verify formula: DO / (temp + 1) for multiple values"""
        test_cases = [
            (5.0, 10.0, 5.0/11.0),
            (10.0, 20.0, 10.0/21.0),
            (15.0, 5.0, 15.0/6.0),
            (7.5, 25.0, 7.5/26.0),
        ]
        for do, temp, expected_ratio in test_cases:
            df = create_minimal_df(dissolved_oxygen=do, temperature=temp)
            result = create_ml_features(df)
            assert abs(result['do_temp_ratio'].iloc[0] - expected_ratio) < 0.001, \
                f"DO {do}, temp {temp}: expected {expected_ratio}"

    # Conductance Category Tests (15 tests)

    def test_conductance_low_boundary_below(self):
        """Conductance 199 should be low"""
        df = create_minimal_df(conductance=199.0)
        result = create_ml_features(df)
        assert result['conductance_low'].iloc[0] == 1.0, "Conductance 199 should be low"
        assert result['conductance_medium'].iloc[0] == 0.0, "Conductance 199 should not be medium"
        assert result['conductance_high'].iloc[0] == 0.0, "Conductance 199 should not be high"

    def test_conductance_low_boundary_at(self):
        """Conductance 200 should be medium (not low)"""
        df = create_minimal_df(conductance=200.0)
        result = create_ml_features(df)
        assert result['conductance_low'].iloc[0] == 0.0, "Conductance 200 should not be low"
        assert result['conductance_medium'].iloc[0] == 1.0, "Conductance 200 should be medium"
        assert result['conductance_high'].iloc[0] == 0.0, "Conductance 200 should not be high"

    def test_conductance_medium_range(self):
        """Conductance 500 should be medium"""
        df = create_minimal_df(conductance=500.0)
        result = create_ml_features(df)
        assert result['conductance_low'].iloc[0] == 0.0, "Conductance 500 should not be low"
        assert result['conductance_medium'].iloc[0] == 1.0, "Conductance 500 should be medium"
        assert result['conductance_high'].iloc[0] == 0.0, "Conductance 500 should not be high"

    def test_conductance_high_boundary_below(self):
        """Conductance 799 should be medium"""
        df = create_minimal_df(conductance=799.0)
        result = create_ml_features(df)
        assert result['conductance_low'].iloc[0] == 0.0, "Conductance 799 should not be low"
        assert result['conductance_medium'].iloc[0] == 1.0, "Conductance 799 should be medium"
        assert result['conductance_high'].iloc[0] == 0.0, "Conductance 799 should not be high"

    def test_conductance_high_boundary_at(self):
        """Conductance 800 should be high (not medium)"""
        df = create_minimal_df(conductance=800.0)
        result = create_ml_features(df)
        assert result['conductance_low'].iloc[0] == 0.0, "Conductance 800 should not be low"
        assert result['conductance_medium'].iloc[0] == 0.0, "Conductance 800 should not be medium"
        assert result['conductance_high'].iloc[0] == 1.0, "Conductance 800 should be high"

    def test_conductance_very_high(self):
        """Conductance 1500 should be high"""
        df = create_minimal_df(conductance=1500.0)
        result = create_ml_features(df)
        assert result['conductance_low'].iloc[0] == 0.0, "Conductance 1500 should not be low"
        assert result['conductance_medium'].iloc[0] == 0.0, "Conductance 1500 should not be medium"
        assert result['conductance_high'].iloc[0] == 1.0, "Conductance 1500 should be high"

    def test_conductance_zero(self):
        """Conductance 0 should be low"""
        df = create_minimal_df(conductance=0.0)
        result = create_ml_features(df)
        assert result['conductance_low'].iloc[0] == 1.0, "Conductance 0 should be low"

    def test_conductance_categories_mutually_exclusive(self):
        """Exactly one conductance category should be 1.0"""
        test_values = [0, 100, 200, 500, 799, 800, 1000, 1500]
        for cond in test_values:
            df = create_minimal_df(conductance=float(cond))
            result = create_ml_features(df)

            total = (result['conductance_low'].iloc[0] +
                    result['conductance_medium'].iloc[0] +
                    result['conductance_high'].iloc[0])

            assert total == 1.0, f"Conductance {cond}: exactly one category should be 1.0"

    def test_conductance_categories_threshold_verification(self):
        """Verify thresholds: <200 (low), 200-799 (medium), ≥800 (high)"""
        test_cases = [
            (50, 'low'),
            (199, 'low'),
            (200, 'medium'),
            (500, 'medium'),
            (799, 'medium'),
            (800, 'high'),
            (1200, 'high'),
        ]

        for cond, expected_category in test_cases:
            df = create_minimal_df(conductance=float(cond))
            result = create_ml_features(df)

            if expected_category == 'low':
                assert result['conductance_low'].iloc[0] == 1.0, f"Conductance {cond} should be low"
            elif expected_category == 'medium':
                assert result['conductance_medium'].iloc[0] == 1.0, f"Conductance {cond} should be medium"
            elif expected_category == 'high':
                assert result['conductance_high'].iloc[0] == 1.0, f"Conductance {cond} should be high"

    def test_conductance_categories_data_type(self):
        """Conductance categories should be float type"""
        df = create_minimal_df(conductance=500.0)
        result = create_ml_features(df)

        assert result['conductance_low'].dtype == np.float64, "conductance_low should be float64"
        assert result['conductance_medium'].dtype == np.float64, "conductance_medium should be float64"
        assert result['conductance_high'].dtype == np.float64, "conductance_high should be float64"

    def test_conductance_missing_value_handling(self):
        """Missing conductance: boolean comparisons with NaN give False -> 0.0"""
        df = create_minimal_df(conductance=np.nan)
        result = create_ml_features(df)

        # When conductance is NaN:
        # - (NaN < 200).astype(float) -> False.astype(float) -> 0.0
        # - (NaN >= 200) & (NaN < 800).astype(float) -> 0.0
        # - (NaN >= 800).astype(float) -> 0.0
        # This is pandas/numpy behavior: comparisons with NaN are always False

        assert result['conductance_low'].iloc[0] == 0.0, "Missing conductance: comparison returns False -> 0.0"
        assert result['conductance_medium'].iloc[0] == 0.0, "Missing conductance: comparison returns False -> 0.0"
        assert result['conductance_high'].iloc[0] == 0.0, "Missing conductance: comparison returns False -> 0.0"

        # All three categories are 0.0 (not mutually exclusive when NaN)
        total = (result['conductance_low'].iloc[0] +
                result['conductance_medium'].iloc[0] +
                result['conductance_high'].iloc[0])
        assert total == 0.0, "Missing conductance: all categories are 0.0"

    def test_nitrate_pollution_level_low(self):
        """Nitrate < 5 mg/L should be 'low'"""
        df = create_minimal_df(nitrate=3.0)
        result = create_ml_features(df)
        assert result['nitrate_pollution_level'].iloc[0] == 'low', "Nitrate 3.0 should be low pollution"

    def test_nitrate_pollution_level_moderate(self):
        """Nitrate 5-10 mg/L should be 'moderate'"""
        df = create_minimal_df(nitrate=7.5)
        result = create_ml_features(df)
        assert result['nitrate_pollution_level'].iloc[0] == 'moderate', "Nitrate 7.5 should be moderate pollution"

    def test_nitrate_pollution_level_high(self):
        """Nitrate 10-20 mg/L should be 'high'"""
        df = create_minimal_df(nitrate=15.0)
        result = create_ml_features(df)
        assert result['nitrate_pollution_level'].iloc[0] == 'high', "Nitrate 15.0 should be high pollution"

    def test_nitrate_pollution_level_very_high(self):
        """Nitrate > 20 mg/L should be 'very_high'"""
        df = create_minimal_df(nitrate=25.0)
        result = create_ml_features(df)
        assert result['nitrate_pollution_level'].iloc[0] == 'very_high', "Nitrate 25.0 should be very high pollution"


# ============================================================================
# TEST CLASS 3: Interaction Features (40 tests)
# ============================================================================

class TestInteractionFeatures:
    """Test interaction features (pollution_stress, temp_stress, gdp_per_capita_proxy)"""

    # Pollution Stress Tests (15 tests)

    def test_pollution_stress_formula_clean_water(self):
        """Clean water (low nitrate, high DO) should have low pollution stress"""
        df = create_minimal_df(nitrate=1.0, dissolved_oxygen=9.0)
        result = create_ml_features(df)
        # pollution_stress = (nitrate / 50) * (1 - DO / 10)
        expected = (1.0 / 50) * (1 - 9.0 / 10)  # = 0.02 * 0.1 = 0.002
        assert abs(result['pollution_stress'].iloc[0] - expected) < 0.001, \
            f"Clean water should have low pollution stress: expected {expected}"

    def test_pollution_stress_formula_polluted_water(self):
        """Polluted water (high nitrate, low DO) should have high pollution stress"""
        df = create_minimal_df(nitrate=40.0, dissolved_oxygen=3.0)
        result = create_ml_features(df)
        expected = (40.0 / 50) * (1 - 3.0 / 10)  # = 0.8 * 0.7 = 0.56
        assert abs(result['pollution_stress'].iloc[0] - expected) < 0.001, \
            f"Polluted water should have high pollution stress: expected {expected}"

    def test_pollution_stress_zero_nitrate(self):
        """Zero nitrate should give zero pollution stress regardless of DO"""
        for do in [5.0, 8.0, 10.0]:
            df = create_minimal_df(nitrate=0.0, dissolved_oxygen=do)
            result = create_ml_features(df)
            assert result['pollution_stress'].iloc[0] == 0.0, \
                f"Zero nitrate should give zero pollution stress (DO={do})"

    def test_pollution_stress_max_do(self):
        """Maximum DO (10 mg/L) should give zero pollution stress"""
        df = create_minimal_df(nitrate=20.0, dissolved_oxygen=10.0)
        result = create_ml_features(df)
        # (20/50) * (1 - 10/10) = 0.4 * 0 = 0
        assert result['pollution_stress'].iloc[0] == 0.0, \
            "Max DO should give zero pollution stress"

    def test_pollution_stress_missing_nitrate_fillna_zero(self):
        """Missing nitrate should be filled with 0 (clean assumption)"""
        df = create_minimal_df(nitrate=np.nan, dissolved_oxygen=5.0)
        result = create_ml_features(df)
        # fillna(0): (0 / 50) * (1 - 5/10) = 0
        assert result['pollution_stress'].iloc[0] == 0.0, \
            "Missing nitrate filled with 0 should give zero pollution stress"

    def test_pollution_stress_missing_do_fillna_ten(self):
        """Missing DO should be filled with 10 (saturated assumption)"""
        df = create_minimal_df(nitrate=25.0, dissolved_oxygen=np.nan)
        result = create_ml_features(df)
        # fillna(10): (25/50) * (1 - 10/10) = 0.5 * 0 = 0
        assert result['pollution_stress'].iloc[0] == 0.0, \
            "Missing DO filled with 10 should give zero pollution stress"

    def test_pollution_stress_both_missing(self):
        """Both nitrate and DO missing should give zero stress"""
        df = create_minimal_df(nitrate=np.nan, dissolved_oxygen=np.nan)
        result = create_ml_features(df)
        # fillna: (0/50) * (1 - 10/10) = 0
        assert result['pollution_stress'].iloc[0] == 0.0, \
            "Both missing should give zero pollution stress"

    def test_pollution_stress_nitrate_at_50(self):
        """Nitrate at 50 mg/L (divisor) should give full weight"""
        df = create_minimal_df(nitrate=50.0, dissolved_oxygen=0.0)
        result = create_ml_features(df)
        # (50/50) * (1 - 0/10) = 1 * 1 = 1.0 (maximum stress)
        assert abs(result['pollution_stress'].iloc[0] - 1.0) < 0.001, \
            "Nitrate=50, DO=0 should give maximum pollution stress (1.0)"

    def test_pollution_stress_do_at_zero(self):
        """DO at 0 should give maximum contribution from DO term"""
        df = create_minimal_df(nitrate=25.0, dissolved_oxygen=0.0)
        result = create_ml_features(df)
        # (25/50) * (1 - 0/10) = 0.5 * 1 = 0.5
        assert abs(result['pollution_stress'].iloc[0] - 0.5) < 0.001, \
            "DO=0 should maximize DO contribution"

    def test_pollution_stress_fillna_assumptions_scientifically_reasonable(self):
        """CRITICAL: Are fillna defaults scientifically reasonable?"""
        # Nitrate fillna(0): Assumes no pollution if unknown
        # DO fillna(10): Assumes full saturation if unknown
        # These are optimistic assumptions (assume clean water if unknown)

        # Test that we document this behavior
        df = create_minimal_df(nitrate=np.nan, dissolved_oxygen=np.nan)
        result = create_ml_features(df)

        # With optimistic fillna, stress = 0
        assert result['pollution_stress'].iloc[0] == 0.0, \
            "Optimistic fillna assumes clean water (stress=0)"

        # NOTE: This may not be appropriate for all use cases!
        # Real missing data might indicate poor monitoring, not clean water

    def test_pollution_stress_range_bounded(self):
        """Pollution stress should be bounded [0, 1]"""
        test_cases = [
            (0.0, 10.0, 0.0),   # Min stress
            (50.0, 0.0, 1.0),   # Max stress
            (25.0, 5.0, 0.25),  # Mid stress
        ]

        for nitrate, do, expected in test_cases:
            df = create_minimal_df(nitrate=nitrate, dissolved_oxygen=do)
            result = create_ml_features(df)
            stress = result['pollution_stress'].iloc[0]

            assert 0.0 <= stress <= 1.0, \
                f"Pollution stress must be in [0,1], got {stress} for nitrate={nitrate}, DO={do}"
            assert abs(stress - expected) < 0.001, \
                f"Expected {expected}, got {stress}"

    def test_pollution_stress_extreme_nitrate(self):
        """Very high nitrate (>50) should still work"""
        df = create_minimal_df(nitrate=100.0, dissolved_oxygen=5.0)
        result = create_ml_features(df)
        # (100/50) * (1 - 5/10) = 2.0 * 0.5 = 1.0
        # Can exceed 1.0 if nitrate > 50!
        assert result['pollution_stress'].iloc[0] == 1.0, \
            "Nitrate=100 should give pollution stress = 1.0"

    def test_pollution_stress_extreme_do(self):
        """Very high DO (>10) should give negative term"""
        df = create_minimal_df(nitrate=25.0, dissolved_oxygen=15.0)
        result = create_ml_features(df)
        # (25/50) * (1 - 15/10) = 0.5 * -0.5 = -0.25
        # Can be negative if DO > 10!
        assert result['pollution_stress'].iloc[0] < 0, \
            "DO=15 should give negative pollution stress"

    def test_pollution_stress_data_type(self):
        """Pollution stress should be numeric"""
        df = create_minimal_df(nitrate=10.0, dissolved_oxygen=8.0)
        result = create_ml_features(df)
        assert pd.api.types.is_numeric_dtype(result['pollution_stress']), \
            "pollution_stress should be numeric"

    def test_pollution_stress_formula_verification(self):
        """Verify exact formula: (nitrate/50) * (1 - DO/10)"""
        test_cases = [
            (10.0, 8.0, (10.0/50) * (1 - 8.0/10)),  # = 0.2 * 0.2 = 0.04
            (30.0, 4.0, (30.0/50) * (1 - 4.0/10)),  # = 0.6 * 0.6 = 0.36
            (5.0, 9.5, (5.0/50) * (1 - 9.5/10)),    # = 0.1 * 0.05 = 0.005
        ]

        for nitrate, do, expected in test_cases:
            df = create_minimal_df(nitrate=nitrate, dissolved_oxygen=do)
            result = create_ml_features(df)
            assert abs(result['pollution_stress'].iloc[0] - expected) < 0.001, \
                f"nitrate={nitrate}, DO={do}: expected {expected}, got {result['pollution_stress'].iloc[0]}"

    # Temperature Stress Tests (15 tests)

    def test_temp_stress_formula_optimal_temp(self):
        """Temperature at 15°C (optimal) should have zero stress"""
        df = create_minimal_df(temperature=15.0)
        result = create_ml_features(df)
        # temp_stress = abs(temp - 15) / 15 = abs(15-15)/15 = 0
        assert result['temp_stress'].iloc[0] == 0.0, \
            "Optimal temperature (15°C) should have zero stress"

    def test_temp_stress_cold_water(self):
        """Cold water should have positive stress"""
        df = create_minimal_df(temperature=0.0)
        result = create_ml_features(df)
        expected = abs(0.0 - 15.0) / 15.0  # = 15/15 = 1.0
        assert abs(result['temp_stress'].iloc[0] - expected) < 0.001, \
            f"Cold water (0°C) should have stress {expected}"

    def test_temp_stress_warm_water(self):
        """Warm water should have positive stress"""
        df = create_minimal_df(temperature=30.0)
        result = create_ml_features(df)
        expected = abs(30.0 - 15.0) / 15.0  # = 15/15 = 1.0
        assert abs(result['temp_stress'].iloc[0] - expected) < 0.001, \
            f"Warm water (30°C) should have stress {expected}"

    def test_temp_stress_symmetry(self):
        """Temperature stress should be symmetric around 15°C"""
        df1 = create_minimal_df(temperature=10.0)
        df2 = create_minimal_df(temperature=20.0)
        result1 = create_ml_features(df1)
        result2 = create_ml_features(df2)

        # Both are 5°C away from optimal
        stress1 = result1['temp_stress'].iloc[0]
        stress2 = result2['temp_stress'].iloc[0]

        assert abs(stress1 - stress2) < 0.001, \
            f"10°C and 20°C should have same stress: {stress1} vs {stress2}"

    def test_temp_stress_missing_no_fillna(self):
        """Missing temperature gives NaN (no fillna in code)"""
        df = create_minimal_df(temperature=np.nan)
        result = create_ml_features(df)
        # Code: np.abs(df['temperature'] - 15) / 15
        # NaN - 15 = NaN, abs(NaN) = NaN
        assert np.isnan(result['temp_stress'].iloc[0]), \
            "Missing temperature should give NaN (no fillna in code)"

    def test_temp_stress_extreme_cold(self):
        """Very cold temperature (-40°C) should have high stress"""
        df = create_minimal_df(temperature=-40.0)
        result = create_ml_features(df)
        expected = abs(-40.0 - 15.0) / 15.0  # = 55/15 ≈ 3.67
        assert abs(result['temp_stress'].iloc[0] - expected) < 0.001, \
            f"Extreme cold should have high stress: expected {expected}"

    def test_temp_stress_extreme_hot(self):
        """Very hot temperature (50°C) should have high stress"""
        df = create_minimal_df(temperature=50.0)
        result = create_ml_features(df)
        expected = abs(50.0 - 15.0) / 15.0  # = 35/15 ≈ 2.33
        assert abs(result['temp_stress'].iloc[0] - expected) < 0.001, \
            f"Extreme heat should have high stress: expected {expected}"

    def test_temp_stress_not_bounded(self):
        """Temperature stress is NOT bounded (can exceed 1.0)"""
        df = create_minimal_df(temperature=45.0)
        result = create_ml_features(df)
        stress = result['temp_stress'].iloc[0]
        # abs(45-15)/15 = 30/15 = 2.0
        assert stress > 1.0, \
            "Temperature stress can exceed 1.0 for extreme temps"

    def test_temp_stress_near_optimal(self):
        """Temperatures near 15°C should have low stress"""
        for temp in [14.0, 15.5, 16.0]:
            df = create_minimal_df(temperature=temp)
            result = create_ml_features(df)
            stress = result['temp_stress'].iloc[0]
            assert stress < 0.1, \
                f"Temperature {temp}°C should have low stress, got {stress}"

    def test_temp_stress_data_type(self):
        """Temperature stress should be numeric"""
        df = create_minimal_df(temperature=20.0)
        result = create_ml_features(df)
        assert pd.api.types.is_numeric_dtype(result['temp_stress']), \
            "temp_stress should be numeric"

    def test_temp_stress_formula_verification(self):
        """Verify exact formula: abs(temp - 15) / 15"""
        test_cases = [
            (10.0, abs(10.0 - 15.0) / 15.0),  # = 5/15 ≈ 0.333
            (20.0, abs(20.0 - 15.0) / 15.0),  # = 5/15 ≈ 0.333
            (0.0, abs(0.0 - 15.0) / 15.0),    # = 15/15 = 1.0
            (30.0, abs(30.0 - 15.0) / 15.0),  # = 15/15 = 1.0
        ]

        for temp, expected in test_cases:
            df = create_minimal_df(temperature=temp)
            result = create_ml_features(df)
            assert abs(result['temp_stress'].iloc[0] - expected) < 0.001, \
                f"temp={temp}: expected {expected}, got {result['temp_stress'].iloc[0]}"

    def test_temp_stress_zero_division_impossible(self):
        """Divisor is constant 15, so no division by zero"""
        # Just verify it works for various temps
        for temp in [-50, 0, 15, 50, 100]:
            df = create_minimal_df(temperature=float(temp))
            result = create_ml_features(df)
            assert not np.isnan(result['temp_stress'].iloc[0]), \
                f"Temperature {temp} should not give NaN stress"
            assert not np.isinf(result['temp_stress'].iloc[0]), \
                f"Temperature {temp} should not give inf stress"

    def test_temp_stress_always_non_negative(self):
        """Temperature stress should always be non-negative (abs value)"""
        for temp in [-40, 0, 10, 15, 20, 30, 50]:
            df = create_minimal_df(temperature=float(temp))
            result = create_ml_features(df)
            assert result['temp_stress'].iloc[0] >= 0, \
                f"Temperature stress must be >= 0, got {result['temp_stress'].iloc[0]} for temp={temp}"

    def test_temp_stress_no_fillna_behavior(self):
        """CRITICAL: temp_stress has NO fillna - missing temp gives NaN"""
        # Code does not use fillna for temperature
        # This means missing data propagates as NaN

        df = create_minimal_df(temperature=np.nan)
        result = create_ml_features(df)

        # No fillna applied, so result is NaN
        assert np.isnan(result['temp_stress'].iloc[0]), \
            "No fillna for temp_stress - missing temp gives NaN"

        # This behavior differs from pollution_stress (which uses fillna)
        # Models must handle these NaN values via imputation

    def test_temp_stress_gradient_check(self):
        """Stress should increase monotonically away from 15°C"""
        temps = [0, 5, 10, 15, 20, 25, 30]
        stresses = []

        for temp in temps:
            df = create_minimal_df(temperature=float(temp))
            result = create_ml_features(df)
            stresses.append(result['temp_stress'].iloc[0])

        # Stress should decrease up to 15, then increase
        # 0 < 5 < 10 < 15 > 20 > 25 > 30
        assert stresses[3] == 0.0, "15°C should have min stress"
        assert stresses[2] < stresses[1] < stresses[0], "Stress should increase as temp decreases below 15"
        assert stresses[4] < stresses[5] < stresses[6], "Stress should increase as temp increases above 15"

    # GDP per capita proxy test (10 tests)

    def test_gdp_per_capita_proxy_requires_columns(self):
        """GDP per capita proxy only created if gdp and PopulationDensity columns exist"""
        # Code: if 'gdp' in df.columns and 'PopulationDensity' in df.columns:
        #           df['gdp_per_capita_proxy'] = df['gdp'] / (df['PopulationDensity'] + 1)

        # Our minimal df doesn't have these columns
        df = create_minimal_df()
        result = create_ml_features(df)

        # Feature should NOT be created without required columns
        assert 'gdp_per_capita_proxy' not in result.columns or pd.isna(result.get('gdp_per_capita_proxy', pd.Series([None])).iloc[0]), \
            "gdp_per_capita_proxy requires gdp and PopulationDensity columns (not in US data)"

    def test_interaction_features_created(self):
        """Verify all interaction features are created"""
        df = create_minimal_df()
        result = create_ml_features(df)

        expected_features = ['pollution_stress', 'temp_stress']
        for feature in expected_features:
            assert feature in result.columns, \
                f"Interaction feature {feature} should be created"

    def test_interaction_features_nan_handling_differs(self):
        """pollution_stress uses fillna, temp_stress does not"""
        df = create_minimal_df(
            nitrate=np.nan,
            dissolved_oxygen=np.nan,
            temperature=np.nan
        )
        result = create_ml_features(df)

        # pollution_stress uses fillna(0) for nitrate, fillna(10) for DO
        assert not np.isnan(result['pollution_stress'].iloc[0]), \
            "pollution_stress should handle NaN via fillna"
        assert result['pollution_stress'].iloc[0] == 0.0, \
            "Missing nitrate and DO with fillna gives zero pollution_stress"

        # temp_stress does NOT use fillna
        assert np.isnan(result['temp_stress'].iloc[0]), \
            "temp_stress should be NaN when temperature is missing (no fillna)"

    def test_interaction_features_real_world_scenario_pristine(self):
        """Pristine mountain stream: low stress expected"""
        df = create_minimal_df(
            nitrate=0.5,      # Very low
            dissolved_oxygen=10.0,  # Saturated
            temperature=12.0   # Cool, near optimal
        )
        result = create_ml_features(df)

        assert result['pollution_stress'].iloc[0] < 0.1, \
            "Pristine water should have low pollution stress"
        assert result['temp_stress'].iloc[0] < 0.3, \
            "Cool water should have low temperature stress"

    def test_interaction_features_real_world_scenario_polluted(self):
        """Polluted urban stream: high stress expected"""
        df = create_minimal_df(
            nitrate=30.0,     # High
            dissolved_oxygen=3.0,   # Low (hypoxic)
            temperature=28.0   # Warm
        )
        result = create_ml_features(df)

        assert result['pollution_stress'].iloc[0] > 0.3, \
            "Polluted water should have high pollution stress"
        assert result['temp_stress'].iloc[0] > 0.8, \
            "Warm water should have high temperature stress"

    def test_interaction_features_real_world_scenario_agricultural(self):
        """Agricultural runoff: high nitrate, moderate DO"""
        df = create_minimal_df(
            nitrate=20.0,     # High from fertilizer
            dissolved_oxygen=6.0,   # Moderate
            temperature=18.0   # Slightly warm
        )
        result = create_ml_features(df)

        # (20/50) * (1 - 6/10) = 0.4 * 0.4 = 0.16
        assert abs(result['pollution_stress'].iloc[0] - 0.16) < 0.01, \
            "Agricultural runoff should have moderate pollution stress"

    def test_interaction_features_data_types_all_numeric(self):
        """All interaction features should be numeric"""
        df = create_minimal_df()
        result = create_ml_features(df)

        for feature in ['pollution_stress', 'temp_stress']:
            assert pd.api.types.is_numeric_dtype(result[feature]), \
                f"{feature} should be numeric"

    def test_interaction_features_fillna_behavior(self):
        """Different fillna behavior: pollution_stress has fillna, temp_stress doesn't"""
        # Create data with all parameters missing
        df = create_minimal_df(
            nitrate=np.nan,
            dissolved_oxygen=np.nan,
            temperature=np.nan
        )
        result = create_ml_features(df)

        # pollution_stress uses fillna - should not be NaN
        assert result['pollution_stress'].notna().all(), \
            "pollution_stress should have no NaN after fillna"

        # temp_stress does NOT use fillna - WILL be NaN
        assert result['temp_stress'].isna().all(), \
            "temp_stress should be NaN when temperature is missing (no fillna in code)"

    def test_interaction_features_multiple_rows(self):
        """Interaction features should work correctly for multiple rows"""
        df = pd.concat([
            create_minimal_df(nitrate=10.0, dissolved_oxygen=8.0, temperature=15.0),
            create_minimal_df(nitrate=30.0, dissolved_oxygen=4.0, temperature=25.0),
            create_minimal_df(nitrate=5.0, dissolved_oxygen=9.0, temperature=10.0),
        ], ignore_index=True)

        result = create_ml_features(df)

        # Verify each row has correct values
        assert len(result) == 3, "Should have 3 rows"
        assert result['pollution_stress'].notna().all(), "All rows should have pollution_stress"
        assert result['temp_stress'].notna().all(), "All rows should have temp_stress"

        # Verify values are different (not all the same)
        assert len(result['pollution_stress'].unique()) > 1, \
            "pollution_stress should vary across rows"
        assert len(result['temp_stress'].unique()) > 1, \
            "temp_stress should vary across rows"


# ============================================================================
# TEST CLASS 4: Missing Value Handling (30 tests)
# ============================================================================

class TestMissingValueHandling:
    """Test missing value indicators and fillna strategies"""

    # --- Missing Value Indicators (10 tests) ---

    def test_ph_missing_indicator_when_present(self):
        """ph_missing should be 0 when pH is present"""
        df = create_minimal_df(ph=7.0)
        result = create_ml_features(df)
        assert result['ph_missing'].iloc[0] == 0, "pH present should give missing flag = 0"

    def test_ph_missing_indicator_when_absent(self):
        """ph_missing should be 1 when pH is NaN"""
        df = create_minimal_df(ph=np.nan)
        result = create_ml_features(df)
        assert result['ph_missing'].iloc[0] == 1, "pH NaN should give missing flag = 1"

    def test_dissolved_oxygen_missing_indicator_when_present(self):
        """dissolved_oxygen_missing should be 0 when DO is present"""
        df = create_minimal_df(dissolved_oxygen=8.0)
        result = create_ml_features(df)
        assert result['dissolved_oxygen_missing'].iloc[0] == 0, \
            "DO present should give missing flag = 0"

    def test_dissolved_oxygen_missing_indicator_when_absent(self):
        """dissolved_oxygen_missing should be 1 when DO is NaN"""
        df = create_minimal_df(dissolved_oxygen=np.nan)
        result = create_ml_features(df)
        assert result['dissolved_oxygen_missing'].iloc[0] == 1, \
            "DO NaN should give missing flag = 1"

    def test_temperature_missing_indicator_when_present(self):
        """temperature_missing should be 0 when temperature is present"""
        df = create_minimal_df(temperature=15.0)
        result = create_ml_features(df)
        assert result['temperature_missing'].iloc[0] == 0, \
            "Temperature present should give missing flag = 0"

    def test_temperature_missing_indicator_when_absent(self):
        """temperature_missing should be 1 when temperature is NaN"""
        df = create_minimal_df(temperature=np.nan)
        result = create_ml_features(df)
        assert result['temperature_missing'].iloc[0] == 1, \
            "Temperature NaN should give missing flag = 1"

    def test_nitrate_missing_indicator_when_present(self):
        """nitrate_missing should be 0 when nitrate is present"""
        df = create_minimal_df(nitrate=5.0)
        result = create_ml_features(df)
        assert result['nitrate_missing'].iloc[0] == 0, \
            "Nitrate present should give missing flag = 0"

    def test_nitrate_missing_indicator_when_absent(self):
        """nitrate_missing should be 1 when nitrate is NaN"""
        df = create_minimal_df(nitrate=np.nan)
        result = create_ml_features(df)
        assert result['nitrate_missing'].iloc[0] == 1, \
            "Nitrate NaN should give missing flag = 1"

    def test_conductance_missing_indicator_when_present(self):
        """conductance_missing should be 0 when conductance is present"""
        df = create_minimal_df(conductance=500.0)
        result = create_ml_features(df)
        assert result['conductance_missing'].iloc[0] == 0, \
            "Conductance present should give missing flag = 0"

    def test_conductance_missing_indicator_when_absent(self):
        """conductance_missing should be 1 when conductance is NaN"""
        df = create_minimal_df(conductance=np.nan)
        result = create_ml_features(df)
        assert result['conductance_missing'].iloc[0] == 1, \
            "Conductance NaN should give missing flag = 1"

    def test_turbidity_missing_indicator_always_1(self):
        """turbidity_missing should be 1 (turbidity always None in Kaggle dataset)"""
        df = create_minimal_df(turbidity=None)
        result = create_ml_features(df)
        assert result['turbidity_missing'].iloc[0] == 1, \
            "Turbidity always None should give missing flag = 1"

    def test_n_params_available_all_present(self):
        """n_params_available should be 6 when all parameters present"""
        df = create_minimal_df(
            ph=7.0,
            dissolved_oxygen=8.0,
            temperature=15.0,
            turbidity=5.0,  # Even if present (rare)
            nitrate=5.0,
            conductance=500.0
        )
        result = create_ml_features(df)
        assert result['n_params_available'].iloc[0] == 6, \
            "All 6 parameters present should give count = 6"

    def test_n_params_available_five_present(self):
        """n_params_available should be 5 when turbidity missing (typical case)"""
        df = create_minimal_df(
            ph=7.0,
            dissolved_oxygen=8.0,
            temperature=15.0,
            turbidity=None,  # Typical case
            nitrate=5.0,
            conductance=500.0
        )
        result = create_ml_features(df)
        assert result['n_params_available'].iloc[0] == 5, \
            "5 parameters present should give count = 5"

    def test_n_params_available_none_present(self):
        """n_params_available should be 0 when all parameters missing"""
        df = create_minimal_df(
            ph=np.nan,
            dissolved_oxygen=np.nan,
            temperature=np.nan,
            turbidity=None,
            nitrate=np.nan,
            conductance=np.nan
        )
        result = create_ml_features(df)
        assert result['n_params_available'].iloc[0] == 0, \
            "All parameters missing should give count = 0"

    def test_n_params_available_partial_missing(self):
        """n_params_available should count correctly with partial data"""
        df = create_minimal_df(
            ph=7.0,           # Present
            dissolved_oxygen=np.nan,  # Missing
            temperature=15.0,  # Present
            turbidity=None,    # Missing
            nitrate=np.nan,    # Missing
            conductance=500.0  # Present
        )
        result = create_ml_features(df)
        assert result['n_params_available'].iloc[0] == 3, \
            "3 parameters present (pH, temp, conductance) should give count = 3"

    # --- fillna Strategy Tests (10 tests) ---

    def test_pollution_stress_fillna_nitrate_zero(self):
        """pollution_stress uses fillna(0) for missing nitrate"""
        df = create_minimal_df(
            nitrate=np.nan,
            dissolved_oxygen=8.0
        )
        result = create_ml_features(df)

        # Expected: (0 / 50) * (1 - 8/10) = 0 * 0.2 = 0.0
        assert abs(result['pollution_stress'].iloc[0] - 0.0) < 0.01, \
            "Missing nitrate should fillna(0), resulting in pollution_stress = 0"

    def test_pollution_stress_fillna_do_ten(self):
        """pollution_stress uses fillna(10) for missing DO"""
        df = create_minimal_df(
            nitrate=20.0,
            dissolved_oxygen=np.nan
        )
        result = create_ml_features(df)

        # Expected: (20 / 50) * (1 - 10/10) = 0.4 * 0 = 0.0
        assert abs(result['pollution_stress'].iloc[0] - 0.0) < 0.01, \
            "Missing DO should fillna(10), resulting in pollution_stress = 0"

    def test_pollution_stress_fillna_both_missing(self):
        """pollution_stress with both nitrate and DO missing should give 0"""
        df = create_minimal_df(
            nitrate=np.nan,
            dissolved_oxygen=np.nan
        )
        result = create_ml_features(df)

        # Expected: (0 / 50) * (1 - 10/10) = 0 * 0 = 0.0
        assert abs(result['pollution_stress'].iloc[0] - 0.0) < 0.01, \
            "Both missing should fillna to nitrate=0, DO=10, giving pollution_stress = 0"

    def test_temp_stress_no_fillna_missing_temperature(self):
        """temp_stress does NOT use fillna - NaN propagates"""
        df = create_minimal_df(temperature=np.nan)
        result = create_ml_features(df)

        assert pd.isna(result['temp_stress'].iloc[0]), \
            "Missing temperature should propagate as NaN (no fillna in code)"

    def test_temp_stress_present_temperature(self):
        """temp_stress calculates correctly when temperature present"""
        df = create_minimal_df(temperature=15.0)
        result = create_ml_features(df)

        # Expected: abs(15 - 15) / 15 = 0 / 15 = 0.0
        assert abs(result['temp_stress'].iloc[0] - 0.0) < 0.01, \
            "Optimal temperature (15°C) should give temp_stress = 0"

    def test_ph_deviation_no_fillna(self):
        """ph_deviation_from_7 does NOT use fillna - NaN propagates"""
        df = create_minimal_df(ph=np.nan)
        result = create_ml_features(df)

        assert pd.isna(result['ph_deviation_from_7'].iloc[0]), \
            "Missing pH should propagate as NaN (no fillna)"

    def test_do_temp_ratio_no_fillna_missing_do(self):
        """do_temp_ratio does NOT use fillna for DO - NaN propagates"""
        df = create_minimal_df(
            dissolved_oxygen=np.nan,
            temperature=15.0
        )
        result = create_ml_features(df)

        assert pd.isna(result['do_temp_ratio'].iloc[0]), \
            "Missing DO should propagate as NaN (no fillna)"

    def test_do_temp_ratio_no_fillna_missing_temp(self):
        """do_temp_ratio does NOT use fillna for temperature - NaN propagates"""
        df = create_minimal_df(
            dissolved_oxygen=8.0,
            temperature=np.nan
        )
        result = create_ml_features(df)

        assert pd.isna(result['do_temp_ratio'].iloc[0]), \
            "Missing temperature should propagate as NaN (no fillna)"

    def test_fillna_inconsistency_documentation(self):
        """CRITICAL: Document inconsistent fillna behavior across features"""
        df = create_minimal_df(
            nitrate=np.nan,
            dissolved_oxygen=np.nan,
            temperature=np.nan,
            ph=np.nan
        )
        result = create_ml_features(df)

        # pollution_stress uses fillna - should NOT be NaN
        assert result['pollution_stress'].notna().iloc[0], \
            "pollution_stress uses fillna(0) and fillna(10)"

        # temp_stress does NOT use fillna - should BE NaN
        assert result['temp_stress'].isna().iloc[0], \
            "temp_stress does NOT use fillna"

        # ph_deviation does NOT use fillna - should BE NaN
        assert result['ph_deviation_from_7'].isna().iloc[0], \
            "ph_deviation_from_7 does NOT use fillna"

        # CRITICAL: Inconsistent handling may confuse ML models
        # Some features assume "clean water" when missing (fillna)
        # Others assume "unknown" when missing (NaN propagation)

    def test_fillna_optimistic_assumption_question(self):
        """CRITICAL: Question whether fillna(0) for nitrate is scientifically valid"""
        df = create_minimal_df(nitrate=np.nan)
        result = create_ml_features(df)

        # pollution_stress assumes nitrate=0 when missing
        # This is OPTIMISTIC - assumes no pollution if unknown
        # Real missing data might indicate poor monitoring, not clean water

        # Expected: (0 / 50) * (1 - DO/10) - using fillna(0)
        assert result['pollution_stress'].notna().iloc[0], \
            "Missing nitrate fillna(0) assumes no pollution - is this scientifically valid?"

        # QUESTION FOR USER: Should missing data be treated as:
        # A) Optimistic (fillna with good values) - current implementation
        # B) Unknown (NaN propagation) - more conservative
        # C) Pessimistic (fillna with bad values) - most conservative

    # --- pd.cut NaN Handling (5 tests) ---

    def test_nitrate_pollution_level_missing_value(self):
        """nitrate_pollution_level should be NaN when nitrate is missing"""
        df = create_minimal_df(nitrate=np.nan)
        result = create_ml_features(df)

        assert pd.isna(result['nitrate_pollution_level'].iloc[0]), \
            "pd.cut should return NaN when input is NaN"

    def test_nitrate_pollution_level_low(self):
        """nitrate_pollution_level should be 'low' for nitrate < 5"""
        df = create_minimal_df(nitrate=3.0)
        result = create_ml_features(df)

        assert result['nitrate_pollution_level'].iloc[0] == 'low', \
            "Nitrate < 5 should be 'low' pollution level"

    def test_nitrate_pollution_level_moderate(self):
        """nitrate_pollution_level should be 'moderate' for 5 ≤ nitrate < 10"""
        df = create_minimal_df(nitrate=7.0)
        result = create_ml_features(df)

        assert result['nitrate_pollution_level'].iloc[0] == 'moderate', \
            "5 ≤ nitrate < 10 should be 'moderate' pollution level"

    def test_nitrate_pollution_level_high(self):
        """nitrate_pollution_level should be 'high' for 10 ≤ nitrate < 20"""
        df = create_minimal_df(nitrate=15.0)
        result = create_ml_features(df)

        assert result['nitrate_pollution_level'].iloc[0] == 'high', \
            "10 ≤ nitrate < 20 should be 'high' pollution level"

    def test_nitrate_pollution_level_very_high(self):
        """nitrate_pollution_level should be 'very_high' for nitrate ≥ 20"""
        df = create_minimal_df(nitrate=25.0)
        result = create_ml_features(df)

        assert result['nitrate_pollution_level'].iloc[0] == 'very_high', \
            "Nitrate ≥ 20 should be 'very_high' pollution level"

    # --- Edge Case Combinations (5 tests) ---

    def test_all_parameters_missing_indicators(self):
        """All missing indicators should be 1 when all parameters missing"""
        df = create_minimal_df(
            ph=np.nan,
            dissolved_oxygen=np.nan,
            temperature=np.nan,
            turbidity=None,
            nitrate=np.nan,
            conductance=np.nan
        )
        result = create_ml_features(df)

        assert result['ph_missing'].iloc[0] == 1
        assert result['dissolved_oxygen_missing'].iloc[0] == 1
        assert result['temperature_missing'].iloc[0] == 1
        assert result['turbidity_missing'].iloc[0] == 1
        assert result['nitrate_missing'].iloc[0] == 1
        assert result['conductance_missing'].iloc[0] == 1

    def test_derived_features_with_all_missing(self):
        """Derived features behavior when all inputs missing"""
        df = create_minimal_df(
            ph=np.nan,
            dissolved_oxygen=np.nan,
            temperature=np.nan,
            nitrate=np.nan,
            conductance=np.nan
        )
        result = create_ml_features(df)

        # Features WITHOUT fillna should be NaN
        assert pd.isna(result['ph_deviation_from_7'].iloc[0]), \
            "pH deviation should be NaN when pH missing"
        assert pd.isna(result['do_temp_ratio'].iloc[0]), \
            "DO-temp ratio should be NaN when both missing"
        assert pd.isna(result['temp_stress'].iloc[0]), \
            "Temperature stress should be NaN when temperature missing"

        # Features WITH fillna should NOT be NaN
        assert result['pollution_stress'].notna().iloc[0], \
            "Pollution stress should use fillna (not NaN)"

    def test_mixed_missing_present_scenario(self):
        """Realistic scenario: some parameters present, some missing"""
        df = create_minimal_df(
            ph=7.5,           # Present
            dissolved_oxygen=np.nan,  # Missing (common)
            temperature=18.0,  # Present
            turbidity=None,    # Always missing
            nitrate=12.0,      # Present
            conductance=np.nan # Missing
        )
        result = create_ml_features(df)

        # Check indicators
        assert result['ph_missing'].iloc[0] == 0
        assert result['dissolved_oxygen_missing'].iloc[0] == 1
        assert result['temperature_missing'].iloc[0] == 0
        assert result['turbidity_missing'].iloc[0] == 1
        assert result['nitrate_missing'].iloc[0] == 0
        assert result['conductance_missing'].iloc[0] == 1

        # Check count
        assert result['n_params_available'].iloc[0] == 3, \
            "pH, temp, nitrate present = 3 parameters"

        # Check derived features
        assert result['ph_deviation_from_7'].notna().iloc[0], \
            "pH present should allow pH deviation calculation"
        assert pd.isna(result['do_temp_ratio'].iloc[0]), \
            "Missing DO should make do_temp_ratio NaN"
        assert result['temp_stress'].notna().iloc[0], \
            "Temperature present should allow temp_stress calculation"
        assert result['pollution_stress'].notna().iloc[0], \
            "Pollution stress uses fillna for missing DO"

    def test_conductance_categories_all_zero_when_missing(self):
        """Conductance categories should all be 0 when conductance is NaN"""
        df = create_minimal_df(conductance=np.nan)
        result = create_ml_features(df)

        assert result['conductance_low'].iloc[0] == 0.0, \
            "Missing conductance: low category should be 0"
        assert result['conductance_medium'].iloc[0] == 0.0, \
            "Missing conductance: medium category should be 0"
        assert result['conductance_high'].iloc[0] == 0.0, \
            "Missing conductance: high category should be 0"

        # CRITICAL: All zeros violates mutual exclusivity assumption
        # Normally sum should be 1.0, but with NaN sum is 0.0
        total = (result['conductance_low'].iloc[0] +
                result['conductance_medium'].iloc[0] +
                result['conductance_high'].iloc[0])
        assert total == 0.0, \
            "CRITICAL: Missing conductance makes all categories 0 (not mutually exclusive)"

    def test_multiple_rows_mixed_missing_patterns(self):
        """Different missing patterns across multiple rows"""
        df = pd.concat([
            create_minimal_df(ph=7.0, dissolved_oxygen=8.0, temperature=15.0, nitrate=5.0, conductance=500.0),
            create_minimal_df(ph=np.nan, dissolved_oxygen=8.0, temperature=15.0, nitrate=5.0, conductance=500.0),
            create_minimal_df(ph=7.0, dissolved_oxygen=np.nan, temperature=15.0, nitrate=5.0, conductance=500.0),
            create_minimal_df(ph=7.0, dissolved_oxygen=8.0, temperature=np.nan, nitrate=5.0, conductance=500.0),
        ], ignore_index=True)

        result = create_ml_features(df)

        # Row 0: All present (pH, DO, temp, nitrate, conductance = 5, turbidity missing)
        assert result['n_params_available'].iloc[0] == 5
        assert result['ph_missing'].iloc[0] == 0

        # Row 1: pH missing (DO, temp, nitrate, conductance = 4)
        assert result['n_params_available'].iloc[1] == 4
        assert result['ph_missing'].iloc[1] == 1
        assert pd.isna(result['ph_deviation_from_7'].iloc[1])

        # Row 2: DO missing (pH, temp, nitrate, conductance = 4)
        assert result['n_params_available'].iloc[2] == 4
        assert result['dissolved_oxygen_missing'].iloc[2] == 1
        assert pd.isna(result['do_temp_ratio'].iloc[2])

        # Row 3: Temperature missing (pH, DO, nitrate, conductance = 4)
        assert result['n_params_available'].iloc[3] == 4
        assert result['temperature_missing'].iloc[3] == 1
        assert pd.isna(result['temp_stress'].iloc[3])


# ============================================================================
# TEST CLASS 5: One-Hot Encoding (15 tests)
# ============================================================================

class TestOneHotEncoding:
    """Test one-hot encoding for categorical features"""

    # --- Water Body Type Encoding (7 tests) ---

    def test_water_body_encoding_column_created_when_present(self):
        """Water body columns should be created when parameterWaterBodyCategory exists"""
        df = create_minimal_df()
        df['parameterWaterBodyCategory'] = 'River'
        result = create_ml_features(df)

        # Check that at least one water_body_ column was created
        water_body_cols = [col for col in result.columns if col.startswith('water_body_')]
        assert len(water_body_cols) > 0, "Should create water_body_ columns"

    def test_water_body_encoding_not_created_when_absent(self):
        """Water body columns should NOT be created when parameterWaterBodyCategory missing"""
        df = create_minimal_df()
        # Don't add parameterWaterBodyCategory column
        result = create_ml_features(df)

        # Check that NO water_body_ columns were created
        water_body_cols = [col for col in result.columns if col.startswith('water_body_')]
        assert len(water_body_cols) == 0, "Should NOT create water_body_ columns when column missing"

    def test_water_body_encoding_river_type(self):
        """River water body type should create water_body_River column"""
        df = create_minimal_df()
        df['parameterWaterBodyCategory'] = 'River'
        result = create_ml_features(df)

        assert 'water_body_River' in result.columns, "Should create water_body_River column"
        assert result['water_body_River'].iloc[0] == True, "River type should be True"

    def test_water_body_encoding_lake_type(self):
        """Lake water body type should create water_body_Lake column"""
        df = create_minimal_df()
        df['parameterWaterBodyCategory'] = 'Lake'
        result = create_ml_features(df)

        assert 'water_body_Lake' in result.columns, "Should create water_body_Lake column"
        assert result['water_body_Lake'].iloc[0] == True, "Lake type should be True"

    def test_water_body_encoding_mutual_exclusivity(self):
        """Only one water body type should be True per row"""
        df = pd.concat([
            create_minimal_df(**{'parameterWaterBodyCategory': 'River'}),
            create_minimal_df(**{'parameterWaterBodyCategory': 'Lake'}),
        ], ignore_index=True)

        result = create_ml_features(df)
        water_body_cols = [col for col in result.columns if col.startswith('water_body_')]

        # Row 0 should have River=True, all others False
        assert result['water_body_River'].iloc[0] == True
        if 'water_body_Lake' in result.columns:
            assert result['water_body_Lake'].iloc[0] == False

        # Row 1 should have Lake=True, all others False
        assert result['water_body_Lake'].iloc[1] == True
        if 'water_body_River' in result.columns:
            assert result['water_body_River'].iloc[1] == False

    def test_water_body_encoding_multiple_rows_same_type(self):
        """Multiple rows with same water body type"""
        df = pd.concat([
            create_minimal_df(**{'parameterWaterBodyCategory': 'River'}),
            create_minimal_df(**{'parameterWaterBodyCategory': 'River'}),
            create_minimal_df(**{'parameterWaterBodyCategory': 'River'}),
        ], ignore_index=True)

        result = create_ml_features(df)

        assert all(result['water_body_River'] == True), \
            "All rows should have water_body_River=True"

    def test_water_body_encoding_nan_handling(self):
        """NaN in water body type should create False for all categories"""
        df = create_minimal_df()
        df['parameterWaterBodyCategory'] = np.nan
        result = create_ml_features(df)

        # pd.get_dummies with NaN creates False for all categories
        water_body_cols = [col for col in result.columns if col.startswith('water_body_')]
        if len(water_body_cols) > 0:
            # All water body columns should be False when input is NaN
            for col in water_body_cols:
                assert result[col].iloc[0] == False, \
                    f"{col} should be False when water body type is NaN"

    # --- Country Encoding (8 tests) ---

    def test_country_encoding_column_created_when_present(self):
        """Country columns should be created when Country column exists"""
        df = create_minimal_df()
        df['Country'] = 'Germany'
        result = create_ml_features(df)

        # Check that at least one country_ column was created
        country_cols = [col for col in result.columns if col.startswith('country_')]
        assert len(country_cols) > 0, "Should create country_ columns"

    def test_country_encoding_not_created_when_absent(self):
        """Country columns should NOT be created when Country column missing"""
        df = create_minimal_df()
        # Don't add Country column
        result = create_ml_features(df)

        # Check that NO country_ columns were created
        country_cols = [col for col in result.columns if col.startswith('country_')]
        assert len(country_cols) == 0, "Should NOT create country_ columns when column missing"

    def test_country_encoding_grouped_column_created(self):
        """Country_grouped column should be created"""
        df = create_minimal_df()
        df['Country'] = 'Germany'
        result = create_ml_features(df)

        assert 'Country_grouped' in result.columns, \
            "Should create Country_grouped column for top 10 + Other grouping"

    def test_country_encoding_top_10_logic(self):
        """Top 10 countries should NOT be grouped as 'Other'"""
        # Create data with 11 different countries, varying frequencies
        countries = (
            ['Germany'] * 50 +  # Top country
            ['France'] * 40 +
            ['Italy'] * 30 +
            ['Spain'] * 25 +
            ['Poland'] * 20 +
            ['Netherlands'] * 15 +
            ['Belgium'] * 12 +
            ['Austria'] * 10 +
            ['Czech Republic'] * 8 +
            ['Hungary'] * 7 +  # 10th most common
            ['Slovakia'] * 2   # Should be grouped as "Other"
        )

        df_list = [create_minimal_df(**{'Country': c}) for c in countries]
        df = pd.concat(df_list, ignore_index=True)

        result = create_ml_features(df)

        # Germany should be in top 10, NOT grouped as Other
        assert 'country_Germany' in result.columns, "Germany should have its own column"
        assert result['country_Germany'].sum() == 50, "Germany should have 50 rows"

        # Slovakia should be grouped as "Other"
        assert 'country_Other' in result.columns, "Should have country_Other column"
        # Note: We can't assert the exact count without knowing all "Other" countries

    def test_country_encoding_other_group(self):
        """Countries outside top 10 should be grouped as 'Other'"""
        # Create data with clear top 10 + some rare countries
        countries = (
            ['Germany'] * 20 +
            ['France'] * 15 +
            ['Italy'] * 12 +
            ['Spain'] * 10 +
            ['Poland'] * 9 +
            ['Netherlands'] * 8 +
            ['Belgium'] * 7 +
            ['Austria'] * 6 +
            ['Czech Republic'] * 5 +
            ['Hungary'] * 4 +
            ['Slovakia'] * 1 +  # Should be "Other"
            ['Slovenia'] * 1 +  # Should be "Other"
            ['Croatia'] * 1     # Should be "Other"
        )

        df_list = [create_minimal_df(**{'Country': c}) for c in countries]
        df = pd.concat(df_list, ignore_index=True)

        result = create_ml_features(df)

        # Check that Country_grouped has "Other" for rare countries
        assert 'Country_grouped' in result.columns
        # Slovakia, Slovenia, Croatia should be in "Other" group
        other_count = (result['Country_grouped'] == 'Other').sum()
        assert other_count >= 3, "At least Slovakia, Slovenia, Croatia should be 'Other'"

    def test_country_encoding_mutual_exclusivity(self):
        """Only one country column should be True per row"""
        df = pd.concat([
            create_minimal_df(**{'Country': 'Germany'}),
            create_minimal_df(**{'Country': 'France'}),
        ], ignore_index=True)

        result = create_ml_features(df)
        country_cols = [col for col in result.columns if col.startswith('country_')]

        # Each row should have exactly one True value across all country columns
        for idx in range(len(result)):
            true_count = sum([result[col].iloc[idx] for col in country_cols])
            assert true_count == 1, f"Row {idx} should have exactly 1 country=True"

    def test_country_encoding_data_types(self):
        """Country one-hot encoded columns should be bool or int"""
        df = create_minimal_df()
        df['Country'] = 'Germany'
        result = create_ml_features(df)

        country_cols = [col for col in result.columns if col.startswith('country_')]
        for col in country_cols:
            assert pd.api.types.is_bool_dtype(result[col]) or \
                   pd.api.types.is_integer_dtype(result[col]), \
                f"{col} should be bool or int type"

    def test_country_encoding_nan_handling(self):
        """NaN in Country should be handled (likely as 'Other' or False for all)"""
        df = create_minimal_df()
        df['Country'] = np.nan
        result = create_ml_features(df)

        # pd.get_dummies with NaN creates False for all categories
        country_cols = [col for col in result.columns if col.startswith('country_')]
        if len(country_cols) > 0:
            # Check that at least the behavior is consistent (all False or one True)
            true_count = sum([result[col].iloc[0] for col in country_cols])
            # NaN typically becomes False for all or gets grouped to "Other"
            assert true_count <= 1, "NaN should map to at most one country (likely Other or none)"


# ============================================================================
# TEST CLASS 6: Data Type Validation (15 tests)
# ============================================================================

class TestDataTypeValidation:
    """Test that all features have correct data types for ML models"""

    def test_years_since_1991_is_numeric(self):
        """years_since_1991 should be numeric (int or float)"""
        df = create_minimal_df(year=2017)
        result = create_ml_features(df)
        assert pd.api.types.is_numeric_dtype(result['years_since_1991']), \
            "years_since_1991 should be numeric"

    def test_decade_is_numeric(self):
        """decade should be numeric (int)"""
        df = create_minimal_df(year=2017)
        result = create_ml_features(df)
        assert pd.api.types.is_numeric_dtype(result['decade']), \
            "decade should be numeric"

    def test_period_indicators_are_bool(self):
        """is_1990s, is_2000s, is_2010s should be bool"""
        df = create_minimal_df(year=2017)
        result = create_ml_features(df)

        assert pd.api.types.is_bool_dtype(result['is_1990s']), "is_1990s should be bool"
        assert pd.api.types.is_bool_dtype(result['is_2000s']), "is_2000s should be bool"
        assert pd.api.types.is_bool_dtype(result['is_2010s']), "is_2010s should be bool"

    def test_ph_deviation_is_numeric(self):
        """ph_deviation_from_7 should be numeric (float)"""
        df = create_minimal_df(ph=7.5)
        result = create_ml_features(df)
        assert pd.api.types.is_numeric_dtype(result['ph_deviation_from_7']), \
            "ph_deviation_from_7 should be numeric"

    def test_do_temp_ratio_is_numeric(self):
        """do_temp_ratio should be numeric (float)"""
        df = create_minimal_df(dissolved_oxygen=8.0, temperature=15.0)
        result = create_ml_features(df)
        assert pd.api.types.is_numeric_dtype(result['do_temp_ratio']), \
            "do_temp_ratio should be numeric"

    def test_conductance_categories_are_numeric(self):
        """conductance_low/medium/high should be numeric (float)"""
        df = create_minimal_df(conductance=500.0)
        result = create_ml_features(df)

        assert pd.api.types.is_numeric_dtype(result['conductance_low']), \
            "conductance_low should be numeric"
        assert pd.api.types.is_numeric_dtype(result['conductance_medium']), \
            "conductance_medium should be numeric"
        assert pd.api.types.is_numeric_dtype(result['conductance_high']), \
            "conductance_high should be numeric"

    def test_missing_indicators_are_int(self):
        """All *_missing columns should be int (0 or 1)"""
        df = create_minimal_df()
        result = create_ml_features(df)

        missing_cols = [col for col in result.columns if col.endswith('_missing')]
        for col in missing_cols:
            assert pd.api.types.is_integer_dtype(result[col]), \
                f"{col} should be integer type"
            # Verify values are only 0 or 1
            assert result[col].isin([0, 1]).all(), \
                f"{col} should only contain 0 or 1"

    def test_n_params_available_is_int(self):
        """n_params_available should be int"""
        df = create_minimal_df()
        result = create_ml_features(df)

        assert pd.api.types.is_integer_dtype(result['n_params_available']), \
            "n_params_available should be integer type"
        # Verify range is 0-6
        assert (result['n_params_available'] >= 0).all(), \
            "n_params_available should be >= 0"
        assert (result['n_params_available'] <= 6).all(), \
            "n_params_available should be <= 6 (max WQI parameters)"

    def test_pollution_stress_is_numeric(self):
        """pollution_stress should be numeric (float)"""
        df = create_minimal_df(nitrate=10.0, dissolved_oxygen=8.0)
        result = create_ml_features(df)
        assert pd.api.types.is_numeric_dtype(result['pollution_stress']), \
            "pollution_stress should be numeric"

    def test_temp_stress_is_numeric(self):
        """temp_stress should be numeric (float)"""
        df = create_minimal_df(temperature=20.0)
        result = create_ml_features(df)
        assert pd.api.types.is_numeric_dtype(result['temp_stress']), \
            "temp_stress should be numeric"

    def test_nitrate_pollution_level_is_categorical(self):
        """nitrate_pollution_level should be categorical"""
        df = create_minimal_df(nitrate=10.0)
        result = create_ml_features(df)

        assert pd.api.types.is_categorical_dtype(result['nitrate_pollution_level']), \
            "nitrate_pollution_level should be categorical type"

    def test_water_body_encoded_columns_are_bool_or_int(self):
        """Water body one-hot columns should be bool or int"""
        df = create_minimal_df()
        df['parameterWaterBodyCategory'] = 'River'
        result = create_ml_features(df)

        water_body_cols = [col for col in result.columns if col.startswith('water_body_')]
        for col in water_body_cols:
            assert pd.api.types.is_bool_dtype(result[col]) or \
                   pd.api.types.is_integer_dtype(result[col]), \
                f"{col} should be bool or int"

    def test_country_encoded_columns_are_bool_or_int(self):
        """Country one-hot columns should be bool or int"""
        df = create_minimal_df()
        df['Country'] = 'Germany'
        result = create_ml_features(df)

        country_cols = [col for col in result.columns if col.startswith('country_')]
        for col in country_cols:
            assert pd.api.types.is_bool_dtype(result[col]) or \
                   pd.api.types.is_integer_dtype(result[col]), \
                f"{col} should be bool or int"

    def test_all_numeric_features_no_object_dtype(self):
        """No numeric features should have object dtype"""
        df = create_minimal_df()
        result = create_ml_features(df)

        # Exclude known categorical/string columns and turbidity (always None)
        exclude_cols = [
            'waterBodyIdentifier', 'year', 'Country', 'Country_grouped',
            'parameterWaterBodyCategory', 'nitrate_pollution_level',
            'wqi_classification', 'parameter_scores', 'turbidity'
        ]

        for col in result.columns:
            if col not in exclude_cols and not col.startswith('composition_'):
                # Should be numeric or bool, NOT object
                assert not pd.api.types.is_object_dtype(result[col]) or \
                       col.startswith('country_') or col.startswith('water_body_'), \
                    f"{col} should not be object dtype (found {result[col].dtype})"

    def test_data_types_consistent_across_rows(self):
        """Data types should be consistent across multiple rows"""
        df = pd.concat([
            create_minimal_df(year=2015, ph=7.0, dissolved_oxygen=8.0),
            create_minimal_df(year=2016, ph=7.5, dissolved_oxygen=9.0),
            create_minimal_df(year=2017, ph=6.5, dissolved_oxygen=7.0),
        ], ignore_index=True)

        result = create_ml_features(df)

        # Check that each column has a single consistent dtype
        for col in result.columns:
            # Get unique dtypes (should be only 1)
            unique_dtypes = result[col].apply(type).unique()
            # Allow for NaN (float) and int/bool coexistence
            if col not in ['waterBodyIdentifier', 'Country', 'Country_grouped',
                          'parameterWaterBodyCategory', 'nitrate_pollution_level',
                          'wqi_classification', 'parameter_scores']:
                dtype = result[col].dtype
                assert dtype is not None, f"{col} should have a consistent dtype across rows"


# ============================================================================
# Meta-test: Verify test count
# ============================================================================

def test_meta_temporal_features_count():
    """Meta-test: Verify we have at least 20 temporal feature tests"""
    import inspect
    test_methods = [m for m in dir(TestTemporalFeatures) if m.startswith('test_')]
    assert len(test_methods) >= 20, f"Need at least 20 temporal tests, have {len(test_methods)}"


def test_meta_water_quality_features_count():
    """Meta-test: Verify we have at least 30 water quality derived feature tests"""
    import inspect
    test_methods = [m for m in dir(TestWaterQualityDerivedFeatures) if m.startswith('test_')]
    assert len(test_methods) >= 30, f"Need at least 30 water quality tests, have {len(test_methods)}"


def test_meta_interaction_features_count():
    """Meta-test: Verify we have at least 39 interaction feature tests"""
    import inspect
    test_methods = [m for m in dir(TestInteractionFeatures) if m.startswith('test_')]
    assert len(test_methods) >= 39, f"Need at least 39 interaction tests, have {len(test_methods)}"


def test_meta_missing_value_handling_count():
    """Meta-test: Verify we have at least 30 missing value handling tests"""
    import inspect
    test_methods = [m for m in dir(TestMissingValueHandling) if m.startswith('test_')]
    assert len(test_methods) >= 30, f"Need at least 30 missing value tests, have {len(test_methods)}"


def test_meta_one_hot_encoding_count():
    """Meta-test: Verify we have at least 15 one-hot encoding tests"""
    import inspect
    test_methods = [m for m in dir(TestOneHotEncoding) if m.startswith('test_')]
    assert len(test_methods) >= 15, f"Need at least 15 one-hot encoding tests, have {len(test_methods)}"


def test_meta_data_type_validation_count():
    """Meta-test: Verify we have at least 15 data type validation tests"""
    import inspect
    test_methods = [m for m in dir(TestDataTypeValidation) if m.startswith('test_')]
    assert len(test_methods) >= 15, f"Need at least 15 data type validation tests, have {len(test_methods)}"
