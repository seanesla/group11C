"""
Test that WQI calculation receives correct nitrate values (not DataFrame column values).

CRITICAL SAFETY TEST: Previous bug caused double conversion where nitrate values
were converted twice (once in extract_wqi_parameters, again in calculate_wqi_labels).
This test verifies that WQI calculation receives the correct single-converted values.

This test is MORE IMPORTANT than tests that only check DataFrame columns, because
the bug didn't affect DataFrame values - it only affected what was passed to calculate_wqi().
"""

import pytest
import pandas as pd
from src.preprocessing.feature_engineering import (
    extract_wqi_parameters,
    calculate_wqi_labels,
    NITRATE_NO3_TO_N,
    load_kaggle_data
)
from src.utils.wqi_calculator import WQICalculator
from pathlib import Path


class TestWQICalculationNitrateInput:
    """Verify WQI calculation receives correctly converted nitrate values."""

    def test_wqi_receives_single_converted_nitrate_not_double(self):
        """
        CRITICAL: Verify nitrate is converted once, not twice, for WQI calculation.

        Double conversion bug:
        - 12.65 mg{NO3}/L → 2.86 mg/L as N (correct, first conversion)
        - 2.86 mg/L as N → 0.645 mg/L as N (WRONG, second conversion)

        This test verifies that WQI calculation uses 2.86, not 0.645.
        """
        kaggle_file = Path("data/raw/waterPollution.csv")
        if not kaggle_file.exists():
            pytest.skip("Kaggle dataset not found")

        # Step 1: Load raw data and get sample nitrate value
        df_raw = load_kaggle_data()
        nitrate_records = df_raw[df_raw['observedPropertyDeterminandCode'] == 'CAS_14797-55-8']

        # Get a specific nitrate value for testing (use first record)
        raw_nitrate_no3 = float(nitrate_records.iloc[0]['resultMeanValue'])

        # Expected single conversion
        expected_nitrate_as_n = raw_nitrate_no3 * NITRATE_NO3_TO_N

        # Expected double conversion (BUG)
        bugged_double_conversion = expected_nitrate_as_n * NITRATE_NO3_TO_N

        # Step 2: Extract parameters (converts once)
        df_extracted = extract_wqi_parameters(df_raw)

        # Step 3: Calculate WQI (should NOT convert again)
        df_wqi = calculate_wqi_labels(df_extracted)

        # Step 4: Get a row with nitrate data
        nitrate_row = df_wqi[df_wqi['nitrate'].notna()].iloc[0]

        # Step 5: Calculate expected WQI with CORRECT single-converted value
        calc = WQICalculator()
        expected_wqi, expected_scores, _ = calc.calculate_wqi(
            ph=nitrate_row['ph'] if pd.notna(nitrate_row['ph']) else None,
            dissolved_oxygen=nitrate_row['dissolved_oxygen'] if pd.notna(nitrate_row['dissolved_oxygen']) else None,
            temperature=nitrate_row['temperature'] if pd.notna(nitrate_row['temperature']) else None,
            turbidity=nitrate_row['turbidity'] if pd.notna(nitrate_row['turbidity']) else None,
            nitrate=nitrate_row['nitrate'],  # Should be single-converted
            conductance=nitrate_row['conductance'] if pd.notna(nitrate_row['conductance']) else None
        )

        # Step 6: Calculate WQI that would result from DOUBLE conversion (bug)
        bugged_wqi, bugged_scores, _ = calc.calculate_wqi(
            ph=nitrate_row['ph'] if pd.notna(nitrate_row['ph']) else None,
            dissolved_oxygen=nitrate_row['dissolved_oxygen'] if pd.notna(nitrate_row['dissolved_oxygen']) else None,
            temperature=nitrate_row['temperature'] if pd.notna(nitrate_row['temperature']) else None,
            turbidity=nitrate_row['turbidity'] if pd.notna(nitrate_row['turbidity']) else None,
            nitrate=nitrate_row['nitrate'] * NITRATE_NO3_TO_N,  # Double conversion
            conductance=nitrate_row['conductance'] if pd.notna(nitrate_row['conductance']) else None
        )

        # Step 7: Verify actual WQI matches expected (single conversion), NOT bugged
        actual_wqi = nitrate_row['wqi_score']
        actual_nitrate_score = nitrate_row['parameter_scores']['nitrate']

        # Assert WQI matches single-converted expectation
        assert abs(actual_wqi - expected_wqi) < 0.1, \
            f"WQI mismatch! Expected {expected_wqi:.2f} (single conversion), got {actual_wqi:.2f}"

        # Assert nitrate score matches single-converted expectation
        assert abs(actual_nitrate_score - expected_scores['nitrate']) < 0.1, \
            f"Nitrate score mismatch! Expected {expected_scores['nitrate']:.2f}, got {actual_nitrate_score:.2f}"

        # Assert WQI does NOT match double-conversion (bug) value
        assert abs(actual_wqi - bugged_wqi) > 1.0, \
            f"CRITICAL BUG: WQI {actual_wqi:.2f} matches double-conversion value {bugged_wqi:.2f}! " \
            f"This means the bug still exists!"

        # Assert nitrate score does NOT match double-conversion value
        assert abs(actual_nitrate_score - bugged_scores['nitrate']) > 5.0, \
            f"CRITICAL BUG: Nitrate score {actual_nitrate_score:.2f} matches double-conversion " \
            f"value {bugged_scores['nitrate']:.2f}! Bug still exists!"

    def test_wqi_scores_differ_between_single_and_double_conversion(self):
        """
        Verify that single vs double conversion produces measurably different WQI scores.

        This ensures our test can actually detect the bug.
        """
        calc = WQICalculator()

        # Test value: 12.0 mg{NO3}/L
        nitrate_no3 = 12.0
        nitrate_single = nitrate_no3 * NITRATE_NO3_TO_N  # 2.71 mg/L as N
        nitrate_double = nitrate_single * NITRATE_NO3_TO_N  # 0.61 mg/L as N (BUG)

        # Calculate WQI with other parameters held constant
        wqi_single, scores_single, _ = calc.calculate_wqi(
            ph=7.0, dissolved_oxygen=7.0, temperature=20.0,
            nitrate=nitrate_single, conductance=600
        )

        wqi_double, scores_double, _ = calc.calculate_wqi(
            ph=7.0, dissolved_oxygen=7.0, temperature=20.0,
            nitrate=nitrate_double, conductance=600
        )

        # Verify scores are meaningfully different
        assert abs(wqi_single - wqi_double) > 2.0, \
            f"WQI scores too similar! Single: {wqi_single:.1f}, Double: {wqi_double:.1f}. " \
            f"Test may not detect bug."

        assert abs(scores_single['nitrate'] - scores_double['nitrate']) > 10.0, \
            f"Nitrate scores too similar! Single: {scores_single['nitrate']:.1f}, " \
            f"Double: {scores_double['nitrate']:.1f}. Test may not detect bug."

        # Double conversion should inflate scores (lower nitrate looks better)
        assert wqi_double > wqi_single, \
            "Double conversion should inflate WQI score"
        assert scores_double['nitrate'] > scores_single['nitrate'], \
            "Double conversion should inflate nitrate score"


class TestWQICalculationConsistency:
    """Verify WQI calculation is consistent across pipeline."""

    def test_wqi_scores_match_manual_calculation(self):
        """
        Verify that WQI scores in the dataset match manual calculation using WQICalculator.

        This catches if calculate_wqi_labels() is passing wrong values to calculate_wqi().
        """
        kaggle_file = Path("data/raw/waterPollution.csv")
        if not kaggle_file.exists():
            pytest.skip("Kaggle dataset not found")

        # Load and process data
        df_raw = load_kaggle_data()
        df_extracted = extract_wqi_parameters(df_raw)
        df_wqi = calculate_wqi_labels(df_extracted)

        # Test first 10 rows with complete WQI data
        test_rows = df_wqi[df_wqi['wqi_score'].notna()].head(10)

        calc = WQICalculator()

        for idx, row in test_rows.iterrows():
            # Manual WQI calculation using row values
            manual_wqi, manual_scores, _ = calc.calculate_wqi(
                ph=row['ph'] if pd.notna(row['ph']) else None,
                dissolved_oxygen=row['dissolved_oxygen'] if pd.notna(row['dissolved_oxygen']) else None,
                temperature=row['temperature'] if pd.notna(row['temperature']) else None,
                turbidity=row['turbidity'] if pd.notna(row['turbidity']) else None,
                nitrate=row['nitrate'] if pd.notna(row['nitrate']) else None,
                conductance=row['conductance'] if pd.notna(row['conductance']) else None
            )

            # Compare with stored WQI
            stored_wqi = row['wqi_score']
            stored_nitrate_score = row['parameter_scores'].get('nitrate', 0)

            assert abs(manual_wqi - stored_wqi) < 0.1, \
                f"Row {idx}: WQI mismatch! Manual: {manual_wqi:.2f}, Stored: {stored_wqi:.2f}"

            if pd.notna(row['nitrate']) and 'nitrate' in manual_scores:
                assert abs(manual_scores['nitrate'] - stored_nitrate_score) < 0.1, \
                    f"Row {idx}: Nitrate score mismatch! Manual: {manual_scores['nitrate']:.2f}, " \
                    f"Stored: {stored_nitrate_score:.2f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
