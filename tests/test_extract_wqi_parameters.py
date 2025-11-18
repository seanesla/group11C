"""
Unit Tests for extract_wqi_parameters() Function

CRITICAL: This function now includes nitrate conversion (line 182)
Tests verify:
- Parameter filtering
- Nitrate code mapping
- Pivot to wide format
- NITRATE CONVERSION from mg{NO3}/L to mg/L as N
- Turbidity column addition
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.feature_engineering import (
    load_kaggle_data,
    extract_wqi_parameters,
    NITRATE_NO3_TO_N
)


class TestExtractWQIParameters:
    """Test extract_wqi_parameters() function."""

    def test_extract_filters_parameter_codes(self):
        """Test that only WQI parameter codes are retained."""
        kaggle_file = Path("data/raw/waterPollution.csv")

        if not kaggle_file.exists():
            pytest.skip("Kaggle CSV doesn't exist")

        df = load_kaggle_data()
        df_wide = extract_wqi_parameters(df)

        # Should have standard WQI columns
        expected_params = ['ph', 'temperature', 'dissolved_oxygen', 'nitrate', 'conductance']

        for param in expected_params:
            if param in df_wide.columns:
                # If parameter exists, should have some non-null values
                assert df_wide[param].notna().any(), f"{param} column exists but all values are null"

    def test_extract_maps_nitrate_code_correctly(self):
        """Test that CAS_14797-55-8 is mapped to 'nitrate' column."""
        kaggle_file = Path("data/raw/waterPollution.csv")

        if not kaggle_file.exists():
            pytest.skip("Kaggle CSV doesn't exist")

        df = load_kaggle_data()
        df_wide = extract_wqi_parameters(df)

        assert 'nitrate' in df_wide.columns, "nitrate column should exist"

        # Should have nitrate values from Kaggle data
        nitrate_count = df_wide['nitrate'].notna().sum()
        assert nitrate_count > 0, "Should have some non-null nitrate values"

    def test_extract_pivots_to_wide_format(self):
        """Test that data is pivoted from long to wide format."""
        kaggle_file = Path("data/raw/waterPollution.csv")

        if not kaggle_file.exists():
            pytest.skip("Kaggle CSV doesn't exist")

        df_long = load_kaggle_data()
        df_wide = extract_wqi_parameters(df_long)

        # Wide format should have:
        # - Fewer rows (one per water body + year instead of one per parameter)
        # - More parameter columns
        assert len(df_wide) < len(df_long), \
            "Wide format should have fewer rows than long format"

        # Should have waterBodyIdentifier and year columns
        assert 'waterBodyIdentifier' in df_wide.columns
        assert 'year' in df_wide.columns

    def test_extract_CONVERTS_nitrate_to_mg_as_N(self):
        """
        CRITICAL TEST: Verify nitrate is converted from mg{NO3}/L to mg/L as N.

        This is THE key test that proves the fix works.
        Before fix: nitrate mean ~12.65 mg{NO3}/L
        After fix: nitrate mean ~2.86 mg/L as N
        """
        kaggle_file = Path("data/raw/waterPollution.csv")

        if not kaggle_file.exists():
            pytest.skip("Kaggle CSV doesn't exist")

        # Load raw data and get unconverted nitrate mean
        df_raw = load_kaggle_data()
        nitrate_records = df_raw[df_raw['observedPropertyDeterminandCode'] == 'CAS_14797-55-8']
        raw_values = pd.to_numeric(nitrate_records['resultMeanValue'], errors='coerce').dropna()
        raw_mean = raw_values.mean()

        # Extract with conversion
        df_wide = extract_wqi_parameters(df_raw)
        converted_mean = df_wide['nitrate'].mean()

        # Verify conversion happened
        expected_converted_mean = raw_mean * NITRATE_NO3_TO_N

        assert abs(converted_mean - expected_converted_mean) < 0.5, \
            f"Converted mean {converted_mean:.2f} should be {expected_converted_mean:.2f} " \
            f"(raw {raw_mean:.2f} × {NITRATE_NO3_TO_N})"

        # Verify mean is in correct range for mg/L as N
        assert 1.0 < converted_mean < 10.0, \
            f"Converted mean {converted_mean:.2f} should be 1-10 mg/L as N"

        # Verify it's NOT in mg{NO3}/L range
        assert converted_mean < raw_mean * 0.5, \
            f"Converted mean {converted_mean:.2f} should be much less than raw mean {raw_mean:.2f}"

    def test_extract_adds_turbidity_none_column(self):
        """Test that turbidity column is added with all None values."""
        kaggle_file = Path("data/raw/waterPollution.csv")

        if not kaggle_file.exists():
            pytest.skip("Kaggle CSV doesn't exist")

        df = load_kaggle_data()
        df_wide = extract_wqi_parameters(df)

        assert 'turbidity' in df_wide.columns, "turbidity column should exist"
        assert df_wide['turbidity'].isna().all(), "All turbidity values should be None/NaN"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
