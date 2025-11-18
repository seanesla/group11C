"""
Bug Verification Tests for Nitrate Conversion

These tests DOCUMENT the critical bug where nitrate conversion (line 226) is
applied to WQI calculation but NOT saved to the dataframe, meaning ML models
are trained on UNCONVERTED mg{NO3}/L values instead of mg/L as N.

Expected behavior: All tests in this file should PASS, proving the bug exists.
After fix: Tests will need to be updated to verify the fix.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.feature_engineering import (
    NITRATE_NO3_TO_N,
    load_kaggle_data,
    extract_wqi_parameters,
    calculate_wqi_labels,
    prepare_ml_dataset
)


class TestNitrateBugDocumentation:
    """Document the bug: nitrate conversion not saved to dataframe."""

    def test_BUG_processed_data_nitrate_not_converted(self):
        """
        BUG PROOF: Processed ML data has nitrate mean ~12.6 mg{NO3}/L.

        Expected if bug is fixed: mean should be ~2.86 mg/L as N
        Actual with bug: mean is ~12.60 mg{NO3}/L (unconverted)
        """
        processed_file = Path("data/processed/ml_features.csv")

        if not processed_file.exists():
            pytest.skip("Processed data file doesn't exist yet")

        df = pd.read_csv(processed_file)

        if 'nitrate' not in df.columns:
            pytest.skip("Nitrate column doesn't exist in processed data")

        nitrate_mean = df['nitrate'].mean()

        # BUG: Mean should be ~2.86 mg/L as N (converted)
        # Actual: Mean is ~12.60 mg{NO3}/L (unconverted)
        assert nitrate_mean > 10.0, \
            f"BUG CONFIRMED: Nitrate mean is {nitrate_mean:.2f} mg{{NO3}}/L (should be ~2.86 mg/L as N)"

    def test_BUG_conversion_only_applied_to_wqi_not_dataframe(self):
        """
        BUG PROOF: Conversion at line 226 is used for WQI but not saved to df.

        This test mocks the WQI calculator to capture what value it receives,
        then verifies the dataframe still has the unconverted value.
        """
        # Create test data with known nitrate value
        test_df = pd.DataFrame({
            'waterBodyIdentifier': ['TEST001', 'TEST002'],
            'year': [2024, 2024],
            'ph': [7.0, 7.5],
            'dissolved_oxygen': [8.0, 9.0],
            'temperature': [20.0, 22.0],
            'nitrate': [44.3, 88.6],  # mg{NO3}/L (should convert to 10.0, 20.0)
            'conductance': [500.0, 600.0],
            'turbidity': [None, None]
        })

        original_nitrate = test_df['nitrate'].copy()

        # Mock WQICalculator to capture what it receives
        with patch('src.preprocessing.feature_engineering.WQICalculator') as MockWQI:
            mock_instance = MockWQI.return_value
            mock_instance.calculate_wqi.return_value = (80.0, {}, 'Good')

            result_df = calculate_wqi_labels(test_df)

            # Verify WQI calculator was called with CONVERTED values
            calls = mock_instance.calculate_wqi.call_args_list
            first_call_kwargs = calls[0][1]

            expected_converted = 44.3 * NITRATE_NO3_TO_N  # Should be ~10.0
            actual_received = first_call_kwargs['nitrate']

            assert abs(actual_received - expected_converted) < 0.1, \
                f"WQI received {actual_received:.2f}, expected {expected_converted:.2f}"

            # BUG: Dataframe nitrate should still be UNCONVERTED
            pd.testing.assert_series_equal(
                result_df['nitrate'],
                original_nitrate,
                check_names=False
            )

    def test_BUG_sample_values_still_in_mg_NO3(self):
        """
        BUG PROOF: Sample nitrate values in processed data are in mg{NO3}/L scale.

        If converted to mg/L as N, all values should be < 50 (max EPA drinking water level).
        With bug, values are 4.43x higher.
        """
        processed_file = Path("data/processed/ml_features.csv")

        if not processed_file.exists():
            pytest.skip("Processed data file doesn't exist yet")

        df = pd.read_csv(processed_file)

        if 'nitrate' not in df.columns or df['nitrate'].isna().all():
            pytest.skip("No nitrate data in processed file")

        # Get sample of nitrate values
        sample_values = df[df['nitrate'].notna()]['nitrate'].head(20).tolist()

        # BUG: If in mg/L as N, max should be ~42.5 (converted from 188.24)
        # With bug: Max is ~188.24 (unconverted)
        max_value = max(sample_values)

        assert max_value > 100.0, \
            f"BUG CONFIRMED: Max value {max_value:.2f} is in mg{{NO3}}/L scale (should be < 50 if converted)"

    def test_BUG_extraction_does_not_convert(self):
        """
        BUG PROOF: extract_wqi_parameters() returns unconverted nitrate values.

        This function should convert nitrate but currently doesn't.
        """
        kaggle_file = Path("data/raw/waterPollution.csv")

        if not kaggle_file.exists():
            pytest.skip("Kaggle CSV doesn't exist")

        # Load and extract
        df = load_kaggle_data()
        df_wide = extract_wqi_parameters(df)

        if df_wide.empty or 'nitrate' not in df_wide.columns:
            pytest.skip("No nitrate data extracted")

        nitrate_mean = df_wide['nitrate'].mean()

        # BUG: Mean should be ~2.86 mg/L as N after conversion
        # Actual: Mean is ~12.65 mg{NO3}/L (not converted)
        assert nitrate_mean > 10.0, \
            f"BUG CONFIRMED: Extracted nitrate mean {nitrate_mean:.2f} is unconverted"

    def test_BUG_conversion_ratio_is_identity(self):
        """
        BUG PROOF: Ratio between raw and processed nitrate is ~1.0 (should be 0.2258).

        If conversion worked, processed values would be 0.2258× raw values.
        With bug, processed ≈ raw (ratio ≈ 1.0).
        """
        kaggle_file = Path("data/raw/waterPollution.csv")
        processed_file = Path("data/processed/ml_features.csv")

        if not kaggle_file.exists() or not processed_file.exists():
            pytest.skip("Required files don't exist")

        # Get raw nitrate mean
        df_raw = load_kaggle_data()
        df_extracted = extract_wqi_parameters(df_raw)
        raw_mean = df_extracted['nitrate'].mean()

        # Get processed nitrate mean
        df_processed = pd.read_csv(processed_file)
        processed_mean = df_processed['nitrate'].mean()

        # Calculate ratio
        ratio = processed_mean / raw_mean

        # BUG: Ratio should be ~0.2258 (conversion factor)
        # Actual: Ratio is ~1.0 (no conversion applied)
        assert 0.9 < ratio < 1.1, \
            f"BUG CONFIRMED: Ratio {ratio:.4f} ≈ 1.0 (should be 0.2258 if converted)"


class TestNitrateFixVerification:
    """Verify the fix: nitrate conversion IS saved to dataframe."""

    def test_FIX_processed_data_nitrate_IS_converted(self):
        """
        FIX VERIFIED: Processed ML data has nitrate mean ~2.86 mg/L as N.

        After fix: mean should be ~2.86 mg/L as N (converted)
        Before fix: mean was ~12.60 mg{NO3}/L (unconverted)
        """
        processed_file = Path("data/processed/ml_features.csv")

        if not processed_file.exists():
            pytest.skip("Processed data file doesn't exist yet")

        df = pd.read_csv(processed_file)

        if 'nitrate' not in df.columns:
            pytest.skip("Nitrate column doesn't exist in processed data")

        nitrate_mean = df['nitrate'].mean()

        # FIX: Mean should be ~2.86 mg/L as N (converted)
        assert 1.0 < nitrate_mean < 10.0, \
            f"FIX VERIFIED: Nitrate mean is {nitrate_mean:.2f} mg/L as N (should be ~2.86)"

    def test_FIX_conversion_saved_to_dataframe(self):
        """
        FIX VERIFIED: Conversion is saved to dataframe, not just used for WQI.
        """
        kaggle_file = Path("data/raw/waterPollution.csv")

        if not kaggle_file.exists():
            pytest.skip("Kaggle CSV doesn't exist")

        # Load and extract (should convert)
        df_raw = load_kaggle_data()
        df_extracted = extract_wqi_parameters(df_raw)

        # Get raw mean
        nitrate_records = df_raw[df_raw['observedPropertyDeterminandCode'] == 'CAS_14797-55-8']
        raw_values = pd.to_numeric(nitrate_records['resultMeanValue'], errors='coerce').dropna()
        raw_mean = raw_values.mean()

        # Get extracted mean
        extracted_mean = df_extracted['nitrate'].mean()

        # FIX: Extracted mean should be raw_mean × 0.2258
        expected_converted = raw_mean * NITRATE_NO3_TO_N
        assert abs(extracted_mean - expected_converted) < 1.0, \
            f"FIX VERIFIED: Extracted mean {extracted_mean:.2f} equals raw {raw_mean:.2f} × {NITRATE_NO3_TO_N}"

    def test_FIX_sample_values_in_mg_as_N(self):
        """
        FIX VERIFIED: Sample nitrate values in processed data are in mg/L as N scale.

        After conversion, max should be ~42.5 mg/L as N.
        """
        processed_file = Path("data/processed/ml_features.csv")

        if not processed_file.exists():
            pytest.skip("Processed data file doesn't exist yet")

        df = pd.read_csv(processed_file)

        if 'nitrate' not in df.columns or df['nitrate'].isna().all():
            pytest.skip("No nitrate data in processed file")

        # Get sample of nitrate values
        sample_values = df[df['nitrate'].notna()]['nitrate'].head(20).tolist()

        # FIX: Max should be ~42.5 mg/L as N (converted from 188.24)
        max_value = max(sample_values)

        assert max_value < 100.0, \
            f"FIX VERIFIED: Max value {max_value:.2f} is in mg/L as N scale (< 100)"

    def test_FIX_extraction_DOES_convert(self):
        """
        FIX VERIFIED: extract_wqi_parameters() returns converted nitrate values.
        """
        kaggle_file = Path("data/raw/waterPollution.csv")

        if not kaggle_file.exists():
            pytest.skip("Kaggle CSV doesn't exist")

        # Load and extract
        df = load_kaggle_data()
        df_wide = extract_wqi_parameters(df)

        if df_wide.empty or 'nitrate' not in df_wide.columns:
            pytest.skip("No nitrate data extracted")

        nitrate_mean = df_wide['nitrate'].mean()

        # FIX: Mean should be ~2.86 mg/L as N after conversion
        assert 1.0 < nitrate_mean < 10.0, \
            f"FIX VERIFIED: Extracted nitrate mean {nitrate_mean:.2f} is converted"

    def test_FIX_conversion_ratio_is_0_2258(self):
        """
        FIX VERIFIED: Ratio between raw and processed nitrate is ~0.2258 (conversion factor).
        """
        kaggle_file = Path("data/raw/waterPollution.csv")
        processed_file = Path("data/processed/ml_features.csv")

        if not kaggle_file.exists() or not processed_file.exists():
            pytest.skip("Required files don't exist")

        # Get raw nitrate mean
        df_raw = load_kaggle_data()
        df_extracted = extract_wqi_parameters(df_raw)
        raw_mean = (df_extracted['nitrate'] / NITRATE_NO3_TO_N).mean()  # Un-convert to get original

        # Get processed nitrate mean
        df_processed = pd.read_csv(processed_file)
        processed_mean = df_processed['nitrate'].mean()

        # Calculate ratio
        ratio = processed_mean / raw_mean

        # FIX: Ratio should be ~0.2258 (conversion factor)
        assert 0.20 < ratio < 0.25, \
            f"FIX VERIFIED: Ratio {ratio:.4f} ≈ 0.2258 (conversion applied)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
