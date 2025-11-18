"""
Integration Tests for Kaggle Nitrate Conversion Pipeline

Full end-to-end tests that verify nitrate conversion works correctly
through the complete pipeline: CSV → extraction → WQI → ML data.
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.feature_engineering import (
    load_kaggle_data,
    extract_wqi_parameters,
    calculate_wqi_labels,
    prepare_ml_dataset,
    NITRATE_NO3_TO_N
)


class TestKaggleNitrateIntegration:
    """Integration tests for complete nitrate conversion pipeline."""

    def test_full_pipeline_nitrate_conversion(self):
        """Test complete pipeline: CSV → features → WQI → ML data with nitrate conversion."""
        kaggle_file = Path("data/raw/waterPollution.csv")

        if not kaggle_file.exists():
            pytest.skip("Kaggle CSV doesn't exist")

        # Step 1: Load raw data and get original nitrate mean
        df_raw = load_kaggle_data()
        nitrate_records = df_raw[df_raw['observedPropertyDeterminandCode'] == 'CAS_14797-55-8']
        raw_values = pd.to_numeric(nitrate_records['resultMeanValue'], errors='coerce').dropna()
        raw_mean = raw_values.mean()

        # Step 2: Extract parameters (should convert nitrate)
        df_extracted = extract_wqi_parameters(df_raw)
        extracted_mean = df_extracted['nitrate'].mean()

        # Step 3: Calculate WQI
        df_wqi = calculate_wqi_labels(df_extracted)
        wqi_mean = df_wqi['nitrate'].mean()

        # Step 4: Full pipeline
        df_final = prepare_ml_dataset()
        final_mean = df_final['nitrate'].mean()

        # VERIFY: Conversion happened at extraction step
        expected_converted = raw_mean * NITRATE_NO3_TO_N
        assert abs(extracted_mean - expected_converted) < 1.0, \
            f"Extracted mean {extracted_mean:.2f} should equal raw {raw_mean:.2f} × {NITRATE_NO3_TO_N}"

        # VERIFY: Conversion persisted through WQI calculation
        assert abs(wqi_mean - extracted_mean) < 1.0, \
            f"WQI mean {wqi_mean:.2f} should equal extracted mean {extracted_mean:.2f}"

        # VERIFY: Conversion persisted to final dataset
        assert abs(final_mean - extracted_mean) < 1.0, \
            f"Final mean {final_mean:.2f} should equal extracted mean {extracted_mean:.2f}"

        # VERIFY: All means are in mg/L as N range (not mg{NO3}/L)
        assert 1.0 < extracted_mean < 10.0, "Extracted mean should be 1-10 mg/L as N"
        assert 1.0 < wqi_mean < 10.0, "WQI mean should be 1-10 mg/L as N"
        assert 1.0 < final_mean < 10.0, "Final mean should be 1-10 mg/L as N"

    def test_before_after_conversion_values(self):
        """Test that nitrate mean is correctly converted."""
        kaggle_file = Path("data/raw/waterPollution.csv")

        if not kaggle_file.exists():
            pytest.skip("Kaggle CSV doesn't exist")

        # Get raw mean
        df_raw = load_kaggle_data()
        nitrate_records = df_raw[df_raw['observedPropertyDeterminandCode'] == 'CAS_14797-55-8']
        raw_values = pd.to_numeric(nitrate_records['resultMeanValue'], errors='coerce').dropna()
        raw_mean = raw_values.mean()

        # Extract (should convert)
        df_extracted = extract_wqi_parameters(df_raw)
        converted_mean = df_extracted['nitrate'].mean()

        # Verify mean is converted correctly
        expected_converted_mean = raw_mean * NITRATE_NO3_TO_N
        assert abs(converted_mean - expected_converted_mean) < 0.5, \
            f"Converted mean {converted_mean:.2f} should equal raw mean {raw_mean:.2f} × {NITRATE_NO3_TO_N}"

        # Verify converted mean is much less than raw mean
        assert converted_mean < raw_mean * 0.3, \
            f"Converted mean {converted_mean:.2f} should be much less than raw mean {raw_mean:.2f}"

    def test_sample_value_verification(self):
        """Test specific known conversions are correct."""
        # Create test case with known values
        test_conversions = [
            (11.58, 2.61),   # Sample from Kaggle data
            (44.3, 10.0),    # EPA MCL equivalent
            (88.6, 20.0),    # 2× EPA MCL
            (188.24, 42.50), # Max value in dataset
        ]

        for raw_mg_no3, expected_mg_n in test_conversions:
            calculated = raw_mg_no3 * NITRATE_NO3_TO_N
            assert abs(calculated - expected_mg_n) < 0.5, \
                f"{raw_mg_no3} mg{{NO3}}/L should convert to ~{expected_mg_n} mg/L as N, got {calculated:.2f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
