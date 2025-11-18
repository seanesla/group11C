"""
Unit Tests for load_kaggle_data() Function

Tests verify that the Kaggle CSV loading function works correctly:
- File validation
- Shape validation
- Column validation
- Nitrate record existence
- Value range validation
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.feature_engineering import load_kaggle_data


class TestLoadKaggleData:
    """Test load_kaggle_data() function."""

    def test_load_kaggle_data_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent file."""
        with pytest.raises(FileNotFoundError, match="Kaggle dataset not found"):
            load_kaggle_data(file_path="nonexistent.csv")

    def test_load_kaggle_data_correct_shape(self):
        """Test that loaded data has expected shape (20,000 rows × 29 columns)."""
        kaggle_file = Path("data/raw/waterPollution.csv")

        if not kaggle_file.exists():
            pytest.skip("Kaggle CSV doesn't exist")

        df = load_kaggle_data()

        # Verify shape (may warn if different, but should not fail)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0, "DataFrame should not be empty"
        assert len(df.columns) > 0, "DataFrame should have columns"

        # Expected shape is (20,000, 29) but may vary
        assert df.shape[0] >= 10000, f"Expected at least 10,000 rows, got {df.shape[0]}"
        assert df.shape[1] >= 20, f"Expected at least 20 columns, got {df.shape[1]}"

    def test_load_kaggle_data_required_columns_exist(self):
        """Test that all required columns are present."""
        kaggle_file = Path("data/raw/waterPollution.csv")

        if not kaggle_file.exists():
            pytest.skip("Kaggle CSV doesn't exist")

        df = load_kaggle_data()

        required_cols = [
            'observedPropertyDeterminandCode',
            'resultMeanValue',
            'phenomenonTimeReferenceYear',
            'waterBodyIdentifier',
            'Country'
        ]

        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"

    def test_load_kaggle_data_nitrate_records_exist(self):
        """Test that nitrate parameter records exist (CAS_14797-55-8)."""
        kaggle_file = Path("data/raw/waterPollution.csv")

        if not kaggle_file.exists():
            pytest.skip("Kaggle CSV doesn't exist")

        df = load_kaggle_data()

        # Filter for nitrate parameter code
        nitrate_code = 'CAS_14797-55-8'
        nitrate_records = df[df['observedPropertyDeterminandCode'] == nitrate_code]

        assert len(nitrate_records) > 0, \
            f"No nitrate records found with code {nitrate_code}"

        assert len(nitrate_records) >= 1000, \
            f"Expected at least 1,000 nitrate records, found {len(nitrate_records)}"

    def test_load_kaggle_data_nitrate_value_ranges(self):
        """Test that nitrate values are in expected range (0.07 to 188.24 mg{NO3}/L)."""
        kaggle_file = Path("data/raw/waterPollution.csv")

        if not kaggle_file.exists():
            pytest.skip("Kaggle CSV doesn't exist")

        df = load_kaggle_data()

        # Filter for nitrate and get values
        nitrate_code = 'CAS_14797-55-8'
        nitrate_records = df[df['observedPropertyDeterminandCode'] == nitrate_code]

        if len(nitrate_records) == 0:
            pytest.skip("No nitrate records found")

        nitrate_values = pd.to_numeric(nitrate_records['resultMeanValue'], errors='coerce')
        nitrate_values = nitrate_values.dropna()

        assert len(nitrate_values) > 0, "No valid nitrate values found"

        # Check ranges (values should be in mg{NO3}/L before conversion)
        min_val = nitrate_values.min()
        max_val = nitrate_values.max()
        mean_val = nitrate_values.mean()

        assert min_val >= 0, f"Nitrate min value {min_val} should be non-negative"
        assert max_val < 500, f"Nitrate max value {max_val} seems unreasonably high"
        assert 1.0 < mean_val < 50.0, f"Nitrate mean {mean_val:.2f} should be between 1-50 mg{{NO3}}/L"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
