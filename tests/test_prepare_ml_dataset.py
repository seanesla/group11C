"""
Unit Tests for prepare_ml_dataset() Function

Tests verify the complete pipeline:
- CSV → features → WQI → ML-ready dataset
- Saved CSV has converted nitrate
- Nitrate mean in correct range
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.feature_engineering import (
    prepare_ml_dataset,
    NITRATE_NO3_TO_N
)


class TestPrepareMLDataset:
    """Test prepare_ml_dataset() function."""

    def test_prepare_ml_dataset_end_to_end(self):
        """Test complete pipeline produces ML-ready dataset."""
        kaggle_file = Path("data/raw/waterPollution.csv")

        if not kaggle_file.exists():
            pytest.skip("Kaggle CSV doesn't exist")

        # Run full pipeline
        df = prepare_ml_dataset()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0, "Should produce non-empty dataset"

        # Should have WQI columns
        assert 'wqi_score' in df.columns
        assert 'wqi_classification' in df.columns
        assert 'is_safe' in df.columns

        # Should have nitrate column
        assert 'nitrate' in df.columns

    def test_prepare_ml_dataset_saves_with_converted_nitrate(self):
        """Test that saved CSV has converted nitrate values."""
        processed_file = Path("data/processed/ml_features.csv")

        if not processed_file.exists():
            pytest.skip("Processed file doesn't exist - run prepare_ml_dataset() first")

        df = pd.read_csv(processed_file)

        if 'nitrate' not in df.columns or df['nitrate'].isna().all():
            pytest.skip("No nitrate data in processed file")

        nitrate_mean = df['nitrate'].mean()

        # After fix: mean should be ~2.86 mg/L as N (not ~12.65 mg{NO3}/L)
        assert 1.0 < nitrate_mean < 10.0, \
            f"Nitrate mean {nitrate_mean:.2f} should be 1-10 mg/L as N (converted)"

        # Should NOT be in mg{NO3}/L range
        assert nitrate_mean < 11.0, \
            f"Nitrate mean {nitrate_mean:.2f} should not be in mg{{NO3}}/L range (10+)"

    def test_prepare_ml_dataset_nitrate_sample_values(self):
        """Test that sample nitrate values are in mg/L as N scale."""
        processed_file = Path("data/processed/ml_features.csv")

        if not processed_file.exists():
            pytest.skip("Processed file doesn't exist")

        df = pd.read_csv(processed_file)

        if 'nitrate' not in df.columns or df['nitrate'].isna().all():
            pytest.skip("No nitrate data")

        # Get sample values
        sample_values = df[df['nitrate'].notna()]['nitrate'].head(100).tolist()

        # After conversion, max should be ~42.5 mg/L as N (188.24 × 0.2258)
        max_val = max(sample_values)
        assert max_val < 100.0, \
            f"Max sample value {max_val:.2f} should be < 100 mg/L as N"

        # Most values should be < EPA MCL (10 mg/L as N)
        below_mcl = sum(1 for v in sample_values if v < 10.0)
        pct_below = (below_mcl / len(sample_values)) * 100
        assert pct_below > 30, \
            f"At least 30% of values should be < EPA MCL, got {pct_below:.1f}%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
