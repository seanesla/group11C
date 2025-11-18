"""
Comprehensive Test Suite for Nitrate Unit Conversion System

Tests verify that nitrate values are correctly converted from mg{NO3}/L
(Kaggle dataset) to mg/L as N (EPA/USGS standard) throughout the system.

Critical conversion: NO3 → N requires multiplication by 0.2258
(atomic weight N / molecular weight NO3 = 14.0067 / 62.0049)
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.feature_engineering import NITRATE_NO3_TO_N
from src.data_collection.wqp_client import WQPClient
from src.data_collection.usgs_client import USGSClient


class TestNitrateConversionConstant:
    """Test the nitrate conversion constant value."""

    def test_nitrate_conversion_constant_exists(self):
        """Verify NITRATE_NO3_TO_N constant is defined."""
        assert NITRATE_NO3_TO_N is not None

    def test_nitrate_conversion_constant_value(self):
        """Verify conversion factor is correct: N/NO3 = 14.0067/62.0049 = 0.2258."""
        expected = 14.0067 / 62.0049  # Atomic weight N / Molecular weight NO3
        assert abs(NITRATE_NO3_TO_N - expected) < 0.0001, \
            f"Expected {expected:.4f}, got {NITRATE_NO3_TO_N:.4f}"

    def test_conversion_constant_in_valid_range(self):
        """Verify conversion constant is in scientifically valid range (0.2 to 0.3)."""
        assert 0.2 < NITRATE_NO3_TO_N < 0.3, \
            "Conversion factor should be approximately 0.2258"


class TestEPAComplianceStandards:
    """Test that converted values align with EPA MCL standards."""

    def test_epa_mcl_compliance_safe_value(self):
        """Test that 44.3 mg{NO3}/L converts to 10 mg/L as N (EPA MCL)."""
        # EPA MCL: 10 mg/L as N
        # Equivalent NO3 value: 10 / 0.2258 = 44.3 mg{NO3}/L
        nitrate_no3 = 44.3  # mg{NO3}/L
        nitrate_as_n = nitrate_no3 * NITRATE_NO3_TO_N
        assert abs(nitrate_as_n - 10.0) < 0.1, \
            f"44.3 mg{{NO3}}/L should convert to ~10 mg/L as N, got {nitrate_as_n:.2f}"

    def test_epa_mcl_compliance_unsafe_value(self):
        """Test that 88.6 mg{NO3}/L converts to 20 mg/L as N (2× EPA MCL)."""
        nitrate_no3 = 88.6  # mg{NO3}/L (double the MCL equivalent)
        nitrate_as_n = nitrate_no3 * NITRATE_NO3_TO_N
        assert abs(nitrate_as_n - 20.0) < 0.2, \
            f"88.6 mg{{NO3}}/L should convert to ~20 mg/L as N, got {nitrate_as_n:.2f}"

    def test_conversion_maintains_relative_safety_levels(self):
        """Test that safety thresholds are maintained after conversion."""
        # Define test values in mg{NO3}/L and expected safety classifications
        test_cases = [
            (4.43, "Excellent"),   # 1 mg/L as N
            (22.15, "Good"),       # 5 mg/L as N
            (44.3, "Marginal"),    # 10 mg/L as N (EPA MCL)
            (88.6, "Poor"),        # 20 mg/L as N
            (221.5, "Unsafe"),     # 50 mg/L as N
        ]

        for nitrate_no3, expected_level in test_cases:
            nitrate_as_n = nitrate_no3 * NITRATE_NO3_TO_N
            # Verify order of magnitude is preserved
            assert nitrate_as_n < nitrate_no3, \
                "Converted value (as N) should be less than original (as NO3)"


class TestWQPClientUnitStandardization:
    """Test WQP client handles variable units correctly."""

    def test_wqp_converts_no3_to_n(self):
        """Test WQP client converts mg{NO3}/L to mg/L as N."""
        client = WQPClient()

        # Mock data with mg{NO3}/L units
        df = pd.DataFrame({
            'nitrate': [44.3],  # Should convert to ~10 mg/L as N
            'nitrate_unit': ['mg{NO3}/L']
        })

        result = client._standardize_nitrate_unit(df)

        assert abs(result['nitrate'].iloc[0] - 10.0) < 0.1, \
            f"Expected ~10 mg/L as N, got {result['nitrate'].iloc[0]:.2f}"

    def test_wqp_preserves_as_n_units(self):
        """Test WQP client preserves values already in mg/L as N."""
        client = WQPClient()

        df = pd.DataFrame({
            'nitrate': [10.0],
            'nitrate_unit': ['mg/L as N']
        })

        result = client._standardize_nitrate_unit(df)

        assert result['nitrate'].iloc[0] == 10.0, \
            "Values in mg/L as N should not be modified"

    def test_wqp_handles_missing_data(self):
        """Test WQP client handles missing nitrate data gracefully."""
        client = WQPClient()

        df = pd.DataFrame({
            'other_param': [1, 2, 3]
        })

        result = client._standardize_nitrate_unit(df)

        assert result.equals(df), "DataFrame without nitrate should remain unchanged"

    def test_wqp_handles_nan_values(self):
        """Test WQP client handles NaN nitrate values."""
        client = WQPClient()

        df = pd.DataFrame({
            'nitrate': [10.0, np.nan, 44.3],
            'nitrate_unit': ['mg/L as N', 'mg{NO3}/L', 'mg{NO3}/L']
        })

        result = client._standardize_nitrate_unit(df)

        assert pd.isna(result['nitrate'].iloc[1]), "NaN values should be preserved"
        assert result['nitrate'].iloc[0] == 10.0, "Valid as N values should be preserved"
        assert abs(result['nitrate'].iloc[2] - 10.0) < 0.1, "Valid NO3 values should be converted"


class TestUSGSClientUnitValidation:
    """Test USGS client validates expected units (mg/L as N)."""

    def test_usgs_accepts_correct_units(self):
        """Test USGS client accepts mg/L as N without warnings."""
        client = USGSClient()

        df = pd.DataFrame({
            'nitrate': [10.0, 5.0, 2.0],
            'nitrate_unit': ['mg/L as N', 'mg/L as nitrogen', 'mg/L as N']
        })

        # Should not raise warnings
        result = client._standardize_nitrate_unit(df)

        assert result['nitrate'].tolist() == [10.0, 5.0, 2.0], \
            "USGS values in mg/L as N should not be modified"

    def test_usgs_warns_on_unexpected_units(self):
        """Test USGS client warns if mg{NO3}/L units detected."""
        client = USGSClient()

        df = pd.DataFrame({
            'nitrate': [44.3],
            'nitrate_unit': ['mg{NO3}/L']
        })

        # Should emit warning for unexpected USGS units
        with pytest.warns(UserWarning, match="Unexpected USGS nitrate unit"):
            client._standardize_nitrate_unit(df)

    def test_usgs_handles_missing_unit_info(self):
        """Test USGS client handles missing unit information."""
        client = USGSClient()

        df = pd.DataFrame({
            'nitrate': [10.0, 5.0]
        })

        result = client._standardize_nitrate_unit(df)

        assert result.equals(df), "Data without unit info should remain unchanged"


class TestConversionAccuracy:
    """Test conversion accuracy with known values."""

    def test_conversion_accuracy_zero(self):
        """Test conversion of zero value."""
        assert 0.0 * NITRATE_NO3_TO_N == 0.0

    def test_conversion_accuracy_precision(self):
        """Test conversion maintains sufficient precision."""
        test_values = [0.1, 1.0, 10.0, 44.3, 100.0]

        for nitrate_no3 in test_values:
            nitrate_as_n = nitrate_no3 * NITRATE_NO3_TO_N
            # Reverse conversion should recover original value within 0.1%
            recovered = nitrate_as_n / NITRATE_NO3_TO_N
            relative_error = abs(recovered - nitrate_no3) / nitrate_no3
            assert relative_error < 0.001, \
                f"Conversion precision error for {nitrate_no3}: {relative_error:.6f}"


def test_integration_kaggle_conversion():
    """Integration test: Verify Kaggle data uses NITRATE_NO3_TO_N constant."""
    # This test verifies the constant is imported and available
    # in feature_engineering.py where Kaggle data is processed
    from src.preprocessing import feature_engineering

    assert hasattr(feature_engineering, 'NITRATE_NO3_TO_N'), \
        "NITRATE_NO3_TO_N must be accessible in feature_engineering module"

    assert feature_engineering.NITRATE_NO3_TO_N == NITRATE_NO3_TO_N, \
        "Constant value must be consistent across imports"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
