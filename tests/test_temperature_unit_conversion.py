"""Tests for temperature unit standardization in WQP client."""

import pandas as pd
import pytest

from src.data_collection.wqp_client import WQPClient


class TestTemperatureUnitStandardization:
    """Test _standardize_temperature_unit method."""

    def setup_method(self):
        self.client = WQPClient(rate_limit_delay=0)

    def test_fahrenheit_to_celsius_conversion(self):
        """60.2°F should convert to ~15.7°C."""
        df = pd.DataFrame({
            'CharacteristicName': ['Temperature, water'],
            'ResultMeasureValue': [60.2],
            'ResultMeasure/MeasureUnitCode': ['deg F'],
        })
        result = self.client._standardize_temperature_unit(df)
        expected = (60.2 - 32) * 5 / 9
        assert abs(result['ResultMeasureValue'].iloc[0] - expected) < 0.01
        assert result['ResultMeasure/MeasureUnitCode'].iloc[0] == 'deg C'

    def test_celsius_unchanged(self):
        """Celsius values should remain unchanged."""
        df = pd.DataFrame({
            'CharacteristicName': ['Temperature, water'],
            'ResultMeasureValue': [15.0],
            'ResultMeasure/MeasureUnitCode': ['deg C'],
        })
        result = self.client._standardize_temperature_unit(df)
        assert result['ResultMeasureValue'].iloc[0] == 15.0

    @pytest.mark.parametrize("unit", ['deg F', 'Far', 'fahrenheit', 'F', 'degF'])
    def test_fahrenheit_variants(self, unit):
        """All Fahrenheit unit variants should be converted."""
        df = pd.DataFrame({
            'CharacteristicName': ['Temperature, water'],
            'ResultMeasureValue': [68.0],
            'ResultMeasure/MeasureUnitCode': [unit],
        })
        result = self.client._standardize_temperature_unit(df)
        expected = (68.0 - 32) * 5 / 9
        assert abs(result['ResultMeasureValue'].iloc[0] - expected) < 0.01

    @pytest.mark.parametrize("unit", ['deg C', 'Cel', 'celsius', 'C', 'degC'])
    def test_celsius_variants(self, unit):
        """All Celsius unit variants should be recognized."""
        df = pd.DataFrame({
            'CharacteristicName': ['Temperature, water'],
            'ResultMeasureValue': [20.0],
            'ResultMeasure/MeasureUnitCode': [unit],
        })
        result = self.client._standardize_temperature_unit(df)
        assert result['ResultMeasureValue'].iloc[0] == 20.0

    def test_non_temperature_unchanged(self):
        """Non-temperature rows should not be modified."""
        df = pd.DataFrame({
            'CharacteristicName': ['pH', 'Temperature, water'],
            'ResultMeasureValue': [7.5, 68.0],
            'ResultMeasure/MeasureUnitCode': ['None', 'deg F'],
        })
        result = self.client._standardize_temperature_unit(df)
        assert result.loc[0, 'ResultMeasureValue'] == 7.5
        assert abs(result.loc[1, 'ResultMeasureValue'] - 20.0) < 0.1

    def test_empty_dataframe(self):
        """Empty DataFrame should return unchanged."""
        df = pd.DataFrame()
        result = self.client._standardize_temperature_unit(df)
        assert result.empty

    def test_missing_unit_column(self):
        """Missing unit column should assume Celsius."""
        df = pd.DataFrame({
            'CharacteristicName': ['Temperature, water'],
            'ResultMeasureValue': [20.0],
        })
        result = self.client._standardize_temperature_unit(df)
        assert result['ResultMeasureValue'].iloc[0] == 20.0

    def test_nan_values_unchanged(self):
        """NaN values should be left unchanged."""
        df = pd.DataFrame({
            'CharacteristicName': ['Temperature, water', 'Temperature, water'],
            'ResultMeasureValue': [float('nan'), 68.0],
            'ResultMeasure/MeasureUnitCode': ['deg F', 'deg F'],
        })
        result = self.client._standardize_temperature_unit(df)
        assert pd.isna(result['ResultMeasureValue'].iloc[0])
        expected = (68.0 - 32) * 5 / 9
        assert abs(result['ResultMeasureValue'].iloc[1] - expected) < 0.01

    def test_mixed_units(self):
        """Mixed Fahrenheit and Celsius should both be handled correctly."""
        df = pd.DataFrame({
            'CharacteristicName': ['Temperature, water', 'Temperature, water', 'Temperature, water'],
            'ResultMeasureValue': [68.0, 20.0, 50.0],
            'ResultMeasure/MeasureUnitCode': ['deg F', 'deg C', 'deg F'],
        })
        result = self.client._standardize_temperature_unit(df)
        # 68F -> 20C
        assert abs(result['ResultMeasureValue'].iloc[0] - 20.0) < 0.1
        # 20C stays 20C
        assert result['ResultMeasureValue'].iloc[1] == 20.0
        # 50F -> 10C
        assert abs(result['ResultMeasureValue'].iloc[2] - 10.0) < 0.1
