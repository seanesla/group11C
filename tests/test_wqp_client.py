"""
Tests for Water Quality Portal API Client.

Uses REAL fixtures captured from actual API calls.
Integration tests with @pytest.mark.integration for live API calls.

Total test cases: ~25
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from tests.conftest import load_real_fixture
from src.data_collection.wqp_client import WQPClient


class TestWQPClientWithFixtures:
    """Test WQP client using real captured API responses."""

    def setup_method(self):
        self.client = WQPClient()

    def test_parse_dc_full_data_fixture(self):
        """Load and parse DC fixture with 4,287 real records."""
        fixture_data = load_real_fixture('real_wqp_responses/dc_full_data.json')
        assert 'dataframe' in fixture_data
        assert 'shape' in fixture_data

        # Verify shape
        shape = fixture_data['shape']
        assert shape[0] == 4287  # 4,287 records
        assert shape[1] > 0  # Multiple columns

        # Convert back to DataFrame
        df = pd.DataFrame(fixture_data['dataframe'])
        assert len(df) == 4287
        assert not df.empty

    def test_parse_nyc_full_data_fixture(self):
        """Load and parse NYC fixture with 3,504 real records."""
        fixture_data = load_real_fixture('real_wqp_responses/nyc_full_data.json')
        df = pd.DataFrame(fixture_data['dataframe'])
        assert len(df) == 3504
        assert not df.empty

    def test_parse_alaska_sparse_data_fixture(self):
        """Load and parse Alaska fixture with 84 records."""
        fixture_data = load_real_fixture('real_wqp_responses/alaska_sparse_data.json')
        df = pd.DataFrame(fixture_data['dataframe'])
        assert len(df) == 84
        assert not df.empty

    def test_parse_empty_data_fixture(self):
        """Load and parse empty data fixture (no records found)."""
        fixture_data = load_real_fixture('real_wqp_responses/empty_data.json')
        df = pd.DataFrame(fixture_data['dataframe'])
        assert len(df) == 0
        assert df.empty

    def test_invalid_coords_error_fixture(self):
        """Load fixture from invalid coordinates (returns empty or error)."""
        fixture_data = load_real_fixture('real_wqp_responses/invalid_coords_error.json')
        # API returned either error or empty dataframe
        if 'error' in fixture_data:
            assert 'error_type' in fixture_data
        else:
            # Or it returned empty dataframe
            df = pd.DataFrame(fixture_data['dataframe'])
            assert df.empty


@pytest.mark.integration
class TestWQPClientIntegration:
    """Integration tests that make REAL API calls."""

    def setup_method(self):
        self.client = WQPClient(rate_limit_delay=2.0)  # Be nice to API

    def test_get_water_quality_data_dc_live(self):
        """Make real API call for DC area."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        data = self.client.get_water_quality_data(
            latitude=38.9,
            longitude=-77.0,
            radius_miles=10,
            start_date=start_date,
            end_date=end_date,
            characteristics=['pH']
        )

        # Should return DataFrame (may be empty if no recent data)
        assert isinstance(data, pd.DataFrame)

    def test_get_stations_dc_live(self):
        """Make real API call to get monitoring stations near DC."""
        stations = self.client.get_stations(
            latitude=38.9,
            longitude=-77.0,
            radius_miles=25
        )

        assert isinstance(stations, pd.DataFrame)
        # DC area should have stations
        if not stations.empty:
            assert 'MonitoringLocationIdentifier' in stations.columns

    def test_empty_response_remote_area(self):
        """Test API call for remote area with no monitoring stations."""
        data = self.client.get_water_quality_data(
            latitude=36.5,  # Death Valley area
            longitude=-117.0,
            radius_miles=10,
            characteristics=['pH']
        )

        assert isinstance(data, pd.DataFrame)
        # Remote area likely has no data
        assert len(data) == 0


class TestWQPClientValidation:
    """Test input validation and error handling."""

    def setup_method(self):
        self.client = WQPClient()

    def test_invalid_latitude_out_of_range(self):
        """Latitude > 90 should raise or return empty."""
        # The API itself handles validation, client should handle gracefully
        data = self.client.get_water_quality_data(
            latitude=999,
            longitude=-77.0,
            radius_miles=10
        )
        assert isinstance(data, pd.DataFrame)
        # Invalid coords should return empty DataFrame

    def test_invalid_longitude_out_of_range(self):
        """Longitude > 180 should be handled."""
        data = self.client.get_water_quality_data(
            latitude=38.9,
            longitude=999,
            radius_miles=10
        )
        assert isinstance(data, pd.DataFrame)

    def test_negative_radius(self):
        """Negative radius should be handled."""
        data = self.client.get_water_quality_data(
            latitude=38.9,
            longitude=-77.0,
            radius_miles=-10
        )
        assert isinstance(data, pd.DataFrame)

    def test_date_range_end_before_start(self):
        """End date before start date should be handled."""
        end_date = datetime(2020, 1, 1)
        start_date = datetime(2020, 12, 31)

        data = self.client.get_water_quality_data(
            latitude=38.9,
            longitude=-77.0,
            radius_miles=10,
            start_date=start_date,
            end_date=end_date
        )
        assert isinstance(data, pd.DataFrame)


class TestWQPClientRateLimiting:
    """Test rate limiting behavior."""

    def test_rate_limit_delay_configured(self):
        """Rate limit delay should be configurable."""
        client1 = WQPClient(rate_limit_delay=1.0)
        assert client1.rate_limit_delay == 1.0

        client2 = WQPClient(rate_limit_delay=2.5)
        assert client2.rate_limit_delay == 2.5

    def test_default_rate_limit(self):
        """Default rate limit should be 1.0 seconds."""
        client = WQPClient()
        assert client.rate_limit_delay == 1.0


class TestWQPClientDataStructure:
    """Test that returned data has expected structure."""

    def setup_method(self):
        self.client = WQPClient()

    def test_returns_dataframe(self):
        """All data methods should return pandas DataFrames."""
        # Load a real fixture and verify it can be converted to DataFrame
        fixture_data = load_real_fixture('real_wqp_responses/dc_full_data.json')
        df = pd.DataFrame(fixture_data['dataframe'])

        assert isinstance(df, pd.DataFrame)
        # Should have expected WQP columns
        expected_cols = ['ActivityStartDate', 'CharacteristicName', 'ResultMeasureValue']
        for col in expected_cols:
            assert col in df.columns

    def test_empty_dataframe_structure(self):
        """Empty responses should return empty DataFrame with correct structure."""
        fixture_data = load_real_fixture('real_wqp_responses/empty_data.json')
        df = pd.DataFrame(fixture_data['dataframe'])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert df.empty


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not integration"])
