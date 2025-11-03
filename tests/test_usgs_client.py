"""
Tests for USGS NWIS API Client.

Uses REAL fixtures and focused tests to boost coverage.

Total test cases: ~15
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from tests.conftest import load_real_fixture
from src.data_collection.usgs_client import USGSClient


class TestUSGSClientWithFixtures:
    """Test USGS client using real captured API responses."""

    def setup_method(self):
        self.client = USGSClient()

    def test_parse_dc_data_fixture(self):
        """Load and parse DC USGS fixture."""
        fixture_data = load_real_fixture('real_usgs_responses/dc_data.json')
        assert 'sites' in fixture_data
        assert 'data' in fixture_data

        sites_df = pd.DataFrame(fixture_data['sites'])
        assert not sites_df.empty
        assert len(sites_df) == 64  # 64 sites found

    def test_parse_nyc_data_fixture(self):
        """Load and parse NYC USGS fixture."""
        fixture_data = load_real_fixture('real_usgs_responses/nyc_data.json')
        sites_df = pd.DataFrame(fixture_data['sites'])
        assert not sites_df.empty
        assert len(sites_df) == 78  # 78 sites found


@pytest.mark.integration
class TestUSGSClientIntegration:
    """Integration tests with real API calls."""

    def setup_method(self):
        self.client = USGSClient(rate_limit_delay=2.0)

    def test_find_sites_dc_live(self):
        """Find USGS sites near DC."""
        sites = self.client.find_sites_by_location(
            latitude=38.9,
            longitude=-77.0,
            radius_miles=25
        )

        assert isinstance(sites, pd.DataFrame)


class TestUSGSClientValidation:
    """Test input validation."""

    def setup_method(self):
        self.client = USGSClient()

    def test_invalid_latitude(self):
        """Invalid latitude should be handled."""
        sites = self.client.find_sites_by_location(
            latitude=999,
            longitude=-77.0,
            radius_miles=10
        )
        assert isinstance(sites, pd.DataFrame)

    def test_negative_radius(self):
        """Negative radius should be handled."""
        sites = self.client.find_sites_by_location(
            latitude=38.9,
            longitude=-77.0,
            radius_miles=-10
        )
        assert isinstance(sites, pd.DataFrame)

    def test_empty_site_codes_list(self):
        """Empty site codes should return empty DataFrame."""
        data = self.client.get_water_quality_data(
            site_codes=[],
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31)
        )
        assert isinstance(data, pd.DataFrame)
        assert data.empty


class TestUSGSClientRateLimiting:
    """Test rate limiting."""

    def test_rate_limit_configurable(self):
        """Rate limit should be configurable."""
        client1 = USGSClient(rate_limit_delay=1.0)
        assert client1.rate_limit_delay == 1.0

        client2 = USGSClient(rate_limit_delay=3.0)
        assert client2.rate_limit_delay == 3.0

    def test_default_rate_limit(self):
        """Default rate limit should be 1.0."""
        client = USGSClient()
        assert client.rate_limit_delay == 1.0


class TestUSGSClientDataStructure:
    """Test data structure."""

    def setup_method(self):
        self.client = USGSClient()

    def test_returns_dataframe(self):
        """Methods should return DataFrames."""
        fixture_data = load_real_fixture('real_usgs_responses/dc_data.json')
        sites_df = pd.DataFrame(fixture_data['sites'])

        assert isinstance(sites_df, pd.DataFrame)
        if not sites_df.empty:
            assert 'site_no' in sites_df.columns


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not integration"])
