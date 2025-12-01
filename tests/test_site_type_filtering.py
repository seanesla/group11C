"""
Tests for site type filtering and marine contamination detection.

Tests the constants, validation layer, and API client filtering
that prevents marine/estuarine stations from contaminating
freshwater drinking water quality assessments.
"""

import pytest
import pandas as pd
from src.data_collection.constants import (
    SURFACE_WATER_SITE_TYPES_USGS,
    SURFACE_WATER_SITE_TYPES_WQP,
    MARINE_SITE_TYPES_USGS,
    MARINE_SITE_TYPES_WQP,
    VALID_RANGES,
    MARINE_CONDUCTANCE_THRESHOLD,
)
from src.data_collection.fallback import validate_water_quality_data


class TestSiteTypeConstants:
    """Test site type constant definitions."""

    def test_surface_water_types_usgs_defined(self):
        """USGS surface water types should include streams, lakes, springs, wetlands."""
        assert 'ST' in SURFACE_WATER_SITE_TYPES_USGS  # Stream
        assert 'LK' in SURFACE_WATER_SITE_TYPES_USGS  # Lake
        assert 'SP' in SURFACE_WATER_SITE_TYPES_USGS  # Spring
        assert 'WE' in SURFACE_WATER_SITE_TYPES_USGS  # Wetland

    def test_surface_water_types_usgs_excludes_groundwater(self):
        """USGS surface water types should NOT include groundwater."""
        assert 'GW' not in SURFACE_WATER_SITE_TYPES_USGS

    def test_surface_water_types_usgs_excludes_marine(self):
        """USGS surface water types should NOT include marine types."""
        for marine_type in MARINE_SITE_TYPES_USGS:
            assert marine_type not in SURFACE_WATER_SITE_TYPES_USGS

    def test_surface_water_types_wqp_defined(self):
        """WQP surface water types should include streams, lakes, springs, wetlands."""
        assert 'Stream' in SURFACE_WATER_SITE_TYPES_WQP
        assert any('Lake' in t for t in SURFACE_WATER_SITE_TYPES_WQP)
        assert 'Spring' in SURFACE_WATER_SITE_TYPES_WQP
        assert 'Wetland' in SURFACE_WATER_SITE_TYPES_WQP

    def test_surface_water_types_wqp_excludes_marine(self):
        """WQP surface water types should NOT include marine types."""
        for surface_type in SURFACE_WATER_SITE_TYPES_WQP:
            assert 'Ocean' not in surface_type
            assert 'Estuary' not in surface_type

    def test_marine_types_usgs_defined(self):
        """USGS marine types should include estuary, ocean, coastal, tidal."""
        assert 'ES' in MARINE_SITE_TYPES_USGS  # Estuary
        assert 'OC' in MARINE_SITE_TYPES_USGS  # Ocean

    def test_marine_types_wqp_defined(self):
        """WQP marine types should include estuary and ocean."""
        assert 'Estuary' in MARINE_SITE_TYPES_WQP
        assert 'Ocean' in MARINE_SITE_TYPES_WQP

    def test_marine_conductance_threshold(self):
        """Marine conductance threshold should be 5000 uS/cm."""
        assert MARINE_CONDUCTANCE_THRESHOLD == 5000.0


class TestValidRanges:
    """Test validation range definitions."""

    def test_valid_ranges_includes_all_parameters(self):
        """Should have ranges for all WQI parameters."""
        expected_params = ['ph', 'dissolved_oxygen', 'temperature',
                          'turbidity', 'nitrate', 'conductance']
        for param in expected_params:
            assert param in VALID_RANGES

    def test_ph_range(self):
        """pH range should be 0-14."""
        assert VALID_RANGES['ph'] == (0.0, 14.0)

    def test_conductance_range(self):
        """Conductance range should cap at marine threshold."""
        min_val, max_val = VALID_RANGES['conductance']
        assert min_val == 0.0
        assert max_val == MARINE_CONDUCTANCE_THRESHOLD


class TestValidateWaterQualityData:
    """Test the validation layer for marine contamination detection."""

    def test_empty_dataframe_returns_empty(self):
        """Empty DataFrame should return empty with no warnings."""
        df = pd.DataFrame()
        result, warnings = validate_water_quality_data(df)
        assert result.empty
        assert len(warnings) == 0

    def test_none_returns_none(self):
        """None input should return None with no warnings."""
        result, warnings = validate_water_quality_data(None)
        assert result is None
        assert len(warnings) == 0

    def test_aggregated_format_filters_high_conductance(self):
        """Should filter rows with conductance > threshold in aggregated format."""
        df = pd.DataFrame({
            'conductance': [1000, 2000, 6000, 3000, 10000],
            'ph': [7.0, 7.5, 8.0, 7.2, 7.8]
        })
        result, warnings = validate_water_quality_data(df)

        # Should have filtered out 6000 and 10000
        assert len(result) == 3
        assert result['conductance'].max() <= MARINE_CONDUCTANCE_THRESHOLD
        assert len(warnings) == 1
        assert 'Excluded 2 record(s)' in warnings[0]

    def test_aggregated_format_no_warning_when_valid(self):
        """Should not warn when all conductance values are valid."""
        df = pd.DataFrame({
            'conductance': [1000, 2000, 3000, 4000],
            'ph': [7.0, 7.5, 8.0, 7.2]
        })
        result, warnings = validate_water_quality_data(df)

        assert len(result) == 4
        assert len(warnings) == 0

    def test_wqp_format_filters_high_conductance_stations(self):
        """Should filter stations with high conductance in WQP long format."""
        df = pd.DataFrame({
            'MonitoringLocationIdentifier': ['A', 'A', 'B', 'B', 'C', 'C'],
            'CharacteristicName': [
                'pH', 'Specific conductance',
                'pH', 'Specific conductance',
                'pH', 'Specific conductance'
            ],
            'ResultMeasureValue': [7.0, 1000, 7.5, 8000, 7.2, 2000]
        })
        result, warnings = validate_water_quality_data(df)

        # Station B has conductance 8000 > threshold, should be removed entirely
        assert 'B' not in result['MonitoringLocationIdentifier'].values
        # Stations A and C should remain
        assert 'A' in result['MonitoringLocationIdentifier'].values
        assert 'C' in result['MonitoringLocationIdentifier'].values
        assert len(warnings) == 1

    def test_wqp_format_no_warning_when_valid(self):
        """Should not warn when all conductance values are valid in WQP format."""
        df = pd.DataFrame({
            'MonitoringLocationIdentifier': ['A', 'A', 'B', 'B'],
            'CharacteristicName': [
                'pH', 'Specific conductance',
                'pH', 'Specific conductance'
            ],
            'ResultMeasureValue': [7.0, 1000, 7.5, 2000]
        })
        result, warnings = validate_water_quality_data(df)

        assert len(result) == 4
        assert len(warnings) == 0

    def test_no_conductance_column_returns_unchanged(self):
        """DataFrame without conductance should pass through unchanged."""
        df = pd.DataFrame({
            'ph': [7.0, 7.5, 8.0],
            'temperature': [20.0, 22.0, 25.0]
        })
        result, warnings = validate_water_quality_data(df)

        assert len(result) == 3
        assert len(warnings) == 0


class TestClientSiteTypeParameter:
    """Test that API clients include site type filtering."""

    def test_wqp_client_has_site_types_parameter(self):
        """WQP client methods should accept site_types parameter."""
        from src.data_collection.wqp_client import WQPClient
        import inspect

        client = WQPClient()

        # Check get_water_quality_data
        sig = inspect.signature(client.get_water_quality_data)
        assert 'site_types' in sig.parameters

        # Check get_stations
        sig = inspect.signature(client.get_stations)
        assert 'site_types' in sig.parameters

        # Check get_data_by_location
        sig = inspect.signature(client.get_data_by_location)
        assert 'site_types' in sig.parameters

    def test_usgs_client_has_site_types_parameter(self):
        """USGS client methods should accept site_types parameter."""
        from src.data_collection.usgs_client import USGSClient
        import inspect

        client = USGSClient()

        # Check find_sites_by_location
        sig = inspect.signature(client.find_sites_by_location)
        assert 'site_types' in sig.parameters

        # Check get_data_by_location
        sig = inspect.signature(client.get_data_by_location)
        assert 'site_types' in sig.parameters
