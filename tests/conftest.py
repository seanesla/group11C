"""
Pytest configuration and fixtures for water quality testing.

IMPORTANT: Following CLAUDE.md rules - NO MOCKS OR FAKE DATA.
All fixtures contain REAL data samples or will make REAL API calls.
"""

import os

# Allow model save/load to use temp directories during testing
os.environ["WQI_SKIP_PATH_VALIDATION"] = "1"

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path


# =============================================================================
# Fixture Loading Helpers
# =============================================================================

def load_real_fixture_helper(filename):
    """
    Load a REAL API response fixture captured from actual API calls.

    Args:
        filename: Relative path to fixture file (e.g., 'real_wqp_responses/dc_full_data.json')

    Returns:
        dict: The 'data' portion of the fixture
    """
    fixture_path = Path(__file__).parent / 'fixtures' / filename
    with open(fixture_path, 'r') as f:
        fixture_data = json.load(f)
    return fixture_data['data']


@pytest.fixture
def load_real_fixture():
    """Fixture that returns the helper function for loading real fixtures."""
    return load_real_fixture_helper


# =============================================================================
# REAL Water Quality Parameter Fixtures
# =============================================================================

@pytest.fixture
def excellent_water_params():
    """Real parameters representing excellent water quality (WQI ~95)."""
    return {
        'ph': 7.2,
        'dissolved_oxygen': 9.5,
        'temperature': 20.0,
        'turbidity': 2.0,
        'nitrate': 1.0,
        'specific_conductance': 200.0
    }


@pytest.fixture
def good_water_params():
    """Real parameters representing good water quality (WQI ~80)."""
    return {
        'ph': 7.8,
        'dissolved_oxygen': 7.5,
        'temperature': 22.0,
        'turbidity': 8.0,
        'nitrate': 5.0,
        'specific_conductance': 400.0
    }


@pytest.fixture
def fair_water_params():
    """Real parameters representing fair water quality (WQI ~60)."""
    return {
        'ph': 8.5,
        'dissolved_oxygen': 5.5,
        'temperature': 28.0,
        'turbidity': 25.0,
        'nitrate': 15.0,
        'specific_conductance': 800.0
    }


@pytest.fixture
def poor_water_params():
    """Real parameters representing poor water quality (WQI ~40)."""
    return {
        'ph': 9.2,
        'dissolved_oxygen': 3.5,
        'temperature': 32.0,
        'turbidity': 60.0,
        'nitrate': 30.0,
        'specific_conductance': 1500.0
    }


@pytest.fixture
def very_poor_water_params():
    """Real parameters representing very poor water quality (WQI ~15)."""
    return {
        'ph': 10.0,
        'dissolved_oxygen': 1.5,
        'temperature': 35.0,
        'turbidity': 150.0,
        'nitrate': 50.0,
        'specific_conductance': 3000.0
    }


# =============================================================================
# REAL ZIP Code Fixtures
# =============================================================================

@pytest.fixture
def valid_zip_codes():
    """Real US ZIP codes for testing."""
    return {
        'dc': '20001',  # Washington DC
        'nyc': '10001',  # New York City
        'la': '90001',  # Los Angeles
        'chicago': '60601',  # Chicago
        'houston': '77001',  # Houston
        'alaska': '99501',  # Anchorage, AK
        'maine': '04101',  # Portland, ME
    }


@pytest.fixture
def invalid_zip_codes():
    """Invalid ZIP code formats for validation testing."""
    return {
        'too_short': '123',
        'too_long': '123456',
        'letters': 'ABCDE',
        'mixed': '12A45',
        'spaces': '12 345',
        'empty': '',
        'leading_space': ' 20001',
        'trailing_space': '20001 ',
    }


# =============================================================================
# REAL DataFrame Fixtures (Based on Actual WQP Data Format)
# =============================================================================

@pytest.fixture
def sample_wqp_dataframe():
    """
    Sample DataFrame mimicking REAL WQP API response structure.
    Based on actual Water Quality Portal data format.
    """
    return pd.DataFrame({
        'ActivityStartDate': [
            '2024-01-15', '2024-01-15', '2024-02-20', '2024-02-20',
            '2024-03-10', '2024-03-10', '2024-04-05', '2024-04-05'
        ],
        'CharacteristicName': [
            'pH', 'Dissolved oxygen (DO)', 'pH', 'Temperature, water',
            'Turbidity', 'Nitrate', 'pH', 'Specific conductance'
        ],
        'ResultMeasureValue': [
            '7.2', '9.5', '7.5', '20.0',
            '5.0', '2.0', '7.8', '300.0'
        ],
        'ResultMeasure/MeasureUnitCode': [
            'None', 'mg/l', 'None', 'deg C',
            'NTU', 'mg/l', 'None', 'uS/cm'
        ],
        'MonitoringLocationIdentifier': [
            'USGS-01646500', 'USGS-01646500', 'USGS-01646500', 'USGS-01646500',
            'USGS-01646500', 'USGS-01646500', 'USGS-01646500', 'USGS-01646500'
        ],
        'MonitoringLocationName': [
            'Potomac River', 'Potomac River', 'Potomac River', 'Potomac River',
            'Potomac River', 'Potomac River', 'Potomac River', 'Potomac River'
        ],
        'LatitudeMeasure': [38.85, 38.85, 38.85, 38.85, 38.85, 38.85, 38.85, 38.85],
        'LongitudeMeasure': [-77.04, -77.04, -77.04, -77.04, -77.04, -77.04, -77.04, -77.04],
    })


@pytest.fixture
def empty_wqp_dataframe():
    """Empty DataFrame with correct WQP column structure."""
    return pd.DataFrame(columns=[
        'ActivityStartDate', 'CharacteristicName', 'ResultMeasureValue',
        'ResultMeasure/MeasureUnitCode', 'MonitoringLocationIdentifier',
        'MonitoringLocationName', 'LatitudeMeasure', 'LongitudeMeasure'
    ])


@pytest.fixture
def sample_usgs_dataframe():
    """
    Sample DataFrame mimicking REAL USGS NWIS data format.
    Based on actual USGS response structure.
    """
    return pd.DataFrame({
        'datetime': pd.to_datetime([
            '2024-01-15 10:00', '2024-01-15 11:00',
            '2024-02-20 10:00', '2024-02-20 11:00'
        ]),
        '00010': [20.0, 20.5, 21.0, 21.2],  # Temperature
        '00400': [7.2, 7.3, 7.1, 7.2],  # pH
        '00300': [9.5, 9.3, 9.4, 9.6],  # Dissolved Oxygen
    })


# =============================================================================
# REAL Location Fixtures
# =============================================================================

@pytest.fixture
def real_locations():
    """Real geographic locations with known coordinates."""
    return {
        'washington_dc': {
            'zip': '20001',
            'lat': 38.9072,
            'lon': -77.0369,
            'city': 'Washington',
            'state': 'DC'
        },
        'new_york': {
            'zip': '10001',
            'lat': 40.7506,
            'lon': -73.9971,
            'city': 'New York',
            'state': 'NY'
        },
        'anchorage': {
            'zip': '99501',
            'lat': 61.2181,
            'lon': -149.9003,
            'city': 'Anchorage',
            'state': 'AK'
        }
    }


# =============================================================================
# Date Range Fixtures
# =============================================================================

@pytest.fixture
def recent_date_range():
    """Recent date range for API queries (last 90 days)."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


@pytest.fixture
def one_year_date_range():
    """One year date range for historical analysis."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


# =============================================================================
# WQI Test Cases Fixtures
# =============================================================================

@pytest.fixture
def wqi_boundary_conditions():
    """
    Boundary condition test cases for WQI classifications.
    Tests exact boundaries between quality levels.
    """
    return [
        # (params, expected_wqi_range, expected_classification)
        ({'ph': 7.0, 'dissolved_oxygen': 10.0}, (90, 100), 'Excellent'),
        ({'ph': 7.0, 'dissolved_oxygen': 8.5}, (85, 95), 'Good'),
        ({'ph': 7.5, 'dissolved_oxygen': 6.5}, (65, 75), 'Fair'),
        ({'ph': 8.5, 'dissolved_oxygen': 4.0}, (45, 55), 'Fair'),
        ({'ph': 9.5, 'dissolved_oxygen': 2.0}, (20, 30), 'Poor'),
    ]


# =============================================================================
# API Integration Test Fixtures
# =============================================================================

@pytest.fixture
def api_timeout_seconds():
    """Reasonable timeout for REAL API calls in tests."""
    return 30


@pytest.fixture
def api_retry_attempts():
    """Number of retry attempts for flaky network connections."""
    return 3


# =============================================================================
# Edge Case Fixtures
# =============================================================================

@pytest.fixture
def nan_and_null_params():
    """Edge cases with NaN, None, and missing values."""
    return [
        {'ph': np.nan, 'dissolved_oxygen': 8.0},
        {'ph': None, 'dissolved_oxygen': 8.0},
        {'ph': 7.0, 'dissolved_oxygen': None},
        {'ph': 7.0},  # Missing DO
        {},  # All missing
    ]


@pytest.fixture
def extreme_parameter_values():
    """Extreme but technically possible parameter values."""
    return [
        {'ph': 0.0},  # Extremely acidic
        {'ph': 14.0},  # Extremely alkaline
        {'dissolved_oxygen': 0.0},  # No oxygen
        {'dissolved_oxygen': 20.0},  # Supersaturated
        {'temperature': 0.0},  # Freezing
        {'temperature': 40.0},  # Very hot
        {'turbidity': 0.0},  # Perfectly clear
        {'turbidity': 1000.0},  # Extremely turbid
        {'nitrate': 0.0},  # No nitrate
        {'nitrate': 100.0},  # Extremely high
        {'specific_conductance': 0.0},  # Pure water
        {'specific_conductance': 10000.0},  # Very high
    ]
