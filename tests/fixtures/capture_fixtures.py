"""
Script to capture REAL API responses for testing.

This script makes ACTUAL API calls and saves responses as fixtures.
Run this script to regenerate fixtures when needed.

NO MOCKS - ALL REAL DATA.
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_collection.wqp_client import WQPClient
from src.data_collection.usgs_client import USGSClient
from src.geolocation.zipcode_mapper import ZipCodeMapper


def save_fixture(data, filename, description):
    """Save data as fixture with metadata."""
    fixture_data = {
        'captured_at': datetime.now().isoformat(),
        'description': description,
        'data': data
    }

    filepath = Path(__file__).parent / filename
    with open(filepath, 'w') as f:
        json.dump(fixture_data, f, indent=2, default=str)

    print(f"✓ Saved: {filename}")
    print(f"  {description}")
    print()


def capture_wqp_fixtures():
    """Capture REAL Water Quality Portal API responses."""
    print("=" * 70)
    print("CAPTURING REAL WQP API RESPONSES")
    print("=" * 70)
    print()

    wqp = WQPClient()
    mapper = ZipCodeMapper()

    # Date range: last 365 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    # Test Case 1: Washington DC - Expected to have rich data
    print("[1/5] Washington DC (20001) - Full data...")
    try:
        dc_location = mapper.get_location_info('20001')
        dc_data = wqp.get_water_quality_data(
            latitude=dc_location['latitude'],
            longitude=dc_location['longitude'],
            radius_miles=25,
            start_date=start_date,
            end_date=end_date,
            characteristics=[
                'pH', 'Dissolved oxygen (DO)', 'Temperature, water',
                'Turbidity', 'Nitrate', 'Specific conductance'
            ]
        )

        save_fixture(
            data={'dataframe': dc_data.to_dict('records'), 'shape': list(dc_data.shape)},
            filename='real_wqp_responses/dc_full_data.json',
            description=f'Washington DC water quality data - {len(dc_data)} records'
        )
    except Exception as e:
        print(f"✗ Error capturing DC data: {e}")
        save_fixture(
            data={'error': str(e), 'error_type': type(e).__name__},
            filename='real_wqp_responses/dc_error.json',
            description=f'DC data capture error: {type(e).__name__}'
        )
        print()

    # Test Case 2: New York City - Expected to have rich data
    print("[2/5] New York City (10001) - Full data...")
    try:
        nyc_location = mapper.get_location_info('10001')
        nyc_data = wqp.get_water_quality_data(
            latitude=nyc_location['latitude'],
            longitude=nyc_location['longitude'],
            radius_miles=25,
            start_date=start_date,
            end_date=end_date,
            characteristics=[
                'pH', 'Dissolved oxygen (DO)', 'Temperature, water',
                'Turbidity', 'Nitrate', 'Specific conductance'
            ]
        )

        save_fixture(
            data={'dataframe': nyc_data.to_dict('records'), 'shape': list(nyc_data.shape)},
            filename='real_wqp_responses/nyc_full_data.json',
            description=f'New York City water quality data - {len(nyc_data)} records'
        )
    except Exception as e:
        print(f"✗ Error capturing NYC data: {e}")
        save_fixture(
            data={'error': str(e), 'error_type': type(e).__name__},
            filename='real_wqp_responses/nyc_error.json',
            description=f'NYC data capture error: {type(e).__name__}'
        )
        print()

    # Test Case 3: Anchorage, AK - Expected to have sparse data
    print("[3/5] Anchorage, AK (99501) - Sparse data...")
    try:
        ak_location = mapper.get_location_info('99501')
        ak_data = wqp.get_water_quality_data(
            latitude=ak_location['latitude'],
            longitude=ak_location['longitude'],
            radius_miles=50,
            start_date=start_date,
            end_date=end_date,
            characteristics=[
                'pH', 'Dissolved oxygen (DO)', 'Temperature, water',
                'Turbidity', 'Nitrate', 'Specific conductance'
            ]
        )

        save_fixture(
            data={'dataframe': ak_data.to_dict('records'), 'shape': list(ak_data.shape)},
            filename='real_wqp_responses/alaska_sparse_data.json',
            description=f'Anchorage, AK water quality data - {len(ak_data)} records'
        )
    except Exception as e:
        print(f"✗ Error capturing Alaska data: {e}")
        save_fixture(
            data={'error': str(e), 'error_type': type(e).__name__},
            filename='real_wqp_responses/alaska_error.json',
            description=f'Alaska data capture error: {type(e).__name__}'
        )
        print()

    # Test Case 4: Remote area - Expected to have NO data
    print("[4/5] Remote area (Death Valley-ish) - Empty data...")
    try:
        # Death Valley area - likely no monitoring stations
        empty_data = wqp.get_water_quality_data(
            latitude=36.5,
            longitude=-117.0,
            radius_miles=10,
            start_date=start_date,
            end_date=end_date,
            characteristics=['pH']
        )

        save_fixture(
            data={'dataframe': empty_data.to_dict('records'), 'shape': list(empty_data.shape)},
            filename='real_wqp_responses/empty_data.json',
            description=f'Remote area - {len(empty_data)} records (expected 0)'
        )
    except Exception as e:
        print(f"✗ Error capturing empty data: {e}")
        save_fixture(
            data={'error': str(e), 'error_type': type(e).__name__},
            filename='real_wqp_responses/empty_error.json',
            description=f'Empty data capture error: {type(e).__name__}'
        )
        print()

    # Test Case 5: Invalid coordinates - API error
    print("[5/5] Invalid coordinates - API error response...")
    try:
        error_data = wqp.get_water_quality_data(
            latitude=999,  # Invalid
            longitude=999,  # Invalid
            radius_miles=10,
            start_date=start_date,
            end_date=end_date,
            characteristics=['pH']
        )

        save_fixture(
            data={'dataframe': error_data.to_dict('records') if not error_data.empty else [], 'shape': list(error_data.shape)},
            filename='real_wqp_responses/invalid_coords_error.json',
            description='Invalid coordinates - error or empty response'
        )
    except Exception as e:
        # This is expected - save the error
        save_fixture(
            data={'error': str(e), 'error_type': type(e).__name__},
            filename='real_wqp_responses/invalid_coords_error.json',
            description=f'Invalid coordinates - captured error: {type(e).__name__}'
        )


def capture_usgs_fixtures():
    """Capture REAL USGS NWIS API responses."""
    print("=" * 70)
    print("CAPTURING REAL USGS API RESPONSES")
    print("=" * 70)
    print()

    usgs = USGSClient()
    mapper = ZipCodeMapper()

    # Date range: last 30 days (USGS is for recent data)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    # Test Case 1: Washington DC area
    print("[1/3] Washington DC area - USGS data...")
    try:
        dc_location = mapper.get_location_info('20001')
        sites = usgs.find_sites_by_location(
            latitude=dc_location['latitude'],
            longitude=dc_location['longitude'],
            radius_miles=25
        )

        if not sites.empty:
            # Get data for first site
            site_code = sites.iloc[0]['site_no']
            dc_data = usgs.get_water_quality_data(
                site_codes=[site_code],
                start_date=start_date,
                end_date=end_date,
                parameters=['00010', '00400', '00300']  # Temp, pH, DO
            )

            save_fixture(
                data={'sites': sites.to_dict('records'), 'data': dc_data.to_dict('records')},
                filename='real_usgs_responses/dc_data.json',
                description=f'DC USGS data - {len(sites)} sites, {len(dc_data)} records'
            )
        else:
            save_fixture(
                data={'sites': [], 'data': []},
                filename='real_usgs_responses/dc_data.json',
                description='DC USGS data - no sites found'
            )
    except Exception as e:
        print(f"✗ Error capturing USGS DC data: {e}")
        save_fixture(
            data={'error': str(e), 'error_type': type(e).__name__},
            filename='real_usgs_responses/dc_error.json',
            description=f'DC USGS error: {type(e).__name__}'
        )
        print()

    # Test Case 2: New York area
    print("[2/3] New York area - USGS data...")
    try:
        nyc_location = mapper.get_location_info('10001')
        sites = usgs.find_sites_by_location(
            latitude=nyc_location['latitude'],
            longitude=nyc_location['longitude'],
            radius_miles=25
        )

        if not sites.empty:
            site_code = sites.iloc[0]['site_no']
            nyc_data = usgs.get_water_quality_data(
                site_codes=[site_code],
                start_date=start_date,
                end_date=end_date,
                parameters=['00010', '00400', '00300']
            )

            save_fixture(
                data={'sites': sites.to_dict('records'), 'data': nyc_data.to_dict('records')},
                filename='real_usgs_responses/nyc_data.json',
                description=f'NYC USGS data - {len(sites)} sites, {len(nyc_data)} records'
            )
        else:
            save_fixture(
                data={'sites': [], 'data': []},
                filename='real_usgs_responses/nyc_data.json',
                description='NYC USGS data - no sites found'
            )
    except Exception as e:
        print(f"✗ Error capturing USGS NYC data: {e}")
        save_fixture(
            data={'error': str(e), 'error_type': type(e).__name__},
            filename='real_usgs_responses/nyc_error.json',
            description=f'NYC USGS error: {type(e).__name__}'
        )
        print()

    # Test Case 3: Error case - invalid coordinates
    print("[3/3] Invalid coordinates - USGS error...")
    try:
        error_sites = usgs.find_sites_by_location(
            latitude=999,
            longitude=999,
            radius_miles=10
        )

        save_fixture(
            data={'sites': error_sites.to_dict('records')},
            filename='real_usgs_responses/invalid_coords_error.json',
            description='Invalid coords - should be error or empty'
        )
    except Exception as e:
        save_fixture(
            data={'error': str(e), 'error_type': type(e).__name__},
            filename='real_usgs_responses/invalid_coords_error.json',
            description=f'Invalid coords - captured error: {type(e).__name__}'
        )


def main():
    """Capture all fixtures."""
    print()
    print("=" * 70)
    print("REAL API FIXTURE CAPTURE SCRIPT")
    print("NO MOCKS - CAPTURING ACTUAL API RESPONSES")
    print("=" * 70)
    print()

    # Capture WQP fixtures
    capture_wqp_fixtures()

    print()

    # Capture USGS fixtures
    capture_usgs_fixtures()

    print()
    print("=" * 70)
    print("FIXTURE CAPTURE COMPLETE")
    print("=" * 70)
    print()
    print("Fixtures saved in tests/fixtures/")
    print("Use these in tests for fast, repeatable testing with REAL data.")
    print()


if __name__ == '__main__':
    main()
