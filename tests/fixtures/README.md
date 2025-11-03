# Test Fixtures - REAL API Data

**NO MOCKS - ALL REAL DATA**

This directory contains REAL API responses captured from actual Water Quality Portal and USGS NWIS APIs. These fixtures enable fast, repeatable testing without making API calls for every test run.

## Captured: 2025-11-03

## Water Quality Portal (WQP) Fixtures

### `real_wqp_responses/dc_full_data.json`
- **Location:** Washington, DC (ZIP 20001)
- **Records:** 4,287 water quality measurements
- **Date Range:** Last 365 days
- **Parameters:** pH, Dissolved Oxygen, Temperature, Turbidity, Nitrate, Specific Conductance
- **Use For:** Testing rich data scenarios, WQI calculations with complete data

### `real_wqp_responses/nyc_full_data.json`
- **Location:** New York City (ZIP 10001)
- **Records:** 3,504 water quality measurements
- **Date Range:** Last 365 days
- **Parameters:** pH, Dissolved Oxygen, Temperature, Turbidity, Nitrate, Specific Conductance
- **Use For:** Testing rich data scenarios, different geographic region

### `real_wqp_responses/alaska_sparse_data.json`
- **Location:** Anchorage, AK (ZIP 99501)
- **Records:** 84 water quality measurements
- **Date Range:** Last 365 days
- **Parameters:** pH, Dissolved Oxygen, Temperature, Turbidity, Nitrate, Specific Conductance
- **Use For:** Testing sparse data scenarios, limited monitoring locations

### `real_wqp_responses/empty_data.json`
- **Location:** Remote area (Death Valley region)
- **Records:** 0 (no data available)
- **Use For:** Testing empty response handling, areas with no monitoring stations

### `real_wqp_responses/invalid_coords_error.json`
- **Coordinates:** (999, 999) - invalid
- **Result:** HTTP 400 error from API
- **Use For:** Testing error handling for invalid coordinates

## USGS NWIS Fixtures

### `real_usgs_responses/dc_data.json`
- **Location:** Washington, DC area
- **Sites Found:** 64 monitoring sites
- **Data Records:** 0 (sites exist but no recent data for requested parameters)
- **Use For:** Testing site discovery, sparse recent data scenarios

### `real_usgs_responses/nyc_data.json`
- **Location:** New York City area
- **Sites Found:** 78 monitoring sites
- **Data Records:** 0 (sites exist but no recent data for requested parameters)
- **Use For:** Testing site discovery, sparse recent data scenarios

### `real_usgs_responses/invalid_coords_error.json`
- **Coordinates:** (999, 999) - invalid
- **Result:** HTTP 400 error from API
- **Use For:** Testing error handling for invalid coordinates

## Fixture Structure

Each fixture file contains:
```json
{
  "captured_at": "ISO 8601 timestamp",
  "description": "Human-readable description",
  "data": {
    // Actual API response data
  }
}
```

## Regenerating Fixtures

To regenerate fixtures with fresh API data:

```bash
poetry run python tests/fixtures/capture_fixtures.py
```

**Warning:** This makes REAL API calls and may take several minutes.

## Usage in Tests

```python
import json
from pathlib import Path

def load_fixture(filename):
    """Load a test fixture."""
    fixture_path = Path(__file__).parent / 'fixtures' / filename
    with open(fixture_path, 'r') as f:
        fixture_data = json.load(f)
    return fixture_data['data']

# Example usage
dc_data = load_fixture('real_wqp_responses/dc_full_data.json')
```

## Important Notes

- **NO MOCKS:** All fixtures contain real API responses
- **Time-sensitive:** API data changes over time; regenerate fixtures periodically
- **Not committed to git:** These files may be large and should be in .gitignore
- **Represents reality:** Includes real errors, empty responses, and edge cases
