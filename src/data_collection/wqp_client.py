"""
Water Quality Portal (WQP) API Client

This module provides functionality to fetch water quality data from the Water Quality Portal,
which aggregates data from USGS, EPA, and other water quality monitoring agencies.

API Documentation: https://www.waterqualitydata.us/webservices_documentation/
"""

import requests
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any
import time
import logging
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WQPClient:
    """Client for interacting with the Water Quality Portal API."""

    BASE_URL = "https://www.waterqualitydata.us/data"

    # Common water quality characteristics
    CHARACTERISTICS = {
        'ph': 'pH',
        'dissolved_oxygen': 'Dissolved oxygen (DO)',
        'temperature': 'Temperature, water',
        'turbidity': 'Turbidity',
        'nitrate': 'Nitrate',
        'conductivity': 'Specific conductance',
        'total_nitrogen': 'Nitrogen',
        'total_phosphorus': 'Phosphorus',
    }

    def __init__(self, rate_limit_delay: float = 1.0):
        """
        Initialize the WQP client.

        Args:
            rate_limit_delay: Delay in seconds between API requests
        """
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'WaterQualityPrediction/1.0 (Educational Project)'
        })

    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> requests.Response:
        """
        Make an API request with rate limiting and error handling.

        Args:
            endpoint: The API endpoint (e.g., 'Result', 'Station')
            params: Query parameters

        Returns:
            Response object

        Raises:
            requests.RequestException: If the request fails
        """
        time.sleep(self.rate_limit_delay)

        url = f"{self.BASE_URL}/{endpoint}/search"

        try:
            response = self.session.get(url, params=params, timeout=60)
            response.raise_for_status()
            return response
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout for URL: {url}")
            raise
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e.response.text[:200]}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            raise

    def get_stations(
        self,
        bbox: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        radius_miles: Optional[float] = None,
        state_code: Optional[str] = None,
        county_code: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Find water quality monitoring stations.

        Args:
            bbox: Bounding box as "west,south,east,north"
            latitude: Center latitude (requires longitude and radius_miles)
            longitude: Center longitude (requires latitude and radius_miles)
            radius_miles: Search radius in miles
            state_code: Two-letter state code (e.g., 'VA')
            county_code: County FIPS code (e.g., 'US:51:059')

        Returns:
            DataFrame containing station information
        """
        params = {
            'mimeType': 'csv',
            'sorted': 'yes',
        }

        if bbox:
            params['bBox'] = bbox
        elif latitude and longitude and radius_miles:
            params['lat'] = latitude
            params['long'] = longitude
            params['within'] = radius_miles
        elif state_code:
            params['statecode'] = f"US:{state_code}"
        elif county_code:
            params['countycode'] = county_code
        else:
            raise ValueError("Must provide bbox, lat/long/radius, state_code, or county_code")

        try:
            response = self._make_request('Station', params)

            # Parse CSV response
            df = pd.read_csv(io.StringIO(response.text))
            logger.info(f"Found {len(df)} monitoring stations")

            return df

        except Exception as e:
            logger.error(f"Error retrieving stations: {str(e)}")
            return pd.DataFrame()

    def get_water_quality_data(
        self,
        bbox: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        radius_miles: Optional[float] = None,
        state_code: Optional[str] = None,
        county_code: Optional[str] = None,
        characteristics: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        site_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Retrieve water quality measurement data.

        Args:
            bbox: Bounding box as "west,south,east,north"
            latitude: Center latitude
            longitude: Center longitude
            radius_miles: Search radius in miles
            state_code: Two-letter state code
            county_code: County FIPS code
            characteristics: List of characteristic names (e.g., ['pH', 'Temperature, water'])
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            site_ids: List of specific monitoring location IDs

        Returns:
            DataFrame containing water quality measurements
        """
        params = {
            'mimeType': 'csv',
            'sorted': 'yes',
            'sampleMedia': 'Water',  # Focus on water samples
        }

        # Geographic filters
        if bbox:
            params['bBox'] = bbox
        elif latitude and longitude and radius_miles:
            params['lat'] = latitude
            params['long'] = longitude
            params['within'] = radius_miles
        elif state_code:
            params['statecode'] = f"US:{state_code}"
        elif county_code:
            params['countycode'] = county_code
        elif site_ids:
            params['siteid'] = ';'.join(site_ids)
        else:
            raise ValueError("Must provide geographic filter or site_ids")

        # Characteristic filters
        if characteristics:
            # Join characteristic names with semicolon
            params['characteristicName'] = ';'.join(characteristics)

        # Date filters
        if start_date:
            params['startDateLo'] = start_date.strftime('%m-%d-%Y')
        if end_date:
            params['startDateHi'] = end_date.strftime('%m-%d-%Y')

        try:
            response = self._make_request('Result', params)

            # Parse CSV response
            df = pd.read_csv(io.StringIO(response.text))

            if df.empty:
                logger.warning("No water quality data found for the given criteria")
                return df

            logger.info(f"Retrieved {len(df)} water quality measurements")

            # Convert date columns to datetime
            if 'ActivityStartDate' in df.columns:
                df['ActivityStartDate'] = pd.to_datetime(df['ActivityStartDate'])

            return df

        except Exception as e:
            logger.error(f"Error retrieving water quality data: {str(e)}")
            return pd.DataFrame()

    def get_data_by_state(
        self,
        state_code: str,
        start_date: datetime,
        end_date: datetime,
        characteristics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Convenience method to get water quality data for an entire state.

        Args:
            state_code: Two-letter state code (e.g., 'VA', 'MD')
            start_date: Start date
            end_date: End date
            characteristics: List of characteristic names

        Returns:
            DataFrame with water quality data
        """
        if characteristics is None:
            characteristics = list(self.CHARACTERISTICS.values())

        return self.get_water_quality_data(
            state_code=state_code,
            start_date=start_date,
            end_date=end_date,
            characteristics=characteristics
        )

    def get_data_by_location(
        self,
        latitude: float,
        longitude: float,
        radius_miles: float = 50.0,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        characteristics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Convenience method to get water quality data near a location.

        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            radius_miles: Search radius in miles
            start_date: Start date
            end_date: End date
            characteristics: List of characteristic names

        Returns:
            DataFrame with water quality data
        """
        if characteristics is None:
            characteristics = list(self.CHARACTERISTICS.values())

        return self.get_water_quality_data(
            latitude=latitude,
            longitude=longitude,
            radius_miles=radius_miles,
            start_date=start_date,
            end_date=end_date,
            characteristics=characteristics
        )


if __name__ == "__main__":
    # Example usage
    client = WQPClient()

    # Example: Get data near Washington, DC
    dc_lat, dc_lon = 38.9072, -77.0369

    print(f"Fetching water quality data near Washington, DC ({dc_lat}, {dc_lon})...")
    print("This may take a minute...")

    # Get recent data (last 60 days) near DC
    end_date = datetime.now()
    start_date = datetime(2024, 1, 1)  # Get 2024 data

    data = client.get_data_by_location(
        latitude=dc_lat,
        longitude=dc_lon,
        radius_miles=25,
        start_date=start_date,
        end_date=end_date,
        characteristics=['pH', 'Temperature, water', 'Dissolved oxygen (DO)']
    )

    if not data.empty:
        print(f"\n✓ Retrieved {len(data)} measurements")
        print(f"Date range: {data['ActivityStartDate'].min()} to {data['ActivityStartDate'].max()}")

        if 'CharacteristicName' in data.columns:
            print(f"\nCharacteristics found:")
            print(data['CharacteristicName'].value_counts())

        if 'MonitoringLocationIdentifier' in data.columns:
            print(f"\nNumber of monitoring locations: {data['MonitoringLocationIdentifier'].nunique()}")

        print(f"\nSample data:")
        print(data[['ActivityStartDate', 'MonitoringLocationIdentifier',
                    'CharacteristicName', 'ResultMeasureValue', 'ResultMeasure/MeasureUnitCode']].head(10))
    else:
        print("✗ No data retrieved")
