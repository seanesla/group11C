"""
USGS National Water Information System (NWIS) API Client

This module provides functionality to fetch water quality data from the USGS NWIS API.
It supports retrieval of various water quality parameters including pH, dissolved oxygen,
temperature, turbidity, and nitrate levels.

API Documentation: https://waterservices.usgs.gov/rest/
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import time
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class USGSClient:
    """Client for interacting with the USGS NWIS API."""

    BASE_URL = "https://waterservices.usgs.gov/nwis/iv/"
    SITE_URL = "https://waterservices.usgs.gov/nwis/site/"

    # USGS parameter codes for water quality metrics
    # Reference: https://help.waterdata.usgs.gov/parameter_cd?group_cd=PHY
    PARAMETER_CODES = {
        'temperature': '00010',      # Temperature, water, degrees Celsius
        'ph': '00400',               # pH, water, unfiltered, field, standard units
        'dissolved_oxygen': '00300', # Dissolved oxygen, water, unfiltered, milligrams per liter
        'turbidity': '63680',        # Turbidity, water, unfiltered, monochrome near infra-red LED light, 780-900 nm, detection angle 90 +-2.5 degrees, formazin nephelometric units (FNU)
        'specific_conductance': '00095',  # Specific conductance, water, unfiltered, microsiemens per centimeter at 25 degrees Celsius
        'nitrate': '99133',          # Nitrate, water, filtered, milligrams per liter as nitrogen
    }

    def __init__(self, rate_limit_delay: float = 1.0, timeout: Optional[int] = None):
        """
        Initialize the USGS client.

        Args:
            rate_limit_delay: Delay in seconds between API requests to avoid rate limiting
        """
        default_timeout = int(os.getenv("USGS_TIMEOUT", 30))
        self.rate_limit_delay = rate_limit_delay
        self.timeout = timeout if timeout is not None else default_timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'WaterQualityPrediction/1.0 (Educational Project)'
        })

    def _make_request(self, url: str, params: Dict[str, Any]) -> requests.Response:
        """
        Make an API request with rate limiting and error handling.

        Args:
            url: The API endpoint URL
            params: Query parameters for the request

        Returns:
            Response object

        Raises:
            requests.RequestException: If the request fails
        """
        time.sleep(self.rate_limit_delay)

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout for URL: {url}")
            raise
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            raise

    def _calculate_bounding_box(
        self,
        latitude: float,
        longitude: float,
        radius_miles: float
    ) -> str:
        """
        Calculate a bounding box around a point.

        Args:
            latitude: Center latitude
            longitude: Center longitude
            radius_miles: Radius in miles

        Returns:
            Bounding box string in format: "west,south,east,north"
        """
        # Approximate degrees per mile (varies by latitude)
        # At 40° latitude: 1° latitude ≈ 69 miles, 1° longitude ≈ 53 miles
        miles_per_degree_lat = 69.0
        miles_per_degree_lon = 69.0 * abs(np.cos(np.radians(latitude)))

        # Calculate offsets
        lat_offset = radius_miles / miles_per_degree_lat
        lon_offset = radius_miles / miles_per_degree_lon

        # Calculate bounding box coordinates
        south = latitude - lat_offset
        north = latitude + lat_offset
        west = longitude - lon_offset
        east = longitude + lon_offset

        # Format: west,south,east,north
        return f"{west:.6f},{south:.6f},{east:.6f},{north:.6f}"

    def find_sites_by_location(
        self,
        latitude: float,
        longitude: float,
        radius_miles: float = 50.0,
        parameter_codes: Optional[List[str]] = None,
        site_types: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Find monitoring sites near a geographic location.

        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            radius_miles: Search radius in miles (default: 50)
            parameter_codes: List of USGS parameter codes to filter by
            site_types: List of USGS site type codes to include (e.g., ['ST', 'LK']).
                       Defaults to surface water types, excluding marine/estuarine.

        Returns:
            DataFrame containing site information
        """
        from .constants import SURFACE_WATER_SITE_TYPES_USGS

        if parameter_codes is None:
            parameter_codes = list(self.PARAMETER_CODES.values())

        if site_types is None:
            site_types = SURFACE_WATER_SITE_TYPES_USGS

        # Calculate bounding box
        bbox = self._calculate_bounding_box(latitude, longitude, radius_miles)

        params = {
            'format': 'rdb',
            'bBox': bbox,
            'parameterCd': ','.join(parameter_codes),
            'siteType': ','.join(site_types),  # Surface water sites (excludes marine)
            'siteStatus': 'active'
        }

        try:
            response = self._make_request(self.SITE_URL, params)

            # Parse RDB format (tab-delimited with comments starting with #)
            lines = response.text.strip().split('\n')
            data_lines = [line for line in lines if not line.startswith('#')]

            if len(data_lines) < 2:
                logger.warning(f"No sites found near ({latitude}, {longitude})")
                return pd.DataFrame()

            # Parse tab-delimited data
            header = data_lines[0].split('\t')
            # Skip the second line (data type descriptors in RDB format)
            data = [line.split('\t') for line in data_lines[2:]]

            df = pd.DataFrame(data, columns=header)
            logger.info(f"Found {len(df)} sites near ({latitude}, {longitude})")

            return df

        except Exception as e:
            logger.error(f"Error finding sites: {str(e)}")
            return pd.DataFrame()

    def get_water_quality_data(
        self,
        site_codes: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        parameters: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Retrieve water quality data for specified sites and parameters.

        Args:
            site_codes: List of USGS site codes
            start_date: Start date for data retrieval (default: 30 days ago)
            end_date: End date for data retrieval (default: today)
            parameters: List of parameter names to retrieve (default: all available)

        Returns:
            DataFrame containing water quality measurements
        """
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=30)

        if parameters is None:
            param_codes = list(self.PARAMETER_CODES.values())
        else:
            param_codes = [self.PARAMETER_CODES.get(p, p) for p in parameters]

        params = {
            'format': 'json',
            'sites': ','.join(site_codes),
            'startDT': start_date.strftime('%Y-%m-%d'),
            'endDT': end_date.strftime('%Y-%m-%d'),
            'parameterCd': ','.join(param_codes),
            'siteStatus': 'all'
        }

        try:
            response = self._make_request(self.BASE_URL, params)
            data = response.json()

            if 'value' not in data or 'timeSeries' not in data['value']:
                logger.warning(f"No data returned for sites: {site_codes}")
                return pd.DataFrame()

            # Parse the JSON response into a structured DataFrame
            records = []
            for series in data['value']['timeSeries']:
                site_code = series['sourceInfo']['siteCode'][0]['value']
                site_name = series['sourceInfo']['siteName']
                param_code = series['variable']['variableCode'][0]['value']
                param_name = series['variable']['variableDescription']
                unit = series['variable']['unit']['unitCode']

                if 'values' in series and len(series['values']) > 0:
                    for value_obj in series['values'][0]['value']:
                        records.append({
                            'site_code': site_code,
                            'site_name': site_name,
                            'parameter_code': param_code,
                            'parameter_name': param_name,
                            'datetime': pd.to_datetime(value_obj['dateTime']),
                            'value': float(value_obj['value']) if value_obj['value'] != '' else None,
                            'unit': unit,
                            'qualifiers': value_obj.get('qualifiers', [])
                        })

            if not records:
                logger.warning(f"No measurements found for sites: {site_codes}")
                return pd.DataFrame()

            df = pd.DataFrame(records)
            logger.info(f"Retrieved {len(df)} measurements from {len(site_codes)} sites")

            # Validate nitrate units (should be mg/L as N for USGS)
            df = self._standardize_nitrate_unit(df)

            return df

        except Exception as e:
            logger.error(f"Error retrieving water quality data: {str(e)}")
            return pd.DataFrame()

    def get_data_by_location(
        self,
        latitude: float,
        longitude: float,
        radius_miles: float = 50.0,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        parameters: Optional[List[str]] = None,
        site_types: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Retrieve water quality data for all sites near a location.

        This is a convenience method that combines site finding and data retrieval.

        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            radius_miles: Search radius in miles
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            parameters: List of parameter names to retrieve
            site_types: List of USGS site type codes. Defaults to surface water.

        Returns:
            DataFrame containing water quality measurements
        """
        # Find sites near the location
        sites_df = self.find_sites_by_location(
            latitude, longitude, radius_miles, site_types=site_types
        )

        if sites_df.empty:
            logger.warning(f"No sites found near ({latitude}, {longitude})")
            return pd.DataFrame()

        # Extract site codes
        site_codes = sites_df['site_no'].unique().tolist()

        # Get water quality data for these sites
        data_df = self.get_water_quality_data(
            site_codes=site_codes,
            start_date=start_date,
            end_date=end_date,
            parameters=parameters
        )

        return data_df

    def _standardize_nitrate_unit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate nitrate units are in mg/L as N (USGS standard).

        USGS NWIS consistently reports nitrate as mg/L as N (parameter code 99133).
        This method validates units match expectations and warns if anomalies detected.

        EPA MCL for nitrate: 10 mg/L as N
        USGS parameter 99133: "Nitrate, water, filtered, milligrams per liter as nitrogen"

        Args:
            df: DataFrame with water quality data

        Returns:
            DataFrame with validated nitrate units
        """
        if df is None or df.empty or 'nitrate' not in df.columns:
            return df

        # Check if we have unit information
        if 'nitrate_unit' not in df.columns:
            logger.debug("No unit information for nitrate - USGS standard is mg/L as N")
            return df

        # Verify all nitrate values use expected USGS units (mg/L as N)
        for idx, row in df.iterrows():
            if pd.notna(row.get('nitrate')) and pd.notna(row.get('nitrate_unit')):
                unit = str(row['nitrate_unit']).lower()

                # Expected: mg/L as N or variations
                if 'as n' in unit or 'as nitrogen' in unit:
                    logger.debug(f"Nitrate in expected USGS format (mg/L as N): {row['nitrate']:.2f}")

                # Unexpected: mg{NO3}/L would require conversion
                elif 'no3' in unit and 'as n' not in unit:
                    warnings.warn(
                        f"Unexpected USGS nitrate unit '{row['nitrate_unit']}' at index {idx}. "
                        f"USGS parameter 99133 should be mg/L as N. Value may be incorrect: {row['nitrate']:.2f}",
                        UserWarning
                    )

                # Unknown unit
                else:
                    warnings.warn(
                        f"Unknown nitrate unit '{row['nitrate_unit']}' at index {idx}. "
                        f"Expected mg/L as N for USGS data. Please verify correctness.",
                        UserWarning
                    )

        return df


if __name__ == "__main__":
    # Example usage
    client = USGSClient()

    # Example: Get data near Washington, DC
    dc_lat, dc_lon = 38.9072, -77.0369

    print(f"Finding sites near Washington, DC ({dc_lat}, {dc_lon})...")

    # First, find sites
    sites_df = client.find_sites_by_location(
        latitude=dc_lat,
        longitude=dc_lon,
        radius_miles=25
    )

    if not sites_df.empty:
        print(f"\nFound {len(sites_df)} sites")
        # Use only first 5 sites for testing
        site_codes = sites_df['site_no'].unique().tolist()[:5]
        print(f"Testing with {len(site_codes)} sites: {site_codes}")

        # Get last 7 days of data
        data = client.get_water_quality_data(
            site_codes=site_codes,
            start_date=datetime.now() - timedelta(days=7),
            end_date=datetime.now()
        )

        if not data.empty:
            print(f"\nRetrieved {len(data)} measurements")
            print(f"Date range: {data['datetime'].min()} to {data['datetime'].max()}")
            print(f"\nParameter distribution:")
            print(data['parameter_name'].value_counts())
            print(f"\nSample data:")
            print(data.head(10))
        else:
            print("\nNo data retrieved from sites")
    else:
        print("No sites found")
