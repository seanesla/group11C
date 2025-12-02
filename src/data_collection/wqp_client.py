"""
Water Quality Portal (WQP) API Client

This module provides functionality to fetch water quality data from the Water Quality Portal,
which aggregates data from USGS, EPA, and other water quality monitoring agencies.

API Documentation: https://www.waterqualitydata.us/webservices_documentation/
"""

import os
import requests
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any
import time
import logging
import io
import warnings

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

    @staticmethod
    def _get_validated_timeout(env_var: str, default: int, min_val: int = 5, max_val: int = 300) -> int:
        """Get timeout from environment with bounds checking."""
        try:
            value = int(os.getenv(env_var, default))
            if not (min_val <= value <= max_val):
                logger.warning(f"{env_var}={value} outside range [{min_val}, {max_val}], using {default}")
                return default
            return value
        except ValueError:
            logger.warning(f"Invalid {env_var} value, using default {default}")
            return default

    def __init__(self, rate_limit_delay: float = 1.0, timeout: Optional[int] = None):
        """
        Initialize the WQP client.

        Args:
            rate_limit_delay: Delay in seconds between API requests
            timeout: Request timeout in seconds (default: WQP_TIMEOUT env var or 20)
        """
        default_timeout = self._get_validated_timeout("WQP_TIMEOUT", 20)
        self.rate_limit_delay = rate_limit_delay
        self.timeout = timeout if timeout is not None else default_timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'WaterQualityPrediction/1.0 (Educational Project)'
        })

    def _make_request(self, endpoint: str, params: Dict[str, Any], retries: int = 3) -> requests.Response:
        """
        Make an API request with rate limiting, retry logic, and error handling.

        Args:
            endpoint: The API endpoint (e.g., 'Result', 'Station')
            params: Query parameters
            retries: Number of retry attempts for transient failures

        Returns:
            Response object

        Raises:
            requests.RequestException: If the request fails after all retries
        """
        url = f"{self.BASE_URL}/{endpoint}/search"

        last_exc = None
        for attempt in range(retries):
            time.sleep(self.rate_limit_delay * (2 ** attempt if attempt > 0 else 1))

            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                return response
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_exc = e
                if attempt < retries - 1:
                    logger.warning(f"Transient error, retrying ({attempt+1}/{retries}): {type(e).__name__}")
                    continue
                logger.error(f"Request failed after {retries} attempts: {e}")
            except requests.exceptions.HTTPError as e:
                last_exc = e
                status = e.response.status_code if e.response else 'unknown'
                # Handle rate limiting (429)
                if status == 429 and attempt < retries - 1:
                    retry_after = int(e.response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limited (429), waiting {retry_after}s before retry")
                    time.sleep(retry_after)
                    continue
                if status >= 500 and attempt < retries - 1:
                    logger.warning(f"HTTP {status} from WQP, retrying ({attempt+1}/{retries})")
                    continue
                logger.error(f"HTTP error {status} from WQP API")
            except requests.exceptions.RequestException as e:
                last_exc = e
                logger.error(f"Request failed: {e}")
                break

        raise last_exc or RuntimeError("Unknown WQP request failure")

    def get_stations(
        self,
        bbox: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        radius_miles: Optional[float] = None,
        state_code: Optional[str] = None,
        county_code: Optional[str] = None,
        site_types: Optional[List[str]] = None,
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
            site_types: List of WQP site types to include (e.g., ['Stream', 'Lake']).
                       Defaults to surface water types, excluding marine/estuarine.

        Returns:
            DataFrame containing station information
        """
        from .constants import SURFACE_WATER_SITE_TYPES_WQP

        if site_types is None:
            site_types = SURFACE_WATER_SITE_TYPES_WQP

        params = {
            'mimeType': 'csv',
            'sorted': 'yes',
            'siteType': ';'.join(site_types),
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
            df = pd.read_csv(io.StringIO(response.text), dtype=str)
            logger.info(f"Found {len(df)} monitoring stations")

            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error retrieving stations: {e}")
            return pd.DataFrame()
        except (ValueError, KeyError, pd.errors.ParserError) as e:
            logger.error(f"Data parsing error retrieving stations: {e}")
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
        site_ids: Optional[List[str]] = None,
        site_types: Optional[List[str]] = None,
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
            site_types: List of WQP site types to include (e.g., ['Stream', 'Lake']).
                       Defaults to surface water types, excluding marine/estuarine.

        Returns:
            DataFrame containing water quality measurements
        """
        from .constants import SURFACE_WATER_SITE_TYPES_WQP

        if site_types is None:
            site_types = SURFACE_WATER_SITE_TYPES_WQP

        params = {
            'mimeType': 'csv',
            'sorted': 'yes',
            'sampleMedia': 'Water',  # Focus on water samples
            'siteType': ';'.join(site_types),
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

            # Standardize nitrate units (convert mg{NO3}/L → mg/L as N if needed)
            df = self._standardize_nitrate_unit(df)

            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error retrieving water quality data: {e}")
            return pd.DataFrame()
        except (ValueError, KeyError, pd.errors.ParserError) as e:
            logger.error(f"Data parsing error retrieving water quality data: {e}")
            return pd.DataFrame()

    def get_data_by_state(
        self,
        state_code: str,
        start_date: datetime,
        end_date: datetime,
        characteristics: Optional[List[str]] = None,
        site_types: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Convenience method to get water quality data for an entire state.

        Args:
            state_code: Two-letter state code (e.g., 'VA', 'MD')
            start_date: Start date
            end_date: End date
            characteristics: List of characteristic names
            site_types: List of WQP site types to include. Defaults to surface water.

        Returns:
            DataFrame with water quality data
        """
        if characteristics is None:
            characteristics = list(self.CHARACTERISTICS.values())

        return self.get_water_quality_data(
            state_code=state_code,
            start_date=start_date,
            end_date=end_date,
            characteristics=characteristics,
            site_types=site_types,
        )

    def get_data_by_location(
        self,
        latitude: float,
        longitude: float,
        radius_miles: float = 50.0,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        characteristics: Optional[List[str]] = None,
        site_types: Optional[List[str]] = None,
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
            site_types: List of WQP site types to include. Defaults to surface water.

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
            characteristics=characteristics,
            site_types=site_types,
        )

    def _standardize_nitrate_unit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize nitrate units to mg/L as N (EPA standard).

        Supports both:
        - Wide-format training/test data with ``nitrate`` / ``nitrate_unit`` columns
        - Long-format WQP API responses with ``CharacteristicName``,
          ``ResultMeasureValue``, and ``ResultMeasure/MeasureUnitCode``

        Wide-format behaviour is exercised directly in unit tests
        (tests/test_nitrate_unit_system.py) and is kept backwards compatible.
        For long-format WQP data we apply a conservative strategy:
        - If units are reported in µmol/L, convert to mg/L as N using the
          molecular weight of nitrogen.
        - If units are already reported in mg/L, we assume they are mg/L as N,
          which matches USGS conventions for nitrate-as-N reporting.

        Args:
            df: DataFrame with water quality data (long or wide format)

        Returns:
            DataFrame with standardized nitrate units
        """
        if df is None or df.empty:
            return df

        # --- Wide-format path (used in unit tests) ---
        if 'nitrate' in df.columns:
            # Check if we have unit information (only relevant for wide-format data)
            if 'nitrate_unit' not in df.columns:
                logger.warning("No unit information for nitrate - assuming mg/L as N (EPA standard)")
                return df

            # Nitrate unit conversion constant: mg{NO3}/L -> mg/L as N
            NITRATE_NO3_TO_N = 0.2258  # Atomic weight N / Molecular weight NO3

            for idx, row in df.iterrows():
                if pd.notna(row.get('nitrate')) and pd.notna(row.get('nitrate_unit')):
                    unit = str(row['nitrate_unit']).lower()

                    # mg{NO3}/L or mg/l as NO3 → convert to mg/L as N
                    if 'no3' in unit and 'as n' not in unit:
                        original_value = row['nitrate']
                        df.at[idx, 'nitrate'] = original_value * NITRATE_NO3_TO_N
                        logger.debug(
                            "Converted nitrate from mg{NO3}/L to mg/L as N: "
                            f"{original_value:.2f} → {df.at[idx, 'nitrate']:.2f}"
                        )

                    # mg/L as N or mg/L as NO3-N → already correct
                    elif 'as n' in unit or 'no3-n' in unit:
                        logger.debug(f"Nitrate already in mg/L as N: {row['nitrate']:.2f}")

                    # Unexpected unit
                    else:
                        warnings.warn(
                            f"Unexpected nitrate unit '{row['nitrate_unit']}' at index {idx}. "
                            "Assuming mg/L as N. Please verify correctness.",
                            UserWarning,
                        )

            return df

        # --- Long-format WQP path ---
        if 'CharacteristicName' not in df.columns or 'ResultMeasureValue' not in df.columns:
            # Not a schema we know how to normalize; leave unchanged
            return df

        unit_col = 'ResultMeasure/MeasureUnitCode'
        if unit_col not in df.columns:
            logger.debug(
                "No ResultMeasure/MeasureUnitCode column for nitrate - assuming mg/L as N "
                "for any nitrate measurements."
            )
            return df

        # Conversion constant: µmol/L -> mg/L as N
        UMOL_PER_L_TO_MG_PER_L_N = 14.0067 / 1000.0

        nitrate_mask = df['CharacteristicName'].str.contains('Nitrate', case=False, na=False)
        for idx, row in df.loc[nitrate_mask].iterrows():
            value = row.get('ResultMeasureValue')
            unit = row.get(unit_col)
            if pd.isna(value) or pd.isna(unit):
                continue

            unit_str = str(unit).lower()

            if 'umol' in unit_str:
                original_value = value
                df.at[idx, 'ResultMeasureValue'] = float(original_value) * UMOL_PER_L_TO_MG_PER_L_N
                df.at[idx, unit_col] = 'mg/L as N'
                logger.debug(
                    "Converted nitrate from umol/L to mg/L as N: "
                    f"{original_value} → {df.at[idx, 'ResultMeasureValue']:.4f}"
                )
            else:
                # For mg/L-style units we assume mg/L as N, consistent with USGS.
                logger.debug(
                    "Leaving nitrate value unchanged for unit '%s' at index %s",
                    unit_str,
                    idx,
                )

        return df


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
