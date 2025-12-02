"""
Fallback data source utilities for water quality lookups.

Provides helpers to convert USGS NWIS responses into the long-format shape
expected by downstream WQP-based processing, and a generic fetch-with-fallback
routine that first tries WQP then USGS.

Also provides validation utilities to detect marine contamination and
other data quality issues.
"""

import logging
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd

from .constants import MARINE_CONDUCTANCE_THRESHOLD

logger = logging.getLogger(__name__)


# Mapping from USGS parameter codes to the characteristic names used in WQP.
USGS_CODE_TO_CHARACTERISTIC = {
    "00400": "pH",
    "00300": "Dissolved oxygen (DO)",
    "00010": "Temperature, water",
    "63680": "Turbidity",
    "99133": "Nitrate",
    "00095": "Specific conductance",
}


def convert_usgs_to_wqp_format(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a USGS NWIS dataframe to the long-format schema used by WQP.

    Expected input columns (subset):
      - site_code
      - parameter_code
      - parameter_name
      - datetime
      - value

    Output columns:
      - MonitoringLocationIdentifier
      - ActivityStartDate (date)
      - CharacteristicName
      - ResultMeasureValue
    """

    if df is None or df.empty:
        return pd.DataFrame()

    # Map parameter codes to characteristic names; drop rows we cannot map.
    df = df.copy()
    df["CharacteristicName"] = df["parameter_code"].map(USGS_CODE_TO_CHARACTERISTIC)
    df = df.dropna(subset=["CharacteristicName", "value", "datetime", "site_code"])

    # Normalize column names
    out = pd.DataFrame(
        {
            "MonitoringLocationIdentifier": df["site_code"],
            "ActivityStartDate": pd.to_datetime(df["datetime"]).dt.date,
            "CharacteristicName": df["CharacteristicName"],
            "ResultMeasureValue": pd.to_numeric(df["value"], errors="coerce"),
        }
    )

    # Drop any rows that failed numeric coercion
    out = out.dropna(subset=["ResultMeasureValue"])
    return out.reset_index(drop=True)


def validate_water_quality_data(
    df: pd.DataFrame,
    conductance_col: str = "conductance",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validate water quality data and filter out suspected marine contamination.

    This function acts as a safety net to catch any marine/estuarine data that
    may have slipped through the site type filtering at the API level.

    Args:
        df: DataFrame with water quality measurements. Can be either:
            - Aggregated format with 'conductance' column
            - Long-format WQP with 'CharacteristicName' and 'ResultMeasureValue'
        conductance_col: Name of conductance column (for aggregated format)

    Returns:
        Tuple of (filtered_df, warnings_list)
    """
    if df is None or df.empty:
        return df, []

    warnings = []
    df = df.copy()

    # Handle aggregated format (e.g., from Streamlit app)
    if conductance_col in df.columns:
        high_cond = df[conductance_col] > MARINE_CONDUCTANCE_THRESHOLD
        if high_cond.any():
            count = high_cond.sum()
            warnings.append(
                f"Excluded {count} record(s) with conductance > "
                f"{MARINE_CONDUCTANCE_THRESHOLD} uS/cm (possible marine contamination)"
            )
            logger.warning(warnings[-1])
            df = df[~high_cond]

    # Handle long-format WQP data
    elif "CharacteristicName" in df.columns and "ResultMeasureValue" in df.columns:
        cond_mask = df["CharacteristicName"] == "Specific conductance"
        cond_values = pd.to_numeric(
            df.loc[cond_mask, "ResultMeasureValue"], errors="coerce"
        )
        high_cond_idx = cond_values[cond_values > MARINE_CONDUCTANCE_THRESHOLD].index

        if len(high_cond_idx) > 0:
            # Get the monitoring locations with high conductance
            problem_locations = df.loc[high_cond_idx, "MonitoringLocationIdentifier"].unique()
            warnings.append(
                f"Found {len(high_cond_idx)} conductance readings > "
                f"{MARINE_CONDUCTANCE_THRESHOLD} uS/cm from {len(problem_locations)} "
                f"station(s) (possible marine contamination)"
            )
            logger.warning(warnings[-1])

            # Remove ALL records from stations with high conductance
            # (they may have other contaminated readings too)
            if "MonitoringLocationIdentifier" in df.columns:
                df = df[~df["MonitoringLocationIdentifier"].isin(problem_locations)]

    return df, warnings


def fetch_with_fallback(
    *,
    latitude: float,
    longitude: float,
    radius_miles: float,
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    characteristics: List[str],
    wqp_client,
    usgs_client,
) -> Tuple[pd.DataFrame, str]:
    """Try WQP first; if empty or error, try USGS. Returns (df, source_label)."""

    # WQP first
    try:
        df = wqp_client.get_data_by_location(
            latitude=latitude,
            longitude=longitude,
            radius_miles=radius_miles,
            start_date=start_date,
            end_date=end_date,
            characteristics=characteristics,
        )
        if df is not None and not df.empty:
            return df, "WQP"
    except Exception as e:
        logger.warning(f"WQP fetch failed, falling back to USGS: {type(e).__name__}: {e}")

    # USGS fallback
    try:
        usgs_df = usgs_client.get_data_by_location(
            latitude=latitude,
            longitude=longitude,
            radius_miles=radius_miles,
            start_date=start_date,
            end_date=end_date,
            parameters=[
                "ph",
                "dissolved_oxygen",
                "temperature",
                "turbidity",
                "nitrate",
                "specific_conductance",
            ],
        )

        converted = convert_usgs_to_wqp_format(usgs_df)
        if not converted.empty:
            return converted, "USGS NWIS"
    except Exception as e:
        logger.warning(f"USGS fallback also failed: {type(e).__name__}: {e}")

    return pd.DataFrame(), ""
