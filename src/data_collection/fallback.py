"""
Fallback data source utilities for water quality lookups.

Provides helpers to convert USGS NWIS responses into the long-format shape
expected by downstream WQP-based processing, and a generic fetch-with-fallback
routine that first tries WQP then USGS.
"""

from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd


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
    except Exception:
        # Swallow and fall back to USGS
        pass

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
    except Exception:
        pass

    return pd.DataFrame(), ""
