"""Utilities for creating US-based training samples from WQP/USGS data."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd

from src.geolocation.zipcode_mapper import ZipCodeMapper
from src.data_collection.wqp_client import WQPClient
from src.data_collection.usgs_client import USGSClient
from src.data_collection.fallback import fetch_with_fallback
from src.services.search_strategies import build_search_strategies
from src.utils.wqi_calculator import WQICalculator

PARAMETER_MAPPING = {
    "pH": "ph",
    "Dissolved oxygen (DO)": "dissolved_oxygen",
    "Temperature, water": "temperature",
    "Turbidity": "turbidity",
    "Nitrate": "nitrate",
    "Specific conductance": "conductance",
}


def _aggregate_daily_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """Convert long-format WQP data into one row per sample date."""

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["ActivityStartDate"] = pd.to_datetime(df["ActivityStartDate"])
    calculator = WQICalculator()

    rows = []
    for activity_date, group in df.groupby(df["ActivityStartDate"].dt.date):
        params = {}
        for characteristic, key in PARAMETER_MAPPING.items():
            values = pd.to_numeric(
                group.loc[group["CharacteristicName"] == characteristic, "ResultMeasureValue"],
                errors="coerce",
            ).dropna()
            if not values.empty:
                params[key] = float(values.median())

        if len(params) < 3:
            continue  # too little data to compute meaningful WQI

        try:
            wqi, _, classification = calculator.calculate_wqi(**params)
        except Exception:
            continue

        rows.append(
            {
                "sample_date": pd.Timestamp(activity_date),
                "year": pd.Timestamp(activity_date).year,
                "monitoring_location": group["MonitoringLocationIdentifier"].iloc[0],
                "wqi_score": wqi,
                "wqi_classification": classification,
                "is_safe": 1 if wqi >= 70 else 0,
                **params,
            }
        )

    if not rows:
        return pd.DataFrame()

    agg_df = pd.DataFrame(rows)
    agg_df = agg_df.sort_values("sample_date").reset_index(drop=True)
    return agg_df


def collect_us_samples_for_zip(
    zip_code: str,
    *,
    radius_miles: float = 40.0,
    lookback_years: int = 5,
    mapper: Optional[ZipCodeMapper] = None,
    wqp_client: Optional[WQPClient] = None,
    usgs_client: Optional[USGSClient] = None,
) -> pd.DataFrame:
    """Fetch and aggregate water quality samples for a single ZIP code."""

    mapper = mapper or ZipCodeMapper()
    wqp_client = wqp_client or WQPClient()
    usgs_client = usgs_client or USGSClient()

    if not mapper.is_valid_zipcode(zip_code):
        raise ValueError(f"Invalid ZIP code: {zip_code}")

    coords = mapper.get_coordinates(zip_code)
    if not coords:
        raise ValueError(f"No coordinates for ZIP {zip_code}")

    location_info = mapper.get_location_info(zip_code) or {}

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=lookback_years * 365)

    characteristics = list(PARAMETER_MAPPING.keys())
    strategies = build_search_strategies(
        radius_miles=radius_miles,
        start_date=start_date,
        end_date=end_date,
        max_radius=110.0,
    )

    for strategy in strategies:
        df, source_label = fetch_with_fallback(
            latitude=coords[0],
            longitude=coords[1],
            radius_miles=strategy.radius_miles,
            start_date=strategy.start_date,
            end_date=strategy.end_date,
            characteristics=characteristics,
            wqp_client=wqp_client,
            usgs_client=usgs_client,
        )

        aggregated = _aggregate_daily_measurements(df)
        if aggregated.empty:
            continue

        aggregated["zip_code"] = zip_code
        aggregated["state_code"] = location_info.get("state_code")
        aggregated["data_source"] = source_label or "WQP/USGS"
        aggregated["search_radius_mi"] = strategy.radius_miles
        aggregated["history_years"] = round((strategy.end_date - strategy.start_date).days / 365.0, 2)
        aggregated["strategy_label"] = strategy.describe()
        aggregated["auto_adjusted"] = strategy.auto_adjusted
        return aggregated

    return pd.DataFrame()


def merge_us_samples(zip_codes: List[str], *, radius_miles: float = 40.0, lookback_years: int = 5) -> pd.DataFrame:
    """Collect samples for multiple ZIP codes and merge them vertically."""

    mapper = ZipCodeMapper()
    wqp_client = WQPClient()
    usgs_client = USGSClient()

    collected = []
    for zip_code in zip_codes:
        try:
            samples = collect_us_samples_for_zip(
                zip_code,
                radius_miles=radius_miles,
                lookback_years=lookback_years,
                mapper=mapper,
                wqp_client=wqp_client,
                usgs_client=usgs_client,
            )
            if not samples.empty:
                collected.append(samples)
        except Exception:
            continue

    if not collected:
        return pd.DataFrame()

    return pd.concat(collected, ignore_index=True)
