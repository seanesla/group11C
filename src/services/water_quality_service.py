from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable, Dict, List, Optional

import pandas as pd

from data_collection.constants import (
    GROUNDWATER_SITE_TYPES_WQP,
    SURFACE_WATER_SITE_TYPES_WQP,
)
from data_collection.fallback import fetch_with_fallback
from data_collection.usgs_client import USGSClient
from data_collection.wqp_client import WQPClient
from geolocation.zipcode_mapper import ZipCodeMapper
from services.constants import PARAM_MAPPING
from services.result_types import FetchResult, ServiceResult, WQIResult
from services.search_strategies import build_search_strategies
from utils.wqi_calculator import WQICalculator


logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]


def fetch_water_quality_data(
    zip_code: str,
    radius_miles: float,
    start_date: datetime,
    end_date: datetime,
    include_groundwater: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
) -> ServiceResult[FetchResult]:
    """Fetch water quality data for a ZIP code using WQP + USGS with fallbacks."""
    try:
        mapper = ZipCodeMapper()
        client = WQPClient()
        usgs_client = USGSClient()

        # Validate ZIP code
        if not mapper.is_valid_zipcode(zip_code):
            return ServiceResult.fail(f"Invalid ZIP code: {zip_code}")

        coords = mapper.get_coordinates(zip_code)
        if coords is None:
            return ServiceResult.fail(
                f"Could not find coordinates for ZIP code: {zip_code}"
            )

        lat, lon = coords

        characteristics = [
            "pH",
            "Dissolved oxygen (DO)",
            "Temperature, water",
            "Turbidity",
            "Nitrate",
            "Specific conductance",
        ]

        # Build site_types based on user preference
        site_types = SURFACE_WATER_SITE_TYPES_WQP.copy()
        if include_groundwater:
            site_types.extend(GROUNDWATER_SITE_TYPES_WQP)

        strategies = build_search_strategies(
            radius_miles=radius_miles,
            start_date=start_date,
            end_date=end_date,
        )

        attempt_history: List[str] = []

        for idx, strategy in enumerate(strategies, 1):
            description = strategy.describe()
            attempt_history.append(description)

            if progress_callback is not None:
                progress_callback(idx, len(strategies), description)

            df, source_label = fetch_with_fallback(
                latitude=lat,
                longitude=lon,
                radius_miles=strategy.radius_miles,
                start_date=strategy.start_date,
                end_date=strategy.end_date,
                characteristics=characteristics,
                wqp_client=client,
                usgs_client=usgs_client,
                site_types=site_types,
            )

            if df is not None and not df.empty:
                label = source_label or "WQP/USGS"
                context = f"{label} · {description}"
                if strategy.auto_adjusted:
                    context += " (auto-extended)"
                if include_groundwater:
                    context += " [incl. groundwater]"

                return ServiceResult.ok(
                    FetchResult(df=df, source_label=context, attempt_history=attempt_history)
                )

        attempts_text = "; ".join(attempt_history)
        gw_note = (
            " Consider enabling 'Include groundwater data' if this is a groundwater-dependent area."
            if not include_groundwater
            else ""
        )
        error_msg = (
            f"No water quality data found for ZIP {zip_code} after trying: {attempts_text}. "
            f"Try a different ZIP or select a broader date range.{gw_note}"
        )
        return ServiceResult.fail(error_msg)

    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Error fetching water quality data for ZIP %s", zip_code)
        return ServiceResult.fail(f"Error fetching data: {exc}")


def calculate_overall_wqi(df: pd.DataFrame) -> ServiceResult[WQIResult]:
    """Calculate overall WQI from long-format measurements DataFrame."""
    if df.empty:
        return ServiceResult.fail("No measurements available for WQI calculation.")

    calculator = WQICalculator()

    # Aggregate medians by characteristic name
    aggregated: Dict[str, float] = {}
    for characteristic_name, param_key in PARAM_MAPPING.items():
        mask = df["CharacteristicName"] == characteristic_name
        values = pd.to_numeric(
            df.loc[mask, "ResultMeasureValue"],
            errors="coerce",
        ).dropna()
        if len(values) > 0:
            aggregated[param_key] = float(values.median())

    if not aggregated:
        return ServiceResult.fail("No core parameters found in dataset.")

    warnings: List[str] = []
    # Warn if conductance is elevated (possible brackish or mineral-rich water)
    if aggregated.get("conductance", 0.0) > 3000:
        warnings.append(
            f"High conductance ({aggregated['conductance']:.0f} µS/cm) - possible saltwater or mineral influence."
        )

    try:
        wqi, scores, classification = calculator.calculate_wqi(**aggregated)
        result = WQIResult(
            wqi=float(wqi),
            scores=scores,
            classification=classification,
            aggregated=aggregated,
        )
        return ServiceResult.ok(result, warnings=warnings)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Error calculating WQI")
        return ServiceResult.fail(f"Error calculating WQI: {exc}")
