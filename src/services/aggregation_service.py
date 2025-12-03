from __future__ import annotations

from datetime import datetime
from typing import Dict

import pandas as pd

from utils.wqi_calculator import WQICalculator
from services.constants import PARAM_MAPPING


def get_daily_wqi_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw measurements into daily WQI scores (one row per day).

    Expects long-format WQP/USGS data with ``ActivityStartDate``,
    ``CharacteristicName`` and ``ResultMeasureValue`` columns. For each
    day/parameter pair we take the median measurement and compute an overall
    WQI using :class:`WQICalculator`.
    """
    if df.empty or "ActivityStartDate" not in df.columns:
        return pd.DataFrame()

    calculator = WQICalculator()

    df = df.copy()
    df["ActivityStartDate"] = pd.to_datetime(df["ActivityStartDate"])

    rows = []
    for date, date_df in df.groupby("ActivityStartDate"):
        params: Dict[str, float] = {}
        for characteristic_name, param_key in PARAM_MAPPING.items():
            mask = date_df["CharacteristicName"] == characteristic_name
            values = pd.to_numeric(
                date_df.loc[mask, "ResultMeasureValue"],
                errors="coerce",
            ).dropna()
            if len(values) > 0:
                params[param_key] = float(values.median())
        if params:
            try:
                wqi, _, _ = calculator.calculate_wqi(**params)
                rows.append({"Date": date, "WQI": wqi})
            except Exception:
                # Skip days where WQI cannot be computed for any reason
                continue

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)

