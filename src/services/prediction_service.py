from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from preprocessing.us_data_features import prepare_us_features_for_prediction
from services.constants import WQI_COLORS
from services.result_types import MLPredictionResult
from utils.wqi_calculator import WQICalculator


logger = logging.getLogger(__name__)


def make_ml_predictions(
    aggregated_params: Dict[str, float],
    classifier,
    regressor,
    year: Optional[int] = None,
) -> Optional[MLPredictionResult]:
    """Run ML models on aggregated water-quality parameters.

    Returns an :class:`MLPredictionResult` if both models are available and the
    prediction succeeds, otherwise ``None``.
    """
    if classifier is None or regressor is None:
        return None

    if year is None:
        year = datetime.now().year

    try:
        # Prepare 18-feature US prediction schema
        features = prepare_us_features_for_prediction(
            ph=aggregated_params.get("ph"),
            dissolved_oxygen=aggregated_params.get("dissolved_oxygen"),
            temperature=aggregated_params.get("temperature"),
            turbidity=aggregated_params.get("turbidity"),
            nitrate=aggregated_params.get("nitrate"),
            conductance=aggregated_params.get("conductance"),
            year=year,
        )

        # Make predictions
        is_safe_pred = classifier.predict(features)[0]
        is_safe_proba = classifier.predict_proba(features)[0]  # [P(unsafe), P(safe)]
        wqi_pred = float(regressor.predict(features)[0])

        # Derive 5-level classification from regressor's WQI prediction
        classification = WQICalculator.classify_wqi(wqi_pred)
        margins = WQICalculator.get_margin_to_thresholds(wqi_pred)
        is_near, near_threshold = WQICalculator.is_near_threshold(wqi_pred)

        return MLPredictionResult(
            predicted_wqi=wqi_pred,
            predicted_classification=classification,
            wqi_color=WQI_COLORS.get(classification, "#808080"),
            margin_up=margins["margin_up"],
            margin_down=margins["margin_down"],
            next_up=margins["next_up"],
            next_down=margins["next_down"],
            is_near_threshold=is_near,
            near_threshold_name=near_threshold,
            # Keep for backwards compatibility with historical SAFE threshold
            is_safe=bool(wqi_pred >= 70),
            confidence=float(max(is_safe_proba)),
        )

    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("ML prediction failed: %s", exc)
        return None


def build_forecast_from_history(
    daily_wqi: pd.DataFrame,
    periods: int = 12,
    current_wqi_override: Optional[float] = None,
    forecast_start_date: Optional[datetime] = None,
) -> Optional[Dict[str, object]]:
    """Derive a simple month-level WQI forecast from recent history.

    Uses linear extrapolation over the last ~180 days of daily WQI values.
    Returns a dict compatible with the existing Streamlit chart helpers.
    """
    if daily_wqi is None or daily_wqi.empty or len(daily_wqi) < 3:
        return None

    # Use last 180 days to reduce noise
    cutoff = daily_wqi["Date"].max() - pd.Timedelta(days=180)
    recent = daily_wqi[daily_wqi["Date"] >= cutoff] if len(daily_wqi) > 30 else daily_wqi

    ordinals = recent["Date"].map(datetime.toordinal).to_numpy()
    wqis = recent["WQI"].to_numpy()

    # If variance is zero, no meaningful slope
    if np.isclose(wqis.std(), 0):
        return None

    slope, intercept = np.polyfit(ordinals, wqis, 1)  # WQI per day
    historical_last_date = recent["Date"].max()
    historical_current_wqi = float(recent.loc[recent["Date"].idxmax(), "WQI"])

    # Use override for display baseline if provided, otherwise use historical
    baseline_wqi = (
        float(current_wqi_override)
        if current_wqi_override is not None
        else historical_current_wqi
    )

    # Use provided forecast start date (today) or fall back to last historical date
    projection_start = forecast_start_date or historical_last_date

    dates = []
    predictions = []
    for i in range(1, periods + 1):
        future_date = projection_start + relativedelta(months=i)
        days_ahead = (future_date - projection_start).days
        # Project from baseline using historical slope
        pred = baseline_wqi + slope * days_ahead
        pred = max(0.0, min(100.0, float(pred)))
        dates.append(future_date)
        predictions.append(pred)

    final_wqi = predictions[-1]
    wqi_change = final_wqi - baseline_wqi
    if wqi_change > 2:
        trend = "improving"
    elif wqi_change < -2:
        trend = "declining"
    else:
        trend = "stable"

    return {
        "dates": dates,
        "predictions": predictions,
        "trend": trend,
        "trend_slope": wqi_change / periods,
        "current_wqi": baseline_wqi,
        "final_wqi": final_wqi,
        "wqi_change": wqi_change,
        "periods": periods,
        "frequency": "M",
        "method": "historical_linear",
    }

