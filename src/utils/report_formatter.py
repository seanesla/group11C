"""Format water quality results for clipboard export."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from utils.wqi_calculator import WQICalculator

# Fixed units for the 6 WQI parameters
PARAM_UNITS = {
    'ph': '',
    'dissolved_oxygen': 'mg/L',
    'temperature': 'C',
    'turbidity': 'NTU',
    'nitrate': 'mg/L as N',
    'conductance': 'uS/cm',
}

PARAM_DISPLAY_NAMES = {
    'ph': 'pH',
    'dissolved_oxygen': 'Dissolved Oxygen',
    'temperature': 'Temperature',
    'turbidity': 'Turbidity',
    'nitrate': 'Nitrate',
    'conductance': 'Conductance',
}


def _is_null(value) -> bool:
    """Check if value is null/missing in a type-safe way.

    pd.isna() and np.isnan() fail on non-numeric types (strings, objects).
    This helper catches those exceptions and returns False for non-null objects.
    """
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (ValueError, TypeError):
        return False


def _score_status(score: float) -> str:
    """Return quality status based on parameter score (0-100)."""
    if score >= 90:
        return 'Excellent'
    elif score >= 70:
        return 'Good'
    elif score >= 50:
        return 'Fair'
    elif score >= 25:
        return 'Poor'
    else:
        return 'Very Poor'


def format_results_for_clipboard(
    # Required
    zip_code: str,
    wqi: float,
    classification: str,
    scores: Dict[str, float],
    aggregated: Dict[str, float],
    measurement_count: int,
    station_count: int,
    # Optional - omit sections when None
    location_info: Optional[Dict[str, Any]] = None,
    coords: Optional[Tuple[float, float]] = None,
    radius_miles: Optional[float] = None,
    source_label: Optional[str] = None,
    ml_predictions: Optional[Dict[str, Any]] = None,
    clf_importance_df: Optional[pd.DataFrame] = None,
    reg_importance_df: Optional[pd.DataFrame] = None,
    clf_contributions: Optional[Dict[str, Any]] = None,
    reg_contributions: Optional[Dict[str, Any]] = None,
    daily_wqi: Optional[pd.DataFrame] = None,
) -> str:
    """
    Format all water quality results into a text report for clipboard export.

    Args:
        zip_code: User-entered ZIP code
        wqi: Calculated Water Quality Index (0-100)
        classification: WQI classification (Excellent/Good/Fair/Poor/Very Poor)
        scores: Dict of parameter name -> score (0-100)
        aggregated: Dict of parameter name -> measured value
        measurement_count: Total number of measurements
        station_count: Number of monitoring stations
        location_info: Optional dict with place_name, state_name
        coords: Optional (lat, lon) tuple
        radius_miles: Optional search radius in miles
        source_label: Optional data source label
        ml_predictions: Optional dict with ML prediction results
        clf_importance_df: Optional DataFrame with classifier feature importance
        reg_importance_df: Optional DataFrame with regressor feature importance
        clf_contributions: Optional dict with classifier SHAP contributions
        reg_contributions: Optional dict with regressor SHAP contributions
        daily_wqi: Optional DataFrame with Date and WQI columns

    Returns:
        Formatted text string suitable for pasting into a chatbot
    """
    lines: List[str] = []

    # Header
    lines.append('WATER QUALITY DIAGNOSTIC REPORT')
    lines.append('=' * 31)
    lines.append('WARNING: Contains location data. Share only with trusted parties.')
    lines.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append('')

    # Query Parameters
    lines.append('QUERY PARAMETERS')
    lines.append('-' * 16)
    lines.append(f'ZIP Code: {zip_code}')
    if radius_miles is not None:
        lines.append(f'Search Radius: {radius_miles} miles')
    if source_label:
        lines.append(f'Data Source: {source_label}')
    lines.append(f'Measurements: {measurement_count}')
    lines.append(f'Stations: {station_count}')
    lines.append('')

    # Location
    if location_info or coords:
        lines.append('LOCATION')
        lines.append('-' * 8)
        if location_info:
            place = location_info.get('place_name', 'Unknown')
            state = location_info.get('state_name', '')
            if state:
                lines.append(f'Place: {place}, {state}')
            else:
                lines.append(f'Place: {place}')
        if coords:
            lines.append(f'Coordinates: {coords[0]:.6f}, {coords[1]:.6f}')
        lines.append('')

    # Parameter Values
    lines.append('PARAMETER VALUES')
    lines.append('-' * 16)
    for param_key, display_name in PARAM_DISPLAY_NAMES.items():
        if param_key in aggregated:
            value = aggregated[param_key]
            unit = PARAM_UNITS.get(param_key, '')
            if unit:
                lines.append(f'{display_name}: {value:.2f} {unit}')
            else:
                lines.append(f'{display_name}: {value:.2f}')
        else:
            lines.append(f'{display_name}: N/A')
    lines.append('')

    # Parameter Scores
    lines.append('PARAMETER SCORES')
    lines.append('-' * 16)
    for param_key, display_name in PARAM_DISPLAY_NAMES.items():
        if param_key in scores and not _is_null(scores[param_key]):
            score = scores[param_key]
            status = _score_status(score)
            lines.append(f'{display_name}: {score:.1f} ({status})')
        else:
            lines.append(f'{display_name}: N/A')
    lines.append('')

    # Calculated WQI
    lines.append('CALCULATED WQI')
    lines.append('-' * 14)
    lines.append(f'Score: {wqi:.2f}')
    lines.append(f'Classification: {classification}')
    lines.append('')

    # ML Predictions
    if ml_predictions:
        lines.append('ML PREDICTIONS')
        lines.append('-' * 14)

        if 'predicted_wqi' in ml_predictions:
            pred_wqi = ml_predictions['predicted_wqi']
            pred_class = ml_predictions.get('predicted_classification', 'Unknown')
            lines.append(f'Predicted WQI: {pred_wqi:.2f}')
            lines.append(f'Predicted Classification: {pred_class}')

        if 'classifier_probability' in ml_predictions:
            prob = ml_predictions['classifier_probability']
            lines.append(f'Safety Probability: {prob:.1%}')

        if 'classifier_confidence' in ml_predictions:
            conf = ml_predictions['classifier_confidence']
            lines.append(f'Classifier Confidence: {conf:.1%}')

        if 'regressor_confidence' in ml_predictions:
            reg_conf = ml_predictions['regressor_confidence']
            lines.append(f'Regressor Confidence: {reg_conf:.1%}')

        # Threshold margins if available
        if 'margin_to_unsafe' in ml_predictions:
            margin = ml_predictions['margin_to_unsafe']
            lines.append(f'Margin to Unsafe: {margin:.2f} points')

        if 'imputed_count' in ml_predictions:
            imputed = ml_predictions['imputed_count']
            total = ml_predictions.get('total_features', 59)
            lines.append(f'Imputed Features: {imputed}/{total}')

        lines.append('')

    # Feature Importance - Classifier
    if clf_importance_df is not None and not clf_importance_df.empty:
        lines.append('FEATURE IMPORTANCE - CLASSIFIER')
        lines.append('-' * 31)
        top_n = min(20, len(clf_importance_df))
        for _, row in clf_importance_df.head(top_n).iterrows():
            rank = int(row['rank'])
            feature = row['feature']
            importance = row['importance_pct']
            availability = row.get('availability', 'Unknown')
            lines.append(f'{rank:2d}. {feature}: {importance:.2f}% [{availability}]')
        lines.append('')

    # Feature Importance - Regressor
    if reg_importance_df is not None and not reg_importance_df.empty:
        lines.append('FEATURE IMPORTANCE - REGRESSOR')
        lines.append('-' * 30)
        top_n = min(20, len(reg_importance_df))
        for _, row in reg_importance_df.head(top_n).iterrows():
            rank = int(row['rank'])
            feature = row['feature']
            importance = row['importance_pct']
            availability = row.get('availability', 'Unknown')
            lines.append(f'{rank:2d}. {feature}: {importance:.2f}% [{availability}]')
        lines.append('')

    # SHAP - Classifier
    if clf_contributions is not None:
        lines.append('SHAP CONTRIBUTIONS - CLASSIFIER')
        lines.append('-' * 31)
        base_val = clf_contributions.get("base_value")
        pred_val = clf_contributions.get("prediction")
        shap_sum = clf_contributions.get("shap_sum")
        if base_val is not None and not _is_null(base_val):
            lines.append(f'Base Value: {base_val:.4f}')
        if pred_val is not None and not _is_null(pred_val):
            lines.append(f'Prediction: {pred_val:.4f}')
        if shap_sum is not None and not _is_null(shap_sum):
            lines.append(f'SHAP Sum: {shap_sum:.4f}')
        lines.append('')

        contrib_df = clf_contributions.get('contributions')
        if contrib_df is not None and not contrib_df.empty:
            top_n = min(20, len(contrib_df))
            for _, row in contrib_df.head(top_n).iterrows():
                feature = row['feature']
                value = row['value']
                contribution = row['contribution']
                if _is_null(contribution):
                    continue
                sign = '+' if contribution >= 0 else ''
                if _is_null(value):
                    imputed = row.get('imputed_value')
                    if imputed is not None and not _is_null(imputed):
                        lines.append(f'  {feature}: N/A (imputed: {imputed:.2f}) -> {sign}{contribution:.4f}')
                    else:
                        lines.append(f'  {feature}: N/A -> {sign}{contribution:.4f}')
                else:
                    lines.append(f'  {feature}: {value:.4f} -> {sign}{contribution:.4f}')
        lines.append('')

    # SHAP - Regressor
    if reg_contributions is not None:
        lines.append('SHAP CONTRIBUTIONS - REGRESSOR')
        lines.append('-' * 30)
        base_val = reg_contributions.get("base_value")
        pred_val = reg_contributions.get("prediction")
        shap_sum = reg_contributions.get("shap_sum")
        if base_val is not None and not _is_null(base_val):
            lines.append(f'Base Value: {base_val:.2f}')
        if pred_val is not None and not _is_null(pred_val):
            lines.append(f'Prediction: {pred_val:.2f}')
        if shap_sum is not None and not _is_null(shap_sum):
            lines.append(f'SHAP Sum: {shap_sum:.2f}')
        lines.append('')

        contrib_df = reg_contributions.get('contributions')
        if contrib_df is not None and not contrib_df.empty:
            top_n = min(20, len(contrib_df))
            for _, row in contrib_df.head(top_n).iterrows():
                feature = row['feature']
                value = row['value']
                contribution = row['contribution']
                if _is_null(contribution):
                    continue
                sign = '+' if contribution >= 0 else ''
                if _is_null(value):
                    imputed = row.get('imputed_value')
                    if imputed is not None and not _is_null(imputed):
                        lines.append(f'  {feature}: N/A (imputed: {imputed:.2f}) -> {sign}{contribution:.4f}')
                    else:
                        lines.append(f'  {feature}: N/A -> {sign}{contribution:.4f}')
                else:
                    lines.append(f'  {feature}: {value:.4f} -> {sign}{contribution:.4f}')
        lines.append('')

    # Daily WQI (most recent 30 measurements)
    if daily_wqi is not None and not daily_wqi.empty:
        lines.append('DAILY WQI (most recent 30 measurements)')
        lines.append('-' * 24)

        # Sort by date descending, take most recent 30 measurements
        sorted_daily = daily_wqi.sort_values('Date', ascending=False).head(30)

        for _, row in sorted_daily.iterrows():
            date = row['Date']
            day_wqi = row['WQI']
            day_class = WQICalculator.classify_wqi(day_wqi)

            if isinstance(date, pd.Timestamp):
                date_str = date.strftime('%Y-%m-%d')
            else:
                date_str = str(date)

            lines.append(f'{date_str}: {day_wqi:.2f} ({day_class})')
        lines.append('')

    # Warnings
    lines.append('WARNINGS')
    lines.append('-' * 8)
    lines.append('This analysis CANNOT detect:')
    lines.append('  - Lead and heavy metals')
    lines.append('  - Bacteria (E. coli, coliform)')
    lines.append('  - Pesticides and herbicides')
    lines.append('  - PFAS (forever chemicals)')
    lines.append('  - Pharmaceuticals')
    lines.append('')
    lines.append('For comprehensive water safety, laboratory testing is required.')

    return '\n'.join(lines)
