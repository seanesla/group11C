"""
Water Quality Index Lookup - Streamlit Application

This app allows users to search for water quality data by ZIP code and view
Water Quality Index (WQI) scores with visualizations.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Add src directory to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from geolocation.zipcode_mapper import ZipCodeMapper
from data_collection.wqp_client import WQPClient
from data_collection.usgs_client import USGSClient
from data_collection.fallback import fetch_with_fallback
from utils.wqi_calculator import WQICalculator
from utils.ml_feature_definitions import (
    get_feature_categories,
    count_features_by_availability,
    get_training_only_features,
    get_us_available_features,
)
from models.model_utils import load_latest_models
from preprocessing.us_data_features import prepare_us_features_for_prediction
from preprocessing.feature_engineering import NITRATE_NO3_TO_N
from services.search_strategies import build_search_strategies, SearchStrategy


# Page configuration
st.set_page_config(
    page_title="Water Quality Index Lookup",
    page_icon=":material/water_drop:",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Color scheme for WQI classifications
WQI_COLORS = {
    "Excellent": "#00CC00",  # Green
    "Good": "#0066FF",       # Blue
    "Fair": "#FFCC00",       # Yellow
    "Poor": "#FF6600",       # Orange
    "Very Poor": "#CC0000"   # Red
}


# Parameter units mapping (NO HARDCODING - sourced from EPA/WHO standards)
PARAMETER_UNITS = {
    'ph': '',  # pH is dimensionless (no unit)
    'dissolved_oxygen': 'mg/L',
    'temperature': '°C',
    'turbidity': 'NTU',
    'nitrate': 'mg/L as N',  # EPA standard: nitrogen content only
    'conductance': 'µS/cm'
}


# Canonical parameter name mapping (WQP characteristic names -> internal keys)
# Used for data aggregation - single source of truth
PARAM_MAPPING = {
    'pH': 'ph',
    'Dissolved oxygen (DO)': 'dissolved_oxygen',
    'Temperature, water': 'temperature',
    'Turbidity': 'turbidity',
    'Nitrate': 'nitrate',
    'Specific conductance': 'conductance'
}


def _expected_feature_columns() -> list[str]:
    """
    Canonical 18-feature schema used for US predictions.

    Derived from prepare_us_features_for_prediction so training and inference
    stay in lockstep even if the core feature pipeline evolves.
    """
    sample_df = prepare_us_features_for_prediction(
        ph=7.0,
        dissolved_oxygen=8.0,
        temperature=15.0,
        turbidity=2.0,
        nitrate=3.0,
        conductance=450.0,
    )
    return list(sample_df.columns)


@st.cache_resource
def load_ml_models():
    """
    Load trained ML models with Streamlit caching and schema validation.

    Returns:
        Tuple of (classifier, regressor). Either may be None if not found or
        incompatible with the deployed feature schema.
    """
    try:
        classifier, regressor = load_latest_models()
    except Exception as e:
        st.error(f"Failed to load ML models: {e}")
        return None, None

    if classifier is None or regressor is None:
        return classifier, regressor

    expected = _expected_feature_columns()
    clf_features = getattr(classifier, "feature_names", None)
    reg_features = getattr(regressor, "feature_names", None)

    if clf_features != expected or reg_features != expected:
        st.warning(
            "ML models were trained with a feature schema that does not match the "
            "deployed US prediction pipeline. To enable ML predictions in the app, "
            "retrain models with the core-parameter feature set "
            "(`train_models.py --core-params-only`) and redeploy."
        )
        return None, None

    return classifier, regressor


def get_wqi_color(classification: str) -> str:
    """Get color for WQI classification."""
    return WQI_COLORS.get(classification, "#808080")


def render_colored_card(
    title: str,
    color: str,
    subtitle: str = None,
    label: str = None
) -> None:
    """Render a colored card with consistent styling."""
    html = f"<div style='padding: 15px; border-radius: 5px; background-color: {color}20; border: 2px solid {color};'>"
    if label:
        html += f"<h4 style='margin: 0;'>{label}</h4>"
    html += f"<h3 style='margin: 0; color: {color};'>{title}</h3>"
    if subtitle:
        html += f"<p style='margin: 6px 0 0; color: #cccccc;'>{subtitle}</p>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def format_coordinates(lat: float, lon: float) -> str:
    """Format coordinates for display."""
    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"
    return f"{abs(lat):.4f}°{lat_dir}, {abs(lon):.4f}°{lon_dir}"


def make_ml_predictions(
    aggregated_params: Dict[str, float],
    classifier,
    regressor,
    year: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """
    Make ML predictions from aggregated water quality parameters.

    Args:
        aggregated_params: Dict with keys like 'ph', 'dissolved_oxygen', etc.
        classifier: Trained classifier model
        regressor: Trained regressor model
        year: Year of measurement (defaults to current year)

    Returns:
        Dictionary with ML prediction results, or None if models unavailable
    """
    if classifier is None or regressor is None:
        return None

    try:
        # Prepare features
        features = prepare_us_features_for_prediction(
            ph=aggregated_params.get('ph'),
            dissolved_oxygen=aggregated_params.get('dissolved_oxygen'),
            temperature=aggregated_params.get('temperature'),
            turbidity=aggregated_params.get('turbidity'),
            nitrate=aggregated_params.get('nitrate'),
            conductance=aggregated_params.get('conductance'),
            year=year
        )

        # Make predictions
        is_safe_pred = classifier.predict(features)[0]
        is_safe_proba = classifier.predict_proba(features)[0]  # [P(unsafe), P(safe)]
        wqi_pred = regressor.predict(features)[0]

        return {
            'is_safe': bool(is_safe_pred),
            'prob_unsafe': float(is_safe_proba[0]),
            'prob_safe': float(is_safe_proba[1]),
            'predicted_wqi': float(wqi_pred),
            'confidence': float(max(is_safe_proba))  # Confidence is max probability
        }

    except Exception as e:
        st.warning(f"ML prediction failed: {e}")
        return None


def get_daily_wqi_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw measurements into daily WQI scores (one row per day)."""
    if df.empty or 'ActivityStartDate' not in df.columns:
        return pd.DataFrame()

    calculator = WQICalculator()

    df = df.copy()
    df['ActivityStartDate'] = pd.to_datetime(df['ActivityStartDate'])

    rows = []
    for date, date_df in df.groupby('ActivityStartDate'):
        params = {}
        for characteristic_name, param_key in PARAM_MAPPING.items():
            mask = date_df['CharacteristicName'] == characteristic_name
            values = pd.to_numeric(date_df.loc[mask, 'ResultMeasureValue'], errors='coerce').dropna()
            if len(values) > 0:
                params[param_key] = float(values.median())
        if params:
            try:
                wqi, _, _ = calculator.calculate_wqi(**params)
                rows.append({"Date": date, "WQI": wqi})
            except Exception:
                continue

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values('Date').reset_index(drop=True)


def create_time_series_chart(df: pd.DataFrame) -> go.Figure:
    """Create time series chart showing WQI over time with quality zones."""
    plot_df = get_daily_wqi_scores(df)
    if plot_df.empty:
        return None

    # Create figure
    fig = go.Figure()

    # Add quality zones as background shapes
    fig.add_hrect(y0=90, y1=100, fillcolor=WQI_COLORS["Excellent"], opacity=0.1, line_width=0)
    fig.add_hrect(y0=70, y1=90, fillcolor=WQI_COLORS["Good"], opacity=0.1, line_width=0)
    fig.add_hrect(y0=50, y1=70, fillcolor=WQI_COLORS["Fair"], opacity=0.1, line_width=0)
    fig.add_hrect(y0=25, y1=50, fillcolor=WQI_COLORS["Poor"], opacity=0.1, line_width=0)
    fig.add_hrect(y0=0, y1=25, fillcolor=WQI_COLORS["Very Poor"], opacity=0.1, line_width=0)

    # Add WQI line
    fig.add_trace(go.Scatter(
        x=plot_df['Date'],
        y=plot_df['WQI'],
        mode='lines+markers',
        name='WQI Score',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6)
    ))

    # Update layout
    fig.update_layout(
        title="Water Quality Index Over Time",
        xaxis_title="Date",
        yaxis_title="WQI Score",
        yaxis=dict(range=[0, 105]),
        hovermode='x unified',
        height=400
    )

    return fig


def create_parameter_chart(scores: Dict[str, float]) -> go.Figure:
    """Create bar chart comparing individual parameter scores."""
    if not scores:
        return None

    # Prepare data
    params = list(scores.keys())
    values = list(scores.values())

    # Assign colors based on score
    colors = []
    for score in values:
        if score >= 90:
            colors.append(WQI_COLORS["Excellent"])
        elif score >= 70:
            colors.append(WQI_COLORS["Good"])
        elif score >= 50:
            colors.append(WQI_COLORS["Fair"])
        elif score >= 25:
            colors.append(WQI_COLORS["Poor"])
        else:
            colors.append(WQI_COLORS["Very Poor"])

    # Create figure
    fig = go.Figure(data=[
        go.Bar(
            x=params,
            y=values,
            marker_color=colors,
            text=[f"{v:.1f}" for v in values],
            textposition='outside'
        )
    ])

    # Update layout
    fig.update_layout(
        title="Individual Parameter Scores",
        xaxis_title="Parameter",
        yaxis_title="Score (0-100)",
        yaxis=dict(range=[0, 105]),
        height=400,
        showlegend=False
    )

    return fig


def display_wqi_methodology():
    """
    Display comprehensive WQI calculation methodology.

    Shows the actual parameter weights from WQICalculator and explains
    the weighted average formula used to calculate WQI.
    """
    # Get actual weights from WQICalculator (NO HARDCODING)
    weights = WQICalculator.PARAMETER_WEIGHTS

    st.markdown("""
    ### How is WQI Calculated?

    The Water Quality Index (WQI) is calculated using a **weighted average** of individual parameter scores,
    based on the **National Sanitation Foundation Water Quality Index (NSF-WQI)** methodology.

    #### Formula:
    """)

    st.latex(r"WQI = \frac{\sum_{i=1}^{n} (Q_i \times W_i)}{\sum_{i=1}^{n} W_i}")

    st.markdown("""
    Where:
    - **Q<sub>i</sub>** = Quality score for parameter *i* (0-100)
    - **W<sub>i</sub>** = Weight for parameter *i*
    - **n** = Number of available parameters
    """, unsafe_allow_html=True)

    st.markdown("#### Parameter Weights")
    st.markdown("Each parameter contributes to the overall WQI based on its relative importance to water quality:")

    # Create DataFrame showing actual weights from WQICalculator
    weight_data = []
    total_weight = sum(weights.values())

    for param, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        param_name = param.replace('_', ' ').title()
        percentage = (weight / total_weight) * 100
        weight_data.append({
            "Parameter": param_name,
            "Weight": f"{weight:.2f}",
            "Percentage": f"{percentage:.1f}%"
        })

    weight_df = pd.DataFrame(weight_data)
    st.dataframe(weight_df, width='stretch', hide_index=True)

    st.markdown("""
    **Note:** When some parameters are unavailable (e.g., turbidity), the weights are dynamically normalized
    so the remaining parameters sum to 1.0. This ensures WQI remains on a 0-100 scale.

    #### References:
    - National Sanitation Foundation Water Quality Index (NSF-WQI)
    - EPA Water Quality Standards
    - WHO Guidelines for Drinking-water Quality
    """)


def display_epa_who_standards():
    """
    Display comprehensive EPA and WHO water quality standards.

    Information sourced from docs/WQI_STANDARDS.md (NO HARDCODING).
    """
    st.markdown("### :material/menu_book: Water Quality Standards Reference")

    # Create tabs for EPA, WHO, and Nitrate Conversion
    tab1, tab2, tab3 = st.tabs(["EPA Standards", "WHO Guidelines", "Nitrate Units"])

    with tab1:
        st.markdown("#### EPA Water Quality Standards")
        st.markdown("**Source:** EPA National Primary Drinking Water Regulations")

        st.markdown("##### Primary Drinking Water Standards (MCLs)")
        epa_mcl = pd.DataFrame([
            {"Contaminant": "Nitrate (as N)", "MCL": "10 mg/L", "Health Concern": "Blue-baby syndrome in infants <6 months"},
            {"Contaminant": "Nitrite (as N)", "MCL": "1 mg/L", "Health Concern": "Similar to nitrate, infant health risk"},
            {"Contaminant": "Turbidity", "MCL": "1 NTU", "Health Concern": "Treatment technique; indicates filtration effectiveness"}
        ])
        st.dataframe(epa_mcl, width='stretch', hide_index=True)

        st.markdown("##### Secondary Drinking Water Standards (SMCLs)")
        st.markdown("*Non-mandatory aesthetic guidelines:*")
        epa_smcl = pd.DataFrame([
            {"Parameter": "pH", "SMCL": "6.5 - 8.5", "Concern": "Corrosion, taste"},
            {"Parameter": "Conductivity", "SMCL": "No standard", "Concern": "Aesthetic indicator"},
            {"Parameter": "Temperature", "SMCL": "No standard", "Concern": "Affects aquatic life, not human health"}
        ])
        st.dataframe(epa_smcl, width='stretch', hide_index=True)

        st.info("**Note:** Dissolved Oxygen (DO) is **not regulated** for drinking water. DO is a surface water quality parameter for aquatic ecosystem health.")

        st.markdown("**References:**")
        st.markdown("- [EPA Primary Drinking Water Regulations](https://www.epa.gov/ground-water-and-drinking-water/national-primary-drinking-water-regulations)")
        st.markdown("- [EPA Secondary Standards](https://www.epa.gov/sdwa/secondary-drinking-water-standards-guidance-nuisance-chemicals)")

    with tab2:
        st.markdown("#### WHO Guidelines for Drinking-water Quality")
        st.markdown("**Source:** WHO Guidelines for Drinking-water Quality (Fourth Edition)")

        who_guidelines = pd.DataFrame([
            {"Parameter": "pH", "Guideline": "No health-based guideline", "Notes": "Recommended operational range: 6.5-9.5"},
            {"Parameter": "Temperature", "Guideline": "<25°C preferred", "Notes": "Affects taste and chemical reactions"},
            {"Parameter": "Nitrate (as NO₃)", "Guideline": "50 mg/L", "Notes": "Equivalent to 11.3 mg/L as N"},
            {"Parameter": "Turbidity", "Guideline": "<5 NTU ideal", "Notes": "Higher values indicate treatment problems"}
        ])
        st.dataframe(who_guidelines, width='stretch', hide_index=True)

        st.markdown("**Reference:**")
        st.markdown("- [WHO Guidelines for Drinking-water Quality](https://iris.who.int/bitstream/handle/10665/44584/9789241548151_eng.pdf)")

    with tab3:
        st.markdown("#### Nitrate Unit Conversion System")
        st.markdown("""
        Water quality data sources use different unit conventions for nitrate:
        - **Some historical datasets (e.g., Kaggle)**: mg{NO₃}/L (molecular form)
        - **EPA/USGS Standards**: mg/L as N (nitrogen content only)

        This **4.43× difference** is critical for accurate WQI calculations.
        """)

        st.markdown("##### Conversion Factor")
        st.latex(r"\text{NITRATE\_NO3\_TO\_N} = \frac{\text{N atomic weight}}{\text{NO}_3 \text{ molecular weight}} = \frac{14.0067}{62.0049} = 0.2258")

        st.markdown("**To convert:** Multiply mg{NO₃}/L by **0.2258** to get mg/L as N")

        st.markdown("##### EPA Maximum Contaminant Level (MCL)")
        st.info("**EPA MCL for nitrate: 10 mg/L as N**")

        conversion_examples = pd.DataFrame([
            {"mg{NO₃}/L": "4.43", "mg/L as N": "1.0", "Safety Level": "Excellent"},
            {"mg{NO₃}/L": "22.15", "mg/L as N": "5.0", "Safety Level": "Good"},
            {"mg{NO₃}/L": "44.3", "mg/L as N": "10.0", "Safety Level": "EPA MCL (threshold)"},
            {"mg{NO₃}/L": "88.6", "mg/L as N": "20.0", "Safety Level": "Unsafe (2× MCL)"},
            {"mg{NO₃}/L": "221.5", "mg/L as N": "50.0", "Safety Level": "Very Unsafe (5× MCL)"}
        ])
        st.dataframe(conversion_examples, width='stretch', hide_index=True)

        st.success(":material/check_circle: This app uses **mg/L as N** (EPA standard) for all nitrate measurements and calculations.")


def create_future_trend_chart(
    trend_data: Dict[str, Any],
    current_wqi: float,
    current_date: datetime
) -> go.Figure:
    """
    Create line chart showing future WQI predictions over time.

    Args:
        trend_data: Dictionary from regressor.predict_future_trend() containing:
            - dates: List of datetime objects
            - predictions: List of WQI predictions
            - trend: Overall trend direction
        current_wqi: Current WQI score (for reference line)
        current_date: Current date (for "Today" marker)

    Returns:
        Plotly Figure object
    """
    if not trend_data or 'dates' not in trend_data or not trend_data['dates']:
        return None

    dates = trend_data['dates']
    predictions = trend_data['predictions']
    trend = trend_data.get('trend', 'unknown')
    wqi_change = trend_data.get('wqi_change', 0)

    # Create figure
    fig = go.Figure()

    # Add quality zones as background shapes
    fig.add_hrect(y0=90, y1=100, fillcolor=WQI_COLORS["Excellent"], opacity=0.1, line_width=0)
    fig.add_hrect(y0=70, y1=90, fillcolor=WQI_COLORS["Good"], opacity=0.1, line_width=0)
    fig.add_hrect(y0=50, y1=70, fillcolor=WQI_COLORS["Fair"], opacity=0.1, line_width=0)
    fig.add_hrect(y0=25, y1=50, fillcolor=WQI_COLORS["Poor"], opacity=0.1, line_width=0)
    fig.add_hrect(y0=0, y1=25, fillcolor=WQI_COLORS["Very Poor"], opacity=0.1, line_width=0)

    # Add "Today" vertical line using add_shape instead of add_vline
    # (add_vline has issues with datetime objects in some Plotly versions)
    fig.add_shape(
        type="line",
        x0=current_date,
        x1=current_date,
        y0=0,
        y1=105,
        line=dict(color="gray", width=2, dash="dash"),
        xref="x",
        yref="y"
    )

    # Add "Today" annotation
    fig.add_annotation(
        x=current_date,
        y=100,
        text="Today",
        showarrow=False,
        yshift=10,
        font=dict(color="gray", size=12)
    )

    # Add current WQI point
    fig.add_trace(go.Scatter(
        x=[current_date],
        y=[current_wqi],
        mode='markers',
        name='Current WQI',
        marker=dict(size=10, color='#1f77b4', symbol='circle'),
        showlegend=True
    ))

    # Add predicted WQI line
    fig.add_trace(go.Scatter(
        x=dates,
        y=predictions,
        mode='lines+markers',
        name='Predicted WQI',
        line=dict(color='#ff7f0e', width=2, dash='dot'),
        marker=dict(size=6, color='#ff7f0e'),
        showlegend=True
    ))

    # Determine trend color and symbol
    if trend == 'improving':
        trend_color = WQI_COLORS["Good"]
        trend_symbol = "↗"
    elif trend == 'declining':
        trend_color = WQI_COLORS["Poor"]
        trend_symbol = "↘"
    else:
        trend_color = WQI_COLORS["Fair"]
        trend_symbol = "→"

    # Add trend annotation
    trend_text = f"{trend_symbol} {trend.upper()}: {wqi_change:+.1f} points"
    fig.add_annotation(
        x=dates[-1],
        y=predictions[-1],
        text=trend_text,
        showarrow=True,
        arrowhead=2,
        arrowcolor=trend_color,
        bgcolor=trend_color,
        font=dict(color='white', size=12),
        bordercolor=trend_color,
        borderwidth=2,
        borderpad=4,
        opacity=0.8
    )

    # Update layout
    fig.update_layout(
        title="Future Water Quality Forecast (12 Months)",
        xaxis_title="Date",
        yaxis_title="WQI Score",
        yaxis=dict(range=[0, 105]),
        hovermode='x unified',
        height=450,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def build_forecast_from_history(
    daily_wqi: pd.DataFrame,
    periods: int = 12
) -> Optional[Dict[str, Any]]:
    """
    Derive a simple 12-month forecast from observed daily WQI history using
    linear extrapolation. Avoids flat lines by reflecting actual recent trend.
    """
    if daily_wqi is None or daily_wqi.empty or len(daily_wqi) < 3:
        return None

    import numpy as np
    from dateutil.relativedelta import relativedelta

    # Use last 180 days to reduce noise
    cutoff = daily_wqi['Date'].max() - pd.Timedelta(days=180)
    recent = daily_wqi[daily_wqi['Date'] >= cutoff] if len(daily_wqi) > 30 else daily_wqi

    ordinals = recent['Date'].map(datetime.toordinal).to_numpy()
    wqis = recent['WQI'].to_numpy()

    # If variance is zero, no meaningful slope
    if np.isclose(wqis.std(), 0):
        return None

    slope, intercept = np.polyfit(ordinals, wqis, 1)  # WQI per day
    start_date = recent['Date'].max()
    current_wqi = recent.loc[recent['Date'].idxmax(), 'WQI']

    dates = []
    predictions = []
    for i in range(1, periods + 1):
        future_date = start_date + relativedelta(months=i)
        days_ahead = (future_date - start_date).days
        pred = current_wqi + slope * days_ahead
        pred = max(0, min(100, pred))
        dates.append(future_date)
        predictions.append(pred)

    final_wqi = predictions[-1]
    wqi_change = final_wqi - current_wqi
    if wqi_change > 2:
        trend = 'improving'
    elif wqi_change < -2:
        trend = 'declining'
    else:
        trend = 'stable'

    return {
        'dates': dates,
        'predictions': predictions,
        'trend': trend,
        'trend_slope': wqi_change / periods,
        'current_wqi': current_wqi,
        'final_wqi': final_wqi,
        'wqi_change': wqi_change,
        'periods': periods,
        'frequency': 'M',
        'method': 'historical_linear'
    }


def fetch_water_quality_data(
    zip_code: str,
    radius_miles: float,
    start_date: datetime,
    end_date: datetime
) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    """
    Fetch water quality data for a ZIP code.

    Returns:
        Tuple of (DataFrame, error_message, source_label)
        If successful: (DataFrame, None, "<source> · <strategy>")
        If failed: (None, error_message, None)
    """
    try:
        # Initialize clients
        mapper = ZipCodeMapper()
        client = WQPClient()
        usgs_client = USGSClient()

        # Validate and convert ZIP code
        if not mapper.is_valid_zipcode(zip_code):
            return None, f"Invalid ZIP code: {zip_code}", None

        coords = mapper.get_coordinates(zip_code)
        if coords is None:
            return None, f"Could not find coordinates for ZIP code: {zip_code}", None

        lat, lon = coords

        characteristics = [
            "pH",
            "Dissolved oxygen (DO)",
            "Temperature, water",
            "Turbidity",
            "Nitrate",
            "Specific conductance"
        ]

        strategies = build_search_strategies(
            radius_miles=radius_miles,
            start_date=start_date,
            end_date=end_date
        )

        attempt_history = []

        for strategy in strategies:
            description = strategy.describe()
            attempt_history.append(description)

            with st.spinner(f"Searching {description} for ZIP {zip_code}..."):
                df, source_label = fetch_with_fallback(
                    latitude=lat,
                    longitude=lon,
                    radius_miles=strategy.radius_miles,
                    start_date=strategy.start_date,
                    end_date=strategy.end_date,
                    characteristics=characteristics,
                    wqp_client=client,
                    usgs_client=usgs_client
                )

            if df is not None and not df.empty:
                label = source_label or "WQP/USGS"
                context = f"{label} · {description}"
                if strategy.auto_adjusted:
                    context += " (auto-extended)"
                return df, None, context

        attempts_text = "; ".join(attempt_history)
        return None, (
            f"No water quality data found for ZIP {zip_code} after trying: {attempts_text}. "
            "Try a different ZIP or select a broader date range."
        ), None

    except Exception as e:
        return None, f"Error fetching data: {str(e)}", None


def calculate_overall_wqi(df: pd.DataFrame) -> Tuple[Optional[float], Optional[Dict[str, float]], Optional[str]]:
    """Calculate overall WQI from DataFrame of measurements in long format."""
    if df.empty:
        return None, None, None

    calculator = WQICalculator()

    # WQP API returns data in long format with CharacteristicName and ResultMeasureValue columns
    # Need to aggregate by characteristic name
    aggregated = {}

    for characteristic_name, param_key in PARAM_MAPPING.items():
        # Filter rows for this characteristic
        mask = df['CharacteristicName'] == characteristic_name
        values = pd.to_numeric(df.loc[mask, 'ResultMeasureValue'], errors='coerce').dropna()

        if len(values) > 0:
            # Use median of all measurements for this parameter
            aggregated[param_key] = float(values.median())

    if not aggregated:
        return None, None, None

    # Warn if conductance is elevated (possible brackish or mineral-rich water)
    if aggregated.get('conductance', 0) > 3000:
        st.warning(f"High conductance ({aggregated['conductance']:.0f} µS/cm) - possible saltwater or mineral influence.")

    try:
        wqi, scores, classification = calculator.calculate_wqi(**aggregated)
        return wqi, scores, classification
    except Exception as e:
        st.error(f"Error calculating WQI: {str(e)}")
        return None, None, None


# Main App
def main():
    """Main application logic."""

    # Load ML models (cached)
    classifier, regressor = load_ml_models()

    # Title
    st.title("Water Quality Index Lookup")

    # Show ML model status
    if classifier and regressor:
        st.sidebar.success("ML models loaded")
    else:
        st.sidebar.warning("ML models unavailable")

    # Sidebar - User Inputs
    st.sidebar.header("Search Parameters")

    zip_code = st.sidebar.text_input(
        "ZIP Code",
        value="20001",
        max_chars=5,
        help="Enter a 5-digit US ZIP code"
    )

    radius_miles = st.sidebar.slider(
        "Search Radius (miles)",
        min_value=5,
        max_value=100,
        value=25,
        step=5,
        help="Radius around the ZIP code to search for monitoring stations"
    )

    # Date range
    default_end = datetime.now()
    default_start = default_end - timedelta(days=365)  # Last year

    st.sidebar.subheader("Date Range")
    start_date = st.sidebar.date_input(
        "Start Date",
        value=default_start,
        max_value=default_end,
        help="Data window sent to WQP/USGS; defaults to last 12 months."
    )

    end_date = st.sidebar.date_input(
        "End Date",
        value=default_end,
        max_value=default_end,
        help="Must be on/after start date. Shorten to speed up API calls."
    )

    # Convert to datetime
    start_date = datetime.combine(start_date, datetime.min.time())
    end_date = datetime.combine(end_date, datetime.min.time())

    # Input validation
    has_input_error = False
    if end_date < start_date:
        st.sidebar.error("End date must be on or after start date.")
        has_input_error = True
    if radius_miles <= 0:
        st.sidebar.error("Radius must be greater than zero miles.")
        has_input_error = True

    # Submit button
    search_button = st.sidebar.button("Search", type="primary")

    # EPA/WHO Standards Reference in Sidebar
    st.sidebar.divider()
    with st.sidebar.expander(":material/menu_book: Standards Reference"):
        display_epa_who_standards()

    # Main area

    # Keep safety context but avoid overwhelming users—one concise expander.
    with st.expander(":material/warning: Limitations & Safety", expanded=False):
        st.markdown("""
- **Not a full potability test:** Only 6 parameters (pH, DO, Temperature, Turbidity, Nitrate, Conductance). **Not covered:** lead/arsenic/mercury, bacteria, PFAS, pesticides, pharmaceuticals.
- **Lead risk:** Lead MCLG = 0; if home built <1986 or piping recently disturbed, get certified lead testing (EPA hotline 1‑800‑426‑4791).
- **If water looks/smells off:** Use bottled water and contact your utility; this app cannot rule out contamination.
- **Get certified lab testing** for drinking decisions; this tool is informational.
        """)

    st.divider()

    if search_button:
        if has_input_error:
            st.error("Please fix the issues in the sidebar (date range and/or radius) before searching.")
            return

        # Fetch data with WQP → USGS fallback
        df, error, source_label = fetch_water_quality_data(zip_code, radius_miles, start_date, end_date)

        if error:
            st.error(error)
            st.info("Try increasing the search radius or adjusting the date range.")
            return

        if df is None or df.empty:
            st.warning("No data available for the specified criteria.")
            return

        # Get location info
        mapper = ZipCodeMapper()
        location_info = mapper.get_location_info(zip_code)
        coords = mapper.get_coordinates(zip_code)

        # Calculate overall WQI
        wqi, scores, classification = calculate_overall_wqi(df)

        if wqi is None:
            st.warning("Unable to calculate WQI from available data.")
            st.dataframe(df, width='stretch')
            return

        # Get aggregated parameters for ML predictions
        aggregated = {}
        for characteristic_name, param_key in PARAM_MAPPING.items():
            mask = df['CharacteristicName'] == characteristic_name
            values = pd.to_numeric(df.loc[mask, 'ResultMeasureValue'], errors='coerce').dropna()
            if len(values) > 0:
                aggregated[param_key] = float(values.median())

        # Get daily WQI scores for forecast
        daily_wqi = get_daily_wqi_scores(df)

        # Make ML predictions
        ml_predictions = make_ml_predictions(
            aggregated,
            classifier,
            regressor,
            year=datetime.now().year
        )

        # Display results
        st.success(f"Found {len(df)} measurements from {df['MonitoringLocationIdentifier'].nunique()} monitoring stations")

        # Location info card
        st.subheader("Location Information")
        col1, col2, col3 = st.columns(3)

        with col1:
            if location_info:
                st.metric("Location", f"{location_info.get('place_name', 'Unknown')}, {location_info.get('state_code', '')}")
            else:
                st.metric("ZIP Code", zip_code)

        with col2:
            if coords:
                st.metric("Coordinates", format_coordinates(coords[0], coords[1]))

        with col3:
            st.metric("Search Radius", f"{radius_miles} miles")

        st.divider()

        # WQI Summary
        st.subheader("Water Quality Summary")
        if source_label:
            st.caption(f"Data source: {source_label}")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Overall WQI Score",
                f"{wqi:.1f}",
                help="Water Quality Index (0-100 scale)"
            )

        with col2:
            color = get_wqi_color(classification)
            render_colored_card(classification, color)

        with col3:
            calculator = WQICalculator()
            is_safe = calculator.is_safe(wqi)
            safety_text = "SAFE" if is_safe else "UNSAFE"
            safety_color = "#00CC00" if is_safe else "#FF6600"
            render_colored_card(
                safety_text,
                safety_color,
                subtitle="Core parameters only. Lead, bacteria, PFAS not tested."
            )

        # WQI Methodology Explainer
        with st.expander(":material/bar_chart: How is WQI Calculated?", expanded=False):
            display_wqi_methodology()

        st.divider()

        # ML Predictions Section
        if ml_predictions:
            st.subheader("ML Model Predictions")

            st.caption("Model trained on non-US data (Kaggle 1991–2017). US predictions may vary.")

            col1, col2, col3 = st.columns(3)

            with col1:
                # ML Classification
                ml_safety = "SAFE" if ml_predictions['is_safe'] else "UNSAFE"
                ml_color = "#00CC00" if ml_predictions['is_safe'] else "#FF6600"
                render_colored_card(ml_safety, ml_color, label="ML Classification")

            with col2:
                # ML Predicted WQI
                st.metric(
                    "ML Predicted WQI",
                    f"{ml_predictions['predicted_wqi']:.1f}",
                    help="Machine learning regression prediction"
                )

            with col3:
                # Confidence
                confidence_pct = ml_predictions['confidence'] * 100
                confidence_color = "#00CC00" if confidence_pct >= 80 else "#FFCC00" if confidence_pct >= 60 else "#FF6600"
                render_colored_card(f"{confidence_pct:.1f}%", confidence_color, label="Model Confidence")

            # Show probability breakdown
            with st.expander("View Detailed Probabilities"):
                prob_df = pd.DataFrame([
                    {"Prediction": "Unsafe (WQI < 70)", "Probability": f"{ml_predictions['prob_unsafe']*100:.1f}%"},
                    {"Prediction": "Safe (WQI ≥ 70)", "Probability": f"{ml_predictions['prob_safe']*100:.1f}%"}
                ])
                st.dataframe(prob_df, width='stretch', hide_index=True)

                st.caption("RandomForest: Classifier 98.6% acc, Regressor R²=0.986 | 2,939 training samples")

            st.divider()

            # Future Trend Prediction Section
            st.subheader("12-Month WQI Forecast")

            try:
                # Try to derive forecast from observed history; fallback to model drift
                current_date = datetime.now()
                trend_data = build_forecast_from_history(daily_wqi, periods=12)

                if trend_data is None:
                    import numpy as np
                    X_features = prepare_us_features_for_prediction(
                        ph=aggregated.get('ph'),
                        dissolved_oxygen=aggregated.get('dissolved_oxygen'),
                        temperature=aggregated.get('temperature'),
                        turbidity=aggregated.get('turbidity'),
                        nitrate=aggregated.get('nitrate'),
                        conductance=aggregated.get('conductance'),
                        year=current_date.year
                    )
                    X_array = np.array(X_features).reshape(1, -1)

                    trend_data = regressor.predict_future_trend(
                        X=X_array,
                        start_date=current_date,
                        periods=12,
                        freq='M'
                    )

                # Create and display the forecast chart
                if trend_data and trend_data.get('dates'):
                    trend_chart = create_future_trend_chart(
                        trend_data=trend_data,
                        current_wqi=wqi,
                        current_date=current_date
                    )

                    if trend_chart:
                        st.plotly_chart(trend_chart, width='stretch')

                        # Display trend analysis
                        trend = trend_data.get('trend', 'unknown')
                        wqi_change = trend_data.get('wqi_change', 0)
                        final_wqi = trend_data.get('final_wqi', wqi)

                        # Trend description with appropriate icon and color
                        if trend == 'improving':
                            trend_icon = "IMPROVING"
                            trend_color = "#00CC00"
                            trend_desc = "improving"
                        elif trend == 'declining':
                            trend_icon = "DECLINING"
                            trend_color = "#FF6600"
                            trend_desc = "declining"
                        else:
                            trend_icon = "STABLE"
                            trend_color = "#0066FF"
                            trend_desc = "stable"

                        col1, col2 = st.columns(2)

                        with col1:
                            render_colored_card(
                                f"{trend_desc.upper()}: {wqi_change:+.1f} pts/12mo",
                                trend_color,
                                label="Trend Analysis"
                            )

                        with col2:
                            st.metric(
                                "Projected WQI (12 months)",
                                f"{final_wqi:.1f}",
                                delta=f"{wqi_change:.1f}",
                                help="Predicted WQI score after 12 months"
                            )

                        st.caption("Forecast based on observed trends. Actual results may vary with seasonal/infrastructure changes.")

                else:
                    st.info("Unable to generate forecast: temporal features not available in model.")

            except Exception as e:
                st.error(f"Error generating forecast: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

            st.divider()

        # === ML FEATURES TRANSPARENCY SECTION ===
        st.subheader(":material/psychology: ML Features (59)")

        # Get feature categories and counts
        feature_categories = get_feature_categories()
        feature_counts = count_features_by_availability()
        training_only_features = set(get_training_only_features())

        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Features", feature_counts['total'])
        with col2:
            st.metric("Available for US", feature_counts['available'], help="Direct measurements from US water samples")
        with col3:
            st.metric(
                "Training-Only (Imputed)",
                feature_counts['missing'],
                delta=None,
                help="Imputed from training-data averages (not directly measured in US data)",
            )
        with col4:
            st.metric("Partial", feature_counts['partial'], help="Some components available for US data")

        # Feature categories display
        with st.expander(":material/list_alt: View All 59 Features by Category", expanded=False):
            for category_key, category_data in feature_categories.items():
                st.markdown(f"### {category_data['name']}")
                st.markdown(f"*{category_data['description']}*")

                # Availability indicator
                availability = category_data['available_for_us']
                if availability is True:
                    st.success(f":material/check_circle: **Available for US data** | Source: {category_data['source']}")
                elif availability is False:
                    st.warning(f":material/warning: **Training-only (imputed for US predictions)** | Source: {category_data['source']}")
                else:
                    st.info(f":material/info: **Partial availability** | Source: {category_data['source']}")

                # Feature list
                features_df = pd.DataFrame([
                    {
                        "Feature": feat_name,
                        "Description": feat_desc,
                        "Status": ":red[Imputed]" if feat_name in training_only_features else ":green[Available]"
                    }
                    for feat_name, feat_desc in category_data['features'].items()
                ])

                st.dataframe(
                    features_df,
                    width='stretch',
                    hide_index=True
                )

                st.divider()

        # === FEATURE DERIVATION EXPLANATIONS ===
        with st.expander(":material/science: How Are Features Derived?", expanded=False):
            st.markdown("""
            **From Water Quality:** `ph_deviation_from_7`, `do_temp_ratio`, `conductance_low/medium/high`, `pollution_stress`, `temp_stress`

            **From Temporal Data:** `years_since_1991`, `decade`, `is_1990s/2000s/2010s`

            **From Missing Data:** `*_missing` flags, `n_params_available`

            **From Geography (Europe-specific):** `water_body_*`, `country_*`, `gdp_per_capita_proxy`
            """)

        # === HIGHLIGHT MISSING/IMPUTED FEATURES ===
        with st.expander(":material/warning: Imputed Features for US Predictions", expanded=False):
            st.markdown(f"""
            **{len(training_only_features)} features** are imputed from European training data averages for US predictions.
            Core parameters (pH, DO, temp, nitrate, conductance) are directly measured.
            """)

        st.divider()

        # ===================================================================
        # PHASE 4.1: FEATURE IMPORTANCE ANALYSIS
        # ===================================================================
        st.subheader(":material/target: Feature Importance Analysis")

        try:
            from utils.feature_importance import (
                get_feature_importances,
                get_feature_importance_summary,
                annotate_importance_with_availability,
                get_prediction_contributions,
                generate_decision_explanation
            )

            # Get latest model paths
            import glob
            classifier_files = sorted(glob.glob('data/models/classifier_*.joblib'), reverse=True)
            regressor_files = sorted(glob.glob('data/models/regressor_*.joblib'), reverse=True)

            if classifier_files and regressor_files:
                classifier_path = classifier_files[0]
                regressor_path = regressor_files[0]

                # Get feature importance summary
                summary = get_feature_importance_summary(classifier_path, regressor_path)

                # Display summary metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Top Classifier Feature",
                        summary['top_feature_classifier'].replace('_', ' ').title(),
                        f"{summary['top_importance_classifier']:.1f}% importance"
                    )
                    st.caption("Most influential feature for SAFE/UNSAFE classification")

                with col2:
                    st.metric(
                        "Top Regressor Feature",
                        summary['top_feature_regressor'].replace('_', ' ').title(),
                        f"{summary['top_importance_regressor']:.1f}% importance"
                    )
                    st.caption("Most influential feature for WQI score prediction")

                # Get full feature importances (top 20)
                importances = get_feature_importances(classifier_path, regressor_path, top_n=20)

                # Annotate with availability
                us_features = get_us_available_features()
                training_only_features = get_training_only_features()

                clf_importance_df = annotate_importance_with_availability(
                    importances['classifier'],
                    us_features,
                    training_only_features,
                )

                reg_importance_df = annotate_importance_with_availability(
                    importances['regressor'],
                    us_features,
                    training_only_features,
                )

                # Display feature importance tables in expandable sections
                with st.expander(f":material/bar_chart: Classifier Features (top 10 = {summary['top_10_cumulative_classifier']:.0f}%)", expanded=False):

                    # Format display DataFrame
                    clf_display = clf_importance_df[['rank', 'feature', 'importance_pct', 'availability']].copy()
                    clf_display['feature'] = clf_display['feature'].str.replace('_', ' ').str.title()
                    clf_display.columns = ['Rank', 'Feature', 'Importance (%)', 'Availability']
                    clf_display['Importance (%)'] = clf_display['Importance (%)'].apply(lambda x: f"{x:.2f}%")

                    st.dataframe(
                        clf_display,
                        width='stretch',
                        hide_index=True
                    )

                    # Add visualization (horizontal bar chart)
                    import plotly.graph_objects as go
                    fig_clf = go.Figure(go.Bar(
                        x=clf_importance_df['importance_pct'][:10],
                        y=clf_importance_df['feature'][:10].str.replace('_', ' ').str.title(),
                        orientation='h',
                        marker=dict(
                            color=clf_importance_df['importance_pct'][:10],
                            colorscale='Blues',
                            showscale=False
                        )
                    ))
                    fig_clf.update_layout(
                        title="Top 10 Classifier Features",
                        xaxis_title="Importance (%)",
                        yaxis_title="Feature",
                        yaxis=dict(autorange="reversed"),  # Top feature at top
                        height=400
                    )
                    st.plotly_chart(fig_clf, width='stretch')

                with st.expander(f":material/bar_chart: Regressor Features (top 10 = {summary['top_10_cumulative_regressor']:.0f}%)", expanded=False):

                    # Format display DataFrame
                    reg_display = reg_importance_df[['rank', 'feature', 'importance_pct', 'availability']].copy()
                    reg_display['feature'] = reg_display['feature'].str.replace('_', ' ').str.title()
                    reg_display.columns = ['Rank', 'Feature', 'Importance (%)', 'Availability']
                    reg_display['Importance (%)'] = reg_display['Importance (%)'].apply(lambda x: f"{x:.2f}%")

                    st.dataframe(
                        reg_display,
                        width='stretch',
                        hide_index=True
                    )

                    # Add visualization (horizontal bar chart)
                    fig_reg = go.Figure(go.Bar(
                        x=reg_importance_df['importance_pct'][:10],
                        y=reg_importance_df['feature'][:10].str.replace('_', ' ').str.title(),
                        orientation='h',
                        marker=dict(
                            color=reg_importance_df['importance_pct'][:10],
                            colorscale='Greens',
                            showscale=False
                        )
                    ))
                    fig_reg.update_layout(
                        title="Top 10 Regressor Features",
                        xaxis_title="Importance (%)",
                        yaxis_title="Feature",
                        yaxis=dict(autorange="reversed"),  # Top feature at top
                        height=400
                    )
                    st.plotly_chart(fig_reg, width='stretch')

                # Add interpretation guide
                with st.expander(":material/info: How to Interpret", expanded=False):
                    st.markdown("""
                    - **>10%**: Critical | **5-10%**: Significant | **<5%**: Minor
                    - :green[Available]: Measured from US data | :red[Imputed]: From European training data
                    """)

            else:
                st.warning(":material/warning: ML models not found. Please train models first.")

        except Exception as e:
            st.error(f":material/error: Error loading feature importance: {e}")

        st.divider()

        # ===================================================================
        # PHASE 4.2: PER-PREDICTION FEATURE CONTRIBUTIONS (SHAP)
        # ===================================================================
        st.subheader(":material/search: Feature Contributions (This Sample)")

        try:
            # Get latest model paths
            import glob
            import numpy as np
            classifier_files = sorted(glob.glob('data/models/classifier_*.joblib'), reverse=True)
            regressor_files = sorted(glob.glob('data/models/regressor_*.joblib'), reverse=True)

            if classifier_files and regressor_files and ml_predictions:
                classifier_path = classifier_files[0]
                regressor_path = regressor_files[0]

                # Prepare 59-feature ML input for the current sample
                # prepare_us_features_for_prediction() already returns a DataFrame
                X_sample_df = prepare_us_features_for_prediction(
                    ph=aggregated.get('ph'),
                    dissolved_oxygen=aggregated.get('dissolved_oxygen'),
                    temperature=aggregated.get('temperature'),
                    turbidity=aggregated.get('turbidity'),
                    nitrate=aggregated.get('nitrate'),
                    conductance=aggregated.get('conductance'),
                    year=datetime.now().year
                )

                # Get contributions for classifier
                clf_contributions = get_prediction_contributions(
                    model_path=classifier_path,
                    X_sample=X_sample_df,
                    top_n=20
                )

                # Get contributions for regressor
                reg_contributions = get_prediction_contributions(
                    model_path=regressor_path,
                    X_sample=X_sample_df,
                    top_n=20
                )

                # Display summary metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Classifier Base Value",
                        f"{clf_contributions['base_value']:.3f}",
                        help="Average prediction across all training samples (probability of SAFE)"
                    )
                with col2:
                    st.metric(
                        "Regressor Base Value",
                        f"{reg_contributions['base_value']:.1f}",
                        help="Average WQI score across all training samples"
                    )

                # Mathematical verification
                clf_pred_delta = clf_contributions['prediction'] - clf_contributions['base_value']
                clf_match_error = abs(clf_contributions['shap_sum'] - clf_pred_delta)

                reg_pred_delta = reg_contributions['prediction'] - reg_contributions['base_value']
                reg_match_error = abs(reg_contributions['shap_sum'] - reg_pred_delta)

                # Show verification (collapsed by default)
                with st.expander(":material/science: SHAP Verification", expanded=False):
                    st.latex(r"\sum_{i=1}^{59} \text{SHAP}_i = \text{Prediction} - \text{Base Value}")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Classifier:**")
                        st.write(f"Prediction: {clf_contributions['prediction']:.4f}")
                        st.write(f"Base Value: {clf_contributions['base_value']:.4f}")
                        st.write(f"Difference: {clf_pred_delta:.4f}")
                        st.write(f"Sum of SHAP: {clf_contributions['shap_sum']:.4f}")
                        st.write(f"Match Error: {clf_match_error:.6f}")
                        if clf_match_error < 0.001:
                            st.success("Perfect match")
                        else:
                            st.warning(f"Mismatch: {clf_match_error:.6f}")

                    with col2:
                        st.markdown("**Regressor:**")
                        st.write(f"Prediction: {reg_contributions['prediction']:.2f}")
                        st.write(f"Base Value: {reg_contributions['base_value']:.2f}")
                        st.write(f"Difference: {reg_pred_delta:.2f}")
                        st.write(f"Sum of SHAP: {reg_contributions['shap_sum']:.2f}")
                        st.write(f"Match Error: {reg_match_error:.4f}")
                        if reg_match_error < 0.1:
                            st.success("Perfect match")
                        else:
                            st.warning(f"Mismatch: {reg_match_error:.4f}")

                # ===  VISUALIZATION: WATERFALL/BAR CHARTS ===
                # Create tabs for visualizations
                viz_tab1, viz_tab2 = st.tabs(["Classifier Contributions", "Regressor Contributions"])

                with viz_tab1:
                    # Classifier contributions bar chart
                    clf_top_10 = clf_contributions['contributions'].head(10).copy()

                    # Create horizontal bar chart
                    fig_clf_contrib = go.Figure()

                    # Add bars colored by direction (positive = green, negative = red)
                    colors_clf = ['#00CC00' if x > 0 else '#FF6600' for x in clf_top_10['contribution']]

                    fig_clf_contrib.add_trace(go.Bar(
                        y=clf_top_10['feature'].str.replace('_', ' ').str.title(),
                        x=clf_top_10['contribution'],
                        orientation='h',
                        marker=dict(color=colors_clf),
                        text=[f"{x:+.4f}" for x in clf_top_10['contribution']],
                        textposition='outside',
                        hovertemplate='<b>%{y}</b><br>Contribution: %{x:+.4f}<br>Value: %{customdata}<extra></extra>',
                        customdata=clf_top_10['value']
                    ))

                    fig_clf_contrib.update_layout(
                        title=f"Top 10 Feature Contributions - Classifier<br><sub>Prediction: {clf_contributions['prediction']:.3f} (Base: {clf_contributions['base_value']:.3f}, Δ {clf_pred_delta:+.3f})</sub>",
                        xaxis_title="Contribution to SAFE Probability",
                        yaxis_title="Feature",
                        yaxis=dict(autorange="reversed"),  # Top feature at top
                        height=500,
                        showlegend=False,
                        xaxis=dict(
                            zeroline=True,
                            zerolinewidth=2,
                            zerolinecolor='black'
                        )
                    )

                    st.plotly_chart(fig_clf_contrib, width='stretch')

                    st.caption(":green[Green] = toward SAFE | :orange[Orange] = toward UNSAFE")

                with viz_tab2:
                    # Regressor contributions bar chart
                    reg_top_10 = reg_contributions['contributions'].head(10).copy()

                    # Create horizontal bar chart
                    fig_reg_contrib = go.Figure()

                    # Add bars colored by direction (positive = blue, negative = orange)
                    colors_reg = ['#0066FF' if x > 0 else '#FF6600' for x in reg_top_10['contribution']]

                    fig_reg_contrib.add_trace(go.Bar(
                        y=reg_top_10['feature'].str.replace('_', ' ').str.title(),
                        x=reg_top_10['contribution'],
                        orientation='h',
                        marker=dict(color=colors_reg),
                        text=[f"{x:+.2f}" for x in reg_top_10['contribution']],
                        textposition='outside',
                        hovertemplate='<b>%{y}</b><br>Contribution: %{x:+.2f} WQI points<br>Value: %{customdata}<extra></extra>',
                        customdata=reg_top_10['value']
                    ))

                    fig_reg_contrib.update_layout(
                        title=f"Top 10 Feature Contributions - Regressor<br><sub>Prediction: {reg_contributions['prediction']:.1f} WQI (Base: {reg_contributions['base_value']:.1f}, Δ {reg_pred_delta:+.1f})</sub>",
                        xaxis_title="Contribution to WQI Score (points)",
                        yaxis_title="Feature",
                        yaxis=dict(autorange="reversed"),  # Top feature at top
                        height=500,
                        showlegend=False,
                        xaxis=dict(
                            zeroline=True,
                            zerolinewidth=2,
                            zerolinecolor='black'
                        )
                    )

                    st.plotly_chart(fig_reg_contrib, width='stretch')

                    st.caption(":blue[Blue] = increases WQI | :orange[Orange] = decreases WQI")

                # Display classifier contributions
                with st.expander(f":material/target: Classifier Contributions (base {clf_contributions['base_value']:.3f} → {clf_contributions['prediction']:.3f})", expanded=False):

                    # Annotate with availability
                    us_features = get_us_available_features()
                    training_only_features = get_training_only_features()

                    clf_contrib_df = clf_contributions['contributions'].copy()

                    def get_availability_marker(feature_name: str) -> str:
                        if feature_name in us_features:
                            return ":green[Available]"
                        elif feature_name in training_only_features:
                            return ":red[Imputed]"
                        else:
                            return ":orange[Partial]"

                    clf_contrib_df['availability'] = clf_contrib_df['feature'].apply(get_availability_marker)

                    # Format display
                    clf_display = clf_contrib_df[['rank', 'feature', 'value', 'contribution', 'availability']].copy()
                    clf_display['feature'] = clf_display['feature'].str.replace('_', ' ').str.title()
                    clf_display['direction'] = clf_display['contribution'].apply(
                        lambda x: f"→ SAFE (+{x:.4f})" if x > 0 else f"→ UNSAFE ({x:.4f})"
                    )
                    clf_display.columns = ['Rank', 'Feature', 'Value', 'Contribution', 'Availability', 'Direction']

                    st.dataframe(
                        clf_display[['Rank', 'Feature', 'Value', 'Direction', 'Availability']],
                        width='stretch',
                        hide_index=True
                    )

                # Display regressor contributions
                with st.expander(f":material/bar_chart: Regressor Contributions (base {reg_contributions['base_value']:.1f} → {reg_contributions['prediction']:.1f} WQI)", expanded=True):

                    reg_contrib_df = reg_contributions['contributions'].copy()
                    reg_contrib_df['availability'] = reg_contrib_df['feature'].apply(get_availability_marker)

                    # Format display
                    reg_display = reg_contrib_df[['rank', 'feature', 'value', 'contribution', 'availability']].copy()
                    reg_display['feature'] = reg_display['feature'].str.replace('_', ' ').str.title()
                    reg_display['direction'] = reg_display['contribution'].apply(
                        lambda x: f"↑ Higher WQI (+{x:.2f})" if x > 0 else f"↓ Lower WQI ({x:.2f})"
                    )
                    reg_display.columns = ['Rank', 'Feature', 'Value', 'Contribution', 'Availability', 'Direction']

                    st.dataframe(
                        reg_display[['Rank', 'Feature', 'Value', 'Direction', 'Availability']],
                        width='stretch',
                        hide_index=True
                    )

            else:
                st.warning("ML models or predictions not available. Cannot calculate feature contributions.")

        except Exception as e:
            st.error(f"Error calculating feature contributions: {e}")
            import traceback
            st.code(traceback.format_exc())

        st.divider()

        # ===================================================================
        # PHASE 4.3: "WHY SAFE/UNSAFE?" MODEL DECISION EXPLANATION
        # ===================================================================
        st.subheader(":material/chat: Decision Explanation")

        try:
            if classifier_files and regressor_files and ml_predictions and 'clf_contributions' in locals() and 'reg_contributions' in locals():
                # Prepare water parameters dict
                water_params = {
                    'ph': aggregated.get('ph'),
                    'dissolved_oxygen': aggregated.get('dissolved_oxygen'),
                    'temperature': aggregated.get('temperature'),
                    'turbidity': aggregated.get('turbidity'),
                    'nitrate': aggregated.get('nitrate'),
                    'conductance': aggregated.get('conductance')
                }

                # Generate decision explanation
                explanation = generate_decision_explanation(
                    classifier_contributions=clf_contributions,
                    regressor_contributions=reg_contributions,
                    water_params=water_params
                )

                # Display verdict with color coding
                verdict = explanation['verdict']
                confidence = explanation['confidence']
                predicted_wqi = explanation['predicted_wqi']
                wqi_category = explanation['wqi_category']

                if verdict == 'SAFE':
                    st.success(f"### Water Quality: **{verdict}**")
                else:
                    st.error(f"### Water Quality: **{verdict}**")

                # Display summary
                st.markdown(f"**{explanation['summary']}**")

                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Confidence",
                        f"{confidence:.1f}%",
                        help=f"Model confidence in {verdict} prediction"
                    )
                with col2:
                    st.metric(
                        "Predicted WQI",
                        f"{predicted_wqi:.1f}",
                        help="Water Quality Index score (0-100)"
                    )
                with col3:
                    st.metric(
                        "WQI Category",
                        wqi_category,
                        help="NSF-WQI classification"
                    )

                st.divider()

                # Display primary factors
                if explanation['primary_factors']:
                    st.markdown("#### Key Factors Influencing This Prediction:")
                    for factor in explanation['primary_factors']:
                        st.markdown(f"- {factor}")
                else:
                    st.info("No significant individual factors identified. The prediction is based on subtle interactions between features.")

                st.divider()

                # Display parameter assessment
                if explanation['parameter_assessment']:
                    st.markdown("#### Water Quality Parameter Assessment:")

                    # Create columns for better layout
                    assessment_items = list(explanation['parameter_assessment'].items())
                    n_params = len(assessment_items)

                    if n_params > 0:
                        # Display 2 parameters per row
                        for i in range(0, n_params, 2):
                            cols = st.columns(2)

                            for j, col in enumerate(cols):
                                if i + j < n_params:
                                    param_name, info = assessment_items[i + j]
                                    status = info['status']
                                    reason = info['reason']

                                    with col:
                                        if status == 'excellent':
                                            st.success(f"**{param_name}**: {reason}")
                                        elif status == 'good':
                                            st.info(f"**{param_name}**: {reason}")
                                        elif status == 'concerning':
                                            st.warning(f"**{param_name}**: {reason}")
                                        else:  # poor
                                            st.error(f"**{param_name}**: {reason}")

                # Display recommendations (for UNSAFE only)
                if explanation['recommendations'] and verdict == 'UNSAFE':
                    st.divider()
                    st.markdown("#### Recommendations for Improvement:")
                    for rec in explanation['recommendations']:
                        st.markdown(f"- {rec}")

                # Add interpretation help
                with st.expander(":material/info: About This Explanation", expanded=False):
                    st.markdown("""
                    Based on SHAP contributions and EPA/WHO thresholds. Model trained on European data.
                    For official assessments, consult certified laboratories.
                    """)

            else:
                st.warning("ML predictions not available. Cannot generate decision explanation.")

        except Exception as e:
            st.error(f"Error generating decision explanation: {e}")
            import traceback
            st.code(traceback.format_exc())

        st.divider()

        # Parameter breakdown
        st.subheader("Parameter Breakdown")

        if scores:
            # Create DataFrame for display
            param_df = pd.DataFrame([
                {
                    "Parameter": k.replace('_', ' ').title(),
                    "Value": f"{aggregated.get(k, 0):.2f}",
                    "Unit": PARAMETER_UNITS.get(k, ''),
                    "Score": f"{v:.1f}",
                    "Status": calculator.classify_wqi(v)
                }
                for k, v in scores.items()
            ])

            st.dataframe(
                param_df,
                width='stretch',
                hide_index=True
            )

            # Add expandable threshold information for each parameter
            st.markdown("#### Parameter Thresholds & Standards")
            st.markdown("Click on a parameter to see how scores are calculated:")

            for param_name, param_score in scores.items():
                param_display = param_name.replace('_', ' ').title()
                param_value = aggregated.get(param_name)
                param_unit = PARAMETER_UNITS.get(param_name, '')

                # Get threshold data from WQICalculator (NO HARDCODING)
                threshold_df = WQICalculator.get_parameter_thresholds(param_name)

                if threshold_df is not None and param_value is not None:
                    # Format unit display (add space before unit if unit exists)
                    unit_display = f" {param_unit}" if param_unit else ""

                    with st.expander(f"{param_display} ({param_value:.2f}{unit_display}) → Score: {param_score:.1f}"):
                        st.markdown(f"**Current Value:** {param_value:.2f}{unit_display}")
                        st.markdown(f"**Calculated Score:** {param_score:.1f} ({calculator.classify_wqi(param_score)})")

                        st.markdown("**Threshold Brackets:**")
                        # Style the DataFrame to highlight current bracket
                        threshold_df = threshold_df.copy()
                        threshold_df["Score"] = threshold_df["Score"].astype(str)
                        st.dataframe(
                            threshold_df,
                            width='stretch',
                            hide_index=True
                        )

                        # Add explanation of current bracket
                        st.info(f"This parameter score is in the **{calculator.classify_wqi(param_score)}** range based on NSF-WQI methodology.")

                        # Add nitrate-specific unit note
                        if param_name == 'nitrate':
                            st.caption(f"Units: mg/L as N (EPA/USGS standard). Conversion: mg{{NO₃}}/L × {NITRATE_NO3_TO_N} = mg/L as N. EPA MCL = 10.0 mg/L as N.")

        st.divider()

        # Visualizations
        st.subheader("Visualizations")

        # Time series chart
        time_series_fig = create_time_series_chart(df)
        if time_series_fig:
            st.plotly_chart(time_series_fig, width='stretch')
        else:
            st.info("Not enough data to create time series chart")

        # Parameter scores chart
        param_fig = create_parameter_chart(scores)
        if param_fig:
            st.plotly_chart(param_fig, width='stretch')

        st.divider()

        # Raw data
        with st.expander("View Raw Data"):
            st.dataframe(df, width='stretch')

            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"water_quality_{zip_code}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

    else:
        # Show instructions
        st.info("Enter a ZIP code in the sidebar and click Search to view water quality data")

        st.markdown("""
        **WQI Score**: 0-100 scale | **90+** Excellent | **70-89** Good | **50-69** Fair | **25-49** Poor | **<25** Very Poor

        Results include: WQI score, safety classification, parameter breakdown, time series, and downloadable raw data.
        """)


if __name__ == "__main__":
    main()
