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
from utils.wqi_calculator import WQICalculator
from models.model_utils import load_latest_models
from preprocessing.us_data_features import prepare_us_features_for_prediction


# Page configuration
st.set_page_config(
    page_title="Water Quality Index Lookup",
    page_icon="💧",
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


@st.cache_resource
def load_ml_models():
    """
    Load trained ML models with Streamlit caching.

    Returns:
        Tuple of (classifier, regressor). Either may be None if not found.
    """
    try:
        classifier, regressor = load_latest_models()
        return classifier, regressor
    except Exception as e:
        st.error(f"Failed to load ML models: {e}")
        return None, None


def get_wqi_color(classification: str) -> str:
    """Get color for WQI classification."""
    return WQI_COLORS.get(classification, "#808080")


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


def create_time_series_chart(df: pd.DataFrame) -> go.Figure:
    """Create time series chart showing WQI over time with quality zones."""
    if df.empty or 'ActivityStartDate' not in df.columns:
        return None

    # WQP API returns data in long format - need to group by date and calculate WQI
    calculator = WQICalculator()
    wqi_scores = []
    dates = []

    param_mapping = {
        'pH': 'ph',
        'Dissolved oxygen (DO)': 'dissolved_oxygen',
        'Temperature, water': 'temperature',
        'Turbidity': 'turbidity',
        'Nitrate': 'nitrate',
        'Specific conductance': 'conductance'
    }

    # Group by date
    df['ActivityStartDate'] = pd.to_datetime(df['ActivityStartDate'])
    for date, date_df in df.groupby('ActivityStartDate'):
        try:
            # Aggregate parameters for this date
            params = {}
            for characteristic_name, param_key in param_mapping.items():
                mask = date_df['CharacteristicName'] == characteristic_name
                values = pd.to_numeric(date_df.loc[mask, 'ResultMeasureValue'], errors='coerce').dropna()

                if len(values) > 0:
                    params[param_key] = float(values.median())

            if params:  # Only calculate if we have at least some parameters
                wqi, _, _ = calculator.calculate_wqi(**params)
                wqi_scores.append(wqi)
                dates.append(date)
        except Exception:
            continue

    if not wqi_scores:
        return None

    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'Date': dates,
        'WQI': wqi_scores
    }).sort_values('Date')

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


def fetch_water_quality_data(
    zip_code: str,
    radius_miles: float,
    start_date: datetime,
    end_date: datetime
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Fetch water quality data for a ZIP code.

    Returns:
        Tuple of (DataFrame, error_message)
        If successful: (DataFrame, None)
        If failed: (None, error_message)
    """
    try:
        # Initialize clients
        mapper = ZipCodeMapper()
        client = WQPClient()

        # Validate and convert ZIP code
        if not mapper.is_valid_zipcode(zip_code):
            return None, f"Invalid ZIP code: {zip_code}"

        coords = mapper.get_coordinates(zip_code)
        if coords is None:
            return None, f"Could not find coordinates for ZIP code: {zip_code}"

        lat, lon = coords

        # Fetch data
        with st.spinner(f"Fetching water quality data within {radius_miles} miles of ZIP {zip_code}..."):
            df = client.get_data_by_location(
                latitude=lat,
                longitude=lon,
                radius_miles=radius_miles,
                start_date=start_date,
                end_date=end_date,
                characteristics=[
                    "pH",
                    "Dissolved oxygen (DO)",
                    "Temperature, water",
                    "Turbidity",
                    "Nitrate",
                    "Specific conductance"
                ]
            )

        if df.empty:
            return None, f"No water quality data found within {radius_miles} miles of ZIP {zip_code} for the selected date range."

        return df, None

    except Exception as e:
        return None, f"Error fetching data: {str(e)}"


def calculate_overall_wqi(df: pd.DataFrame) -> Tuple[Optional[float], Optional[Dict[str, float]], Optional[str]]:
    """Calculate overall WQI from DataFrame of measurements in long format."""
    if df.empty:
        return None, None, None

    calculator = WQICalculator()

    # WQP API returns data in long format with CharacteristicName and ResultMeasureValue columns
    # Need to aggregate by characteristic name
    aggregated = {}

    param_mapping = {
        'pH': 'ph',
        'Dissolved oxygen (DO)': 'dissolved_oxygen',
        'Temperature, water': 'temperature',
        'Turbidity': 'turbidity',
        'Nitrate': 'nitrate',
        'Specific conductance': 'conductance'
    }

    for characteristic_name, param_key in param_mapping.items():
        # Filter rows for this characteristic
        mask = df['CharacteristicName'] == characteristic_name
        values = pd.to_numeric(df.loc[mask, 'ResultMeasureValue'], errors='coerce').dropna()

        if len(values) > 0:
            # Use median of all measurements for this parameter
            aggregated[param_key] = float(values.median())

    if not aggregated:
        return None, None, None

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
    st.title("💧 Water Quality Index Lookup")
    st.markdown("Search for water quality data by ZIP code and view WQI scores with visualizations.")

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
        min_value=10,
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
        max_value=default_end
    )

    end_date = st.sidebar.date_input(
        "End Date",
        value=default_end,
        max_value=default_end
    )

    # Convert to datetime
    start_date = datetime.combine(start_date, datetime.min.time())
    end_date = datetime.combine(end_date, datetime.min.time())

    # Submit button
    search_button = st.sidebar.button("🔍 Search", type="primary", use_container_width=True)

    # Main area
    if search_button:
        # Fetch data
        df, error = fetch_water_quality_data(zip_code, radius_miles, start_date, end_date)

        if error:
            st.error(error)
            st.info("💡 Try increasing the search radius or adjusting the date range.")
            return

        # Get location info
        mapper = ZipCodeMapper()
        location_info = mapper.get_location_info(zip_code)
        coords = mapper.get_coordinates(zip_code)

        # Calculate overall WQI
        wqi, scores, classification = calculate_overall_wqi(df)

        if wqi is None:
            st.warning("Unable to calculate WQI from available data.")
            st.dataframe(df, use_container_width=True)
            return

        # Get aggregated parameters for ML predictions
        param_mapping = {
            'pH': 'ph',
            'Dissolved oxygen (DO)': 'dissolved_oxygen',
            'Temperature, water': 'temperature',
            'Turbidity': 'turbidity',
            'Nitrate': 'nitrate',
            'Specific conductance': 'conductance'
        }

        aggregated = {}
        for characteristic_name, param_key in param_mapping.items():
            mask = df['CharacteristicName'] == characteristic_name
            values = pd.to_numeric(df.loc[mask, 'ResultMeasureValue'], errors='coerce').dropna()
            if len(values) > 0:
                aggregated[param_key] = float(values.median())

        # Make ML predictions
        ml_predictions = make_ml_predictions(
            aggregated,
            classifier,
            regressor,
            year=datetime.now().year
        )

        # Display results
        st.success(f"✓ Found {len(df)} measurements from {df['MonitoringLocationIdentifier'].nunique()} monitoring stations")

        # Location info card
        st.subheader("📍 Location Information")
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
        st.subheader("💧 Water Quality Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Overall WQI Score",
                f"{wqi:.1f}",
                help="Water Quality Index (0-100 scale)"
            )

        with col2:
            color = get_wqi_color(classification)
            st.markdown(
                f"<div style='padding: 20px; border-radius: 5px; background-color: {color}20; border: 2px solid {color};'>"
                f"<h3 style='margin: 0; color: {color};'>{classification}</h3>"
                f"</div>",
                unsafe_allow_html=True
            )

        with col3:
            calculator = WQICalculator()
            is_safe = calculator.is_safe(wqi)
            safety_icon = "✓" if is_safe else "⚠️"
            safety_text = "Safe for drinking" if is_safe else "May be unsafe"
            safety_color = "#00CC00" if is_safe else "#FF6600"

            st.markdown(
                f"<div style='padding: 20px; border-radius: 5px; background-color: {safety_color}20; border: 2px solid {safety_color};'>"
                f"<h3 style='margin: 0; color: {safety_color};'>{safety_icon} {safety_text}</h3>"
                f"</div>",
                unsafe_allow_html=True
            )

        st.divider()

        # ML Predictions Section
        if ml_predictions:
            st.subheader("🤖 ML Model Predictions")

            # Add disclaimer about European training data
            st.info(
                "**Note:** These predictions come from machine learning models trained on European water quality data (1991-2017). "
                "While chemical relationships are universal, predictions for US locations should be interpreted with caution."
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                # ML Classification
                ml_safety = "SAFE" if ml_predictions['is_safe'] else "UNSAFE"
                ml_color = "#00CC00" if ml_predictions['is_safe'] else "#FF6600"
                st.markdown(
                    f"<div style='padding: 20px; border-radius: 5px; background-color: {ml_color}20; border: 2px solid {ml_color};'>"
                    f"<h4 style='margin: 0;'>ML Classification</h4>"
                    f"<h3 style='margin: 0; color: {ml_color};'>{ml_safety}</h3>"
                    f"</div>",
                    unsafe_allow_html=True
                )

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
                st.markdown(
                    f"<div style='padding: 20px; border-radius: 5px; background-color: {confidence_color}20; border: 2px solid {confidence_color};'>"
                    f"<h4 style='margin: 0;'>Model Confidence</h4>"
                    f"<h3 style='margin: 0; color: {confidence_color};'>{confidence_pct:.1f}%</h3>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            # Show probability breakdown
            with st.expander("View Detailed Probabilities"):
                prob_df = pd.DataFrame([
                    {"Prediction": "Unsafe (WQI < 70)", "Probability": f"{ml_predictions['prob_unsafe']*100:.1f}%"},
                    {"Prediction": "Safe (WQI ≥ 70)", "Probability": f"{ml_predictions['prob_safe']*100:.1f}%"}
                ])
                st.dataframe(prob_df, use_container_width=True, hide_index=True)

                st.markdown("""
                **Model Information:**
                - **Classifier:** RandomForest (98.64% accuracy on test set)
                - **Regressor:** RandomForest (R² = 0.9859 on test set)
                - **Training Data:** 2,939 European water samples (1991-2017)
                - **Limitations:** Geographic mismatch (Europe → US), missing turbidity data
                """)

            st.divider()

            # Future Trend Prediction Section
            st.subheader("📈 Future Water Quality Forecast")

            st.markdown("""
            Based on current water quality parameters and historical trends, here's the predicted WQI over the next 12 months.
            """)

            try:
                # Prepare features for prediction
                import numpy as np
                X_features = prepare_us_features_for_prediction(
                    ph=aggregated.get('ph'),
                    dissolved_oxygen=aggregated.get('dissolved_oxygen'),
                    temperature=aggregated.get('temperature'),
                    turbidity=aggregated.get('turbidity'),
                    nitrate=aggregated.get('nitrate'),
                    conductance=aggregated.get('conductance'),
                    year=datetime.now().year
                )
                X_array = np.array(X_features).reshape(1, -1)

                # Generate 12-month forecast
                current_date = datetime.now()
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
                        st.plotly_chart(trend_chart, use_container_width=True)

                        # Display trend analysis
                        trend = trend_data.get('trend', 'unknown')
                        wqi_change = trend_data.get('wqi_change', 0)
                        final_wqi = trend_data.get('final_wqi', wqi)

                        # Trend description with appropriate icon and color
                        if trend == 'improving':
                            trend_icon = "📈"
                            trend_color = "#00CC00"
                            trend_desc = "improving"
                        elif trend == 'declining':
                            trend_icon = "📉"
                            trend_color = "#FF6600"
                            trend_desc = "declining"
                        else:
                            trend_icon = "➡️"
                            trend_color = "#0066FF"
                            trend_desc = "stable"

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown(
                                f"<div style='padding: 15px; border-radius: 5px; background-color: {trend_color}20; border: 2px solid {trend_color};'>"
                                f"<h4 style='margin: 0;'>{trend_icon} Trend Analysis</h4>"
                                f"<p style='margin: 5px 0 0 0; color: {trend_color}; font-size: 18px; font-weight: bold;'>"
                                f"{trend_desc.upper()}: {wqi_change:+.1f} points over 12 months</p>"
                                f"</div>",
                                unsafe_allow_html=True
                            )

                        with col2:
                            st.metric(
                                "Projected WQI (12 months)",
                                f"{final_wqi:.1f}",
                                delta=f"{wqi_change:.1f}",
                                help="Predicted WQI score after 12 months"
                            )

                        # Forecast disclaimer
                        st.warning(
                            "⚠️ **Forecast Limitations:** These predictions assume current water quality parameters remain constant "
                            "and are based on models trained on historical European data (1991-2017). Actual water quality may vary "
                            "due to seasonal changes, environmental factors, and human activities. Use as guidance only."
                        )

                else:
                    st.info("Unable to generate forecast: temporal features not available in model.")

            except Exception as e:
                st.error(f"Error generating forecast: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

            st.divider()

        # Parameter breakdown
        st.subheader("📊 Parameter Breakdown")

        if scores:
            # Create DataFrame for display
            param_df = pd.DataFrame([
                {"Parameter": k.replace('_', ' ').title(), "Score": f"{v:.1f}", "Status": calculator.classify_wqi(v)}
                for k, v in scores.items()
            ])

            st.dataframe(
                param_df,
                use_container_width=True,
                hide_index=True
            )

        st.divider()

        # Visualizations
        st.subheader("📈 Visualizations")

        # Time series chart
        time_series_fig = create_time_series_chart(df)
        if time_series_fig:
            st.plotly_chart(time_series_fig, use_container_width=True)
        else:
            st.info("Not enough data to create time series chart")

        # Parameter scores chart
        param_fig = create_parameter_chart(scores)
        if param_fig:
            st.plotly_chart(param_fig, use_container_width=True)

        st.divider()

        # Raw data
        with st.expander("🔍 View Raw Data"):
            st.dataframe(df, use_container_width=True)

            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"water_quality_{zip_code}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

    else:
        # Show instructions
        st.info("👈 Enter a ZIP code and click Search to view water quality data")

        st.markdown("""
        ### How to use this app:

        1. **Enter a ZIP code** in the sidebar (e.g., 20001 for Washington DC)
        2. **Adjust search radius** to find monitoring stations near your location
        3. **Select date range** to view historical data
        4. **Click Search** to fetch real water quality data

        ### What you'll see:

        - **Overall WQI Score**: Water Quality Index on a 0-100 scale
        - **Classification**: Excellent, Good, Fair, Poor, or Very Poor
        - **Safety Indicator**: Whether water is safe for drinking (WQI ≥ 70)
        - **Parameter Breakdown**: Individual scores for pH, dissolved oxygen, temperature, etc.
        - **Visualizations**: Time series charts and parameter comparisons
        - **Raw Data**: Complete dataset with download option

        ### WQI Classifications:

        - **90-100**: Excellent - Pristine water quality
        - **70-89**: Good - Safe for most uses
        - **50-69**: Fair - Acceptable but needs monitoring
        - **25-49**: Poor - Treatment recommended
        - **0-24**: Very Poor - Significant contamination
        """)


if __name__ == "__main__":
    main()
