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


def get_wqi_color(classification: str) -> str:
    """Get color for WQI classification."""
    return WQI_COLORS.get(classification, "#808080")


def format_coordinates(lat: float, lon: float) -> str:
    """Format coordinates for display."""
    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"
    return f"{abs(lat):.4f}°{lat_dir}, {abs(lon):.4f}°{lon_dir}"


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

    # Title
    st.title("💧 Water Quality Index Lookup")
    st.markdown("Search for water quality data by ZIP code and view WQI scores with visualizations.")

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
