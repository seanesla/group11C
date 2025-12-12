#!/usr/bin/env python3
"""
Generate poster-quality WQI time series chart for San Francisco, CA.

Fetches real water quality data from USGS/WQP and creates a high-resolution
chart suitable for research posters.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path to import from src/
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import plotly.graph_objects as go
from src.services.water_quality_service import fetch_water_quality_data
from src.services.aggregation_service import get_daily_wqi_scores
from src.services.constants import WQI_COLORS


def main():
    """Main execution function."""
    # Configuration
    zip_code = "94102"
    radius_miles = 25
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    print(f"Fetching water quality data for ZIP {zip_code} (San Francisco, CA)...")
    print(f"  Radius: {radius_miles} miles")
    print(f"  Date range: {start_date.date()} to {end_date.date()}")

    # Fetch data
    result = fetch_water_quality_data(
        zip_code=zip_code,
        radius_miles=radius_miles,
        start_date=start_date,
        end_date=end_date,
        include_groundwater=False
    )

    if not result.success:
        print(f"\nERROR: {result.error_message}")
        sys.exit(1)

    fetch_result = result.data
    df = fetch_result.df

    print(f"  ✓ Retrieved {len(df)} measurements")
    print(f"  ✓ Source: {fetch_result.source_label}")

    # Aggregate to daily WQI
    print("\nCalculating daily WQI scores...")
    daily_wqi = get_daily_wqi_scores(df)

    if daily_wqi.empty:
        print("ERROR: No valid daily WQI scores could be calculated")
        sys.exit(1)

    num_days = len(daily_wqi)
    num_stations = df['MonitoringLocationIdentifier'].nunique()
    wqi_mean = daily_wqi['WQI'].mean()
    wqi_min = daily_wqi['WQI'].min()
    wqi_max = daily_wqi['WQI'].max()
    actual_start = daily_wqi['Date'].min().strftime('%Y-%m-%d')
    actual_end = daily_wqi['Date'].max().strftime('%Y-%m-%d')

    print(f"  ✓ {num_days} days with valid WQI scores")
    print(f"  ✓ WQI range: {wqi_min:.1f} - {wqi_max:.1f}")
    print(f"  ✓ Average WQI: {wqi_mean:.1f}")
    print(f"  ✓ Monitoring stations: {num_stations}")

    # Create figure
    print("\nCreating poster-quality chart...")
    fig = go.Figure()

    # Add quality zone backgrounds
    fig.add_hrect(
        y0=90, y1=100,
        fillcolor=WQI_COLORS["Excellent"],
        opacity=0.15,
        line_width=0,
        annotation_text="Excellent",
        annotation_position="right"
    )
    fig.add_hrect(
        y0=70, y1=90,
        fillcolor=WQI_COLORS["Good"],
        opacity=0.15,
        line_width=0,
        annotation_text="Good",
        annotation_position="right"
    )
    fig.add_hrect(
        y0=50, y1=70,
        fillcolor=WQI_COLORS["Fair"],
        opacity=0.15,
        line_width=0,
        annotation_text="Fair",
        annotation_position="right"
    )
    fig.add_hrect(
        y0=25, y1=50,
        fillcolor=WQI_COLORS["Poor"],
        opacity=0.15,
        line_width=0,
        annotation_text="Poor",
        annotation_position="right"
    )
    fig.add_hrect(
        y0=0, y1=25,
        fillcolor=WQI_COLORS["Very Poor"],
        opacity=0.15,
        line_width=0,
        annotation_text="Very Poor",
        annotation_position="right"
    )

    # Add WQI line
    fig.add_trace(go.Scatter(
        x=daily_wqi['Date'],
        y=daily_wqi['WQI'],
        mode='lines+markers',
        name='WQI Score',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>WQI:</b> %{y:.1f}<extra></extra>'
    ))

    # Caption text
    caption_text = (
        "<b>Water Quality Index Scale:</b><br>"
        "• <b>Excellent (90-100)</b>: Safe for all uses, pristine conditions<br>"
        "• <b>Good (70-89)</b>: Safe for most uses, acceptable quality<br>"
        "• <b>Fair (50-69)</b>: Limited uses, some degradation<br>"
        "• <b>Poor (25-49)</b>: Degraded, restricted uses<br>"
        "• <b>Very Poor (0-24)</b>: Severely degraded, unsuitable<br>"
        "<br>"
        f"<b>Data Source:</b> USGS/Water Quality Portal | "
        f"<b>Location:</b> San Francisco, CA (ZIP {zip_code}, {radius_miles}-mile radius)<br>"
        f"<b>Date Range:</b> {actual_start} to {actual_end} | "
        f"<b>Sample:</b> {num_days} days from {num_stations} monitoring stations<br>"
        "<b>Parameters:</b> pH, dissolved oxygen, temperature, turbidity, nitrate, conductance<br>"
        "<br>"
        "<i>Note: Based on NSF-WQI methodology. Does not test for lead, bacteria, PFAS, or other contaminants.</i>"
    )

    # Add caption below chart
    fig.add_annotation(
        text=caption_text,
        xref="paper", yref="paper",
        x=0, y=-0.18,
        xanchor="left", yanchor="top",
        showarrow=False,
        font=dict(size=18),
        align="left"
    )

    # Update layout for poster quality
    fig.update_layout(
        title=dict(
            text=f"Daily Water Quality Index - San Francisco, CA (ZIP {zip_code})",
            font=dict(size=32, family="Arial, sans-serif"),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title="Date",
            titlefont=dict(size=24, family="Arial, sans-serif"),
            tickfont=dict(size=20),
            gridcolor='lightgray',
            showgrid=True
        ),
        yaxis=dict(
            title="WQI Score",
            range=[0, 105],
            titlefont=dict(size=24, family="Arial, sans-serif"),
            tickfont=dict(size=20),
            gridcolor='lightgray',
            showgrid=True
        ),
        hovermode='x unified',
        width=3000,
        height=2400,
        showlegend=False,
        margin=dict(l=120, r=120, t=180, b=550),  # Extra bottom margin for caption
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Export PNG
    timestamp = datetime.now().strftime('%Y%m%d')
    png_path = output_dir / f"wqi_poster_{zip_code}_{timestamp}.png"

    print(f"\nExporting high-resolution PNG...")
    try:
        fig.write_image(str(png_path), width=3000, height=2400, scale=2)
        file_size = png_path.stat().st_size / (1024 * 1024)  # MB
        print(f"  ✓ Saved PNG: {png_path} ({file_size:.1f} MB)")
        print(f"  ✓ Resolution: 6000 x 4800 pixels (scale=2 for print quality)")
    except Exception as e:
        print(f"\nERROR: Failed to export PNG: {e}")
        print("Make sure kaleido is installed: poetry add kaleido")
        sys.exit(1)

    print(f"\nSUCCESS: Poster-quality chart created and saved.")
    print(f"\nChart details:")
    print(f"  - {num_days} daily WQI measurements")
    print(f"  - Date range: {actual_start} to {actual_end}")
    print(f"  - WQI range: {wqi_min:.1f} to {wqi_max:.1f} (avg: {wqi_mean:.1f})")
    print(f"  - Data from {num_stations} monitoring stations")


if __name__ == "__main__":
    main()
