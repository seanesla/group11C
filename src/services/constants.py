"""Shared service-layer constants.

These constants are imported by both the Streamlit UI and service modules to
avoid duplication of scientific and visualization settings.
"""

from __future__ import annotations

from typing import Dict


# Color scheme for WQI classifications (used in charts and cards)
WQI_COLORS: Dict[str, str] = {
    "Excellent": "#00CC00",  # Green
    "Good": "#0066FF",       # Blue
    "Fair": "#FFCC00",       # Yellow
    "Poor": "#FF6600",       # Orange
    "Very Poor": "#CC0000",  # Red
}


# Canonical parameter name mapping (WQP characteristic names -> internal keys)
# Used for data aggregation – single source of truth across services/UI.
PARAM_MAPPING: Dict[str, str] = {
    "pH": "ph",
    "Dissolved oxygen (DO)": "dissolved_oxygen",
    "Temperature, water": "temperature",
    "Turbidity": "turbidity",
    "Nitrate": "nitrate",
    "Specific conductance": "conductance",
}

