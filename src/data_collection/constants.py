"""
Constants for water quality data collection.

This module centralizes site type definitions and validation ranges used
across the data collection layer to ensure consistent filtering.
"""

from typing import Dict, List, Tuple

# =============================================================================
# SITE TYPE CLASSIFICATIONS
# =============================================================================

# Surface water site types (excludes groundwater for WQI accuracy)
# Groundwater has different chemistry (low DO, high conductance, stable temp)
# that doesn't align well with NSF-WQI assumptions for surface water.

SURFACE_WATER_SITE_TYPES_USGS: List[str] = [
    'ST',   # Stream
    'LK',   # Lake, Reservoir, Impoundment
    'SP',   # Spring
    'WE',   # Wetland
]

SURFACE_WATER_SITE_TYPES_WQP: List[str] = [
    'Stream',
    'Lake, Reservoir, Impoundment',
    'Spring',
    'Wetland',
]

# Marine/estuarine site types to explicitly exclude
# These would contaminate freshwater drinking water assessments with
# saltwater chemistry (high conductance, different parameter baselines)

MARINE_SITE_TYPES_USGS: List[str] = [
    'ES',      # Estuary
    'OC',      # Ocean
    'OC-CO',   # Coastal (oceanic site within 3 nautical miles)
    'ST-TS',   # Tidal stream (tide-influenced)
]

MARINE_SITE_TYPES_WQP: List[str] = [
    'Estuary',
    'Ocean',
]

# =============================================================================
# VALIDATION RANGES
# =============================================================================

# Parameter validation ranges for production use.
# Values outside these ranges indicate measurement errors, sensor malfunction,
# or inappropriate water body type (e.g., marine water in freshwater query).
#
# These are intentionally wider than training data ranges to avoid false
# positives while still catching obvious anomalies.

VALID_RANGES: Dict[str, Tuple[float, float]] = {
    "ph": (0.0, 14.0),              # Standard pH scale
    "dissolved_oxygen": (0.0, 20.0), # mg/L - supersaturation possible
    "temperature": (-5.0, 45.0),     # Celsius - allows for cold/hot springs
    "turbidity": (0.0, 1000.0),      # NTU - allows for storm events
    "nitrate": (0.0, 100.0),         # mg/L as N - allows for agricultural runoff
    "conductance": (0.0, 5000.0),    # uS/cm - above 5000 suggests marine
}

# =============================================================================
# MARINE CONTAMINATION DETECTION
# =============================================================================

# Threshold for flagging potential marine/estuarine contamination.
# Typical freshwater rarely exceeds 2000 uS/cm; seawater is ~50,000 uS/cm.
# 5000 uS/cm is a conservative threshold that catches estuarine water
# while avoiding false positives from mineral-rich freshwater sources.

MARINE_CONDUCTANCE_THRESHOLD: float = 5000.0  # uS/cm
