"""Service-layer utilities for the water-quality application.

Business logic is intentionally separated from the Streamlit UI so that
search, aggregation, prediction, and explanation routines can be reused
by scripts and tests.
"""

from .search_strategies import SearchStrategy, build_search_strategies  # noqa: F401
from .result_types import (  # noqa: F401
    ForecastResult,
    FetchResult,
    MLPredictionResult,
    ModelLoadResult,
    ServiceResult,
    WQIResult,
)
