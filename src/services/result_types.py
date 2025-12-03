from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, TypeVar

import pandas as pd


T = TypeVar("T")


@dataclass
class ServiceResult(Generic[T]):
    """Generic wrapper for service-layer results.

    Services should avoid UI concerns (no streamlit) and instead return
    structured results plus warnings/errors for callers to handle.
    """

    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    @classmethod
    def ok(cls, data: T, warnings: Optional[List[str]] = None) -> "ServiceResult[T]":
        return cls(success=True, data=data, warnings=warnings or [])

    @classmethod
    def fail(
        cls, error: str, warnings: Optional[List[str]] = None
    ) -> "ServiceResult[T]":
        return cls(success=False, data=None, error=error, warnings=warnings or [])


@dataclass
class FetchResult:
    """Result of fetching water quality measurements for a query."""

    df: pd.DataFrame
    source_label: Optional[str] = None
    attempt_history: List[str] = field(default_factory=list)


@dataclass
class WQIResult:
    """Aggregated WQI calculation from a measurements DataFrame."""

    wqi: float
    scores: Dict[str, float]
    classification: str
    # Aggregated parameter medians used as inputs to WQI and ML models
    aggregated: Dict[str, float]


@dataclass
class MLPredictionResult:
    """Single-sample ML prediction on the 0–100 WQI scale."""

    predicted_wqi: float
    predicted_classification: str
    wqi_color: str
    margin_up: Optional[int]
    margin_down: Optional[int]
    next_up: Optional[str]
    next_down: Optional[str]
    is_near_threshold: bool
    near_threshold_name: Optional[str]
    is_safe: bool
    confidence: float

    def __getitem__(self, key: str) -> Any:
        """Allow dict-style access for backward compatibility in the UI."""
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        """Support 'in' operator for backward compatibility."""
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        """Support dict-style .get() for backward compatibility."""
        return getattr(self, key, default)


@dataclass
class ForecastResult:
    """Simple WQI forecast over future dates."""

    dates: List[datetime]
    predictions: List[float]
    trend: str
    wqi_change: float
    final_wqi: float


@dataclass
class ModelLoadResult:
    """Result of loading classifier/regressor models."""

    classifier: Any
    regressor: Any
