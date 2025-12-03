"""Search strategy helpers for water quality lookups.

Provides deterministic fallbacks so large geographies (e.g., California) still
return historical measurements by progressively expanding spatial and temporal
windows. The goal is to keep this logic independent from Streamlit so it can be
shared by scripts, tests, and backend services.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List


@dataclass(frozen=True)
class SearchStrategy:
    """Represents a single API search configuration."""

    radius_miles: float
    start_date: datetime
    end_date: datetime
    label: str
    auto_adjusted: bool

    @property
    def history_label(self) -> str:
        """Human friendly description of time span (e.g., '3 years')."""
        span_days = max((self.end_date - self.start_date).days, 1)
        years = span_days / 365.0
        if years >= 1:
            rounded = round(years)
            if abs(years - rounded) < 0.05:
                years = float(rounded)
            unit = "year" if years == 1 else "years"
            value = int(years) if float(years).is_integer() else round(years, 1)
            return f"{value} {unit}"
        months = max(1, span_days // 30)
        unit = "month" if months == 1 else "months"
        return f"{months} {unit}"

    def describe(self) -> str:
        return f"{self.label} ({int(self.radius_miles)} mi, {self.history_label})"


def _clamp_radius(radius_miles: float, max_radius: float) -> float:
    return max(5.0, min(max_radius, round(radius_miles, 1)))


def _extend_start(start_date: datetime, end_date: datetime, years: int) -> datetime:
    target = end_date - timedelta(days=365 * years)
    return min(start_date, target)


def build_search_strategies(
    *,
    radius_miles: float,
    start_date: datetime,
    end_date: datetime,
    max_radius: float = 100.0,
) -> List[SearchStrategy]:
    """Generate deterministic search strategies.

    Strategies progress from the user's exact request to progressively wider
    searches (radius + modest lookback) capped by ``max_radius``.

    Conservative by design: only one radius expansion and one history expansion
    to avoid "fallback everywhere" behavior. This keeps results anchored to the
    user's intent unless the initial query is truly empty.
    """

    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")

    strategies: List[SearchStrategy] = []
    seen = set()

    def add_strategy(radius: float, start: datetime, label: str, auto: bool) -> None:
        radius = _clamp_radius(radius, max_radius)
        start = min(start, end_date)
        key = (radius, start)
        if key in seen:
            return
        seen.add(key)
        strategies.append(
            SearchStrategy(
                radius_miles=radius,
                start_date=start,
                end_date=end_date,
                label=label,
                auto_adjusted=auto,
            )
        )

    # 1) User's inputs
    add_strategy(radius_miles, start_date, "Local search", auto=False)

    # 2) Single radius expansion for sparse geographies
    if radius_miles < max_radius:
        expanded_radius = max(radius_miles * 1.5, radius_miles + 15, 40)
        add_strategy(expanded_radius, start_date, "Expanded radius", auto=True)

    # 3) Modest history extension to 4 years (no extra radius bump)
    # 4 years ensures coverage of infrequent monitoring programs (e.g., groundwater)
    add_strategy(radius_miles, _extend_start(start_date, end_date, 4), "Extended history (4yr)", True)

    return strategies
