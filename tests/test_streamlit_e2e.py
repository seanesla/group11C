"""
Headless end-to-end flows exercising the same pipeline the Streamlit UI uses.
These are integration tests (network to WQP/USGS) but avoid browser tooling.
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.geolocation.zipcode_mapper import ZipCodeMapper
from src.data_collection.fallback import fetch_with_fallback
from src.data_collection.wqp_client import WQPClient
from src.utils.wqi_calculator import WQICalculator
from src.services.search_strategies import build_search_strategies

pytestmark = pytest.mark.integration


class TestStreamlitE2E:
    def _fetch_zip_data(self, zip_code: str, radius: float = 25.0) -> pd.DataFrame:
        mapper = ZipCodeMapper()
        lat, lon = mapper.get_coordinates(zip_code)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        strategies = build_search_strategies(
            radius_miles=radius,
            start_date=start_date,
            end_date=end_date,
        )
        wqp_client = WQPClient(timeout=15)
        from src.data_collection.usgs_client import USGSClient
        usgs_client = USGSClient()
        df, _ = fetch_with_fallback(
            latitude=lat,
            longitude=lon,
            radius_miles=strategies[0].radius_miles,
            start_date=strategies[0].start_date,
            end_date=strategies[0].end_date,
            characteristics=["pH", "Temperature, water", "Dissolved oxygen (DO)", "Turbidity", "Nitrate"],
            wqp_client=wqp_client,
            usgs_client=usgs_client,
        )
        return df

    def test_e2e_happy_path(self):
        df = self._fetch_zip_data("20001", radius=25)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty, "Expected data for DC zip 20001"

        # Compute WQI on a sample row (if available)
        calculator = WQICalculator()
        numeric_cols = [c for c in df.columns if c.lower() in {"ph", "temperature", "turbidity", "nitrate", "dissolved_oxygen", "specific conductance", "conductance"}]
        sample = df.iloc[0]
        wqi, _, classification = calculator.calculate_wqi(
            ph=sample.get("pH"),
            dissolved_oxygen=sample.get("dissolved_oxygen"),
            temperature=sample.get("temperature"),
            turbidity=sample.get("turbidity"),
            nitrate=pd.to_numeric(sample.get("ResultMeasureValue"), errors="coerce"),
            conductance=sample.get("conductance"),
        )
        assert 0 <= wqi <= 100
        assert classification in {"Excellent", "Good", "Fair", "Poor", "Very Poor"}

    def test_e2e_invalid_zip(self):
        mapper = ZipCodeMapper()
        with pytest.raises(ValueError):
            mapper.get_coordinates("INVALID")

    def test_e2e_no_data(self):
        # Coordinates far from US should yield empty results
        client = WQPClient(timeout=10)
        df = client.get_water_quality_data(
            latitude=0.1,
            longitude=-150.0,
            radius_miles=10,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            characteristics=["pH"],
        )
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_e2e_visualization_rendering(self):
        df = self._fetch_zip_data("20001", radius=15)
        assert not df.empty
        # Should have numeric measurement values to plot
        values = pd.to_numeric(df.get("ResultMeasureValue"), errors="coerce").dropna()
        assert not values.empty

    def test_e2e_data_download(self):
        df = self._fetch_zip_data("20001", radius=15)
        csv_text = df.to_csv(index=False)
        assert "ResultMeasureValue" in csv_text
        assert len(csv_text.splitlines()) >= 2  # header + at least one row
