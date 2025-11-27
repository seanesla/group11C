import pandas as pd
import pytest
from datetime import datetime, timedelta

from src.data_collection.fallback import convert_usgs_to_wqp_format, fetch_with_fallback


def test_convert_usgs_to_wqp_format_maps_and_filters():
    raw = pd.DataFrame(
        {
            "site_code": ["123", "456"],
            "parameter_code": ["00400", "99999"],
            "parameter_name": ["pH", "Unknown"],
            "datetime": ["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"],
            "value": [7.1, 5.0],
        }
    )

    converted = convert_usgs_to_wqp_format(raw)

    assert list(converted.columns) == [
        "MonitoringLocationIdentifier",
        "ActivityStartDate",
        "CharacteristicName",
        "ResultMeasureValue",
    ]
    assert len(converted) == 1  # unmapped code dropped
    row = converted.iloc[0]
    assert row["MonitoringLocationIdentifier"] == "123"
    assert row["CharacteristicName"] == "pH"
    assert row["ResultMeasureValue"] == pytest.approx(7.1)


def test_fetch_with_fallback_prefers_wqp():
    class FakeWQP:
        def get_data_by_location(self, **kwargs):
            return pd.DataFrame(
                {
                    "MonitoringLocationIdentifier": ["A"],
                    "ActivityStartDate": [datetime.now().date()],
                    "CharacteristicName": ["pH"],
                    "ResultMeasureValue": [7.0],
                }
            )

    class FakeUSGS:
        def get_data_by_location(self, **kwargs):
            return pd.DataFrame()

    df, source = fetch_with_fallback(
        latitude=0,
        longitude=0,
        radius_miles=10,
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now(),
        characteristics=["pH"],
        wqp_client=FakeWQP(),
        usgs_client=FakeUSGS(),
    )

    assert not df.empty
    assert source == "WQP"


def test_fetch_with_fallback_uses_usgs_when_wqp_empty():
    class FakeWQP:
        def get_data_by_location(self, **kwargs):
            return pd.DataFrame()

    class FakeUSGS:
        def get_data_by_location(self, **kwargs):
            return pd.DataFrame(
                {
                    "site_code": ["111"],
                    "parameter_code": ["00400"],
                    "parameter_name": ["pH"],
                    "datetime": ["2025-01-01T00:00:00Z"],
                    "value": [7.5],
                }
            )

    df, source = fetch_with_fallback(
        latitude=0,
        longitude=0,
        radius_miles=10,
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now(),
        characteristics=["pH"],
        wqp_client=FakeWQP(),
        usgs_client=FakeUSGS(),
    )

    assert not df.empty
    assert source == "USGS NWIS"
    assert set(df["CharacteristicName"]) == {"pH"}
