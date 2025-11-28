"""Tests for prepare_us_features_for_prediction.

These align with the current 18-feature core schema used by the deployed
classifier/regressor (water chemistry + simple temporal/contextual features).
"""

from datetime import datetime

import numpy as np
import pandas as pd

from src.preprocessing.us_data_features import prepare_us_features_for_prediction, prepare_batch_us_features

EXPECTED_COLUMNS = [
    'year',
    'ph', 'dissolved_oxygen', 'temperature', 'nitrate', 'conductance',
    'years_since_1991', 'decade', 'is_1990s', 'is_2000s', 'is_2010s',
    'ph_deviation_from_7', 'do_temp_ratio',
    'conductance_low', 'conductance_medium', 'conductance_high',
    'pollution_stress', 'temp_stress',
]


def test_feature_count_and_order_complete_sample():
    df = prepare_us_features_for_prediction(
        ph=7.2,
        dissolved_oxygen=8.5,
        temperature=15.0,
        turbidity=2.5,
        nitrate=3.2,
        conductance=450.0,
        year=2024,
    )
    assert list(df.columns) == EXPECTED_COLUMNS
    assert df.shape == (1, len(EXPECTED_COLUMNS))


def test_missing_values_propagate_to_ratios():
    df = prepare_us_features_for_prediction(
        ph=None,
        dissolved_oxygen=9.0,
        temperature=None,
        nitrate=5.0,
        conductance=None,
        year=2023,
    )
    assert pd.isna(df.loc[0, 'ph_deviation_from_7'])
    assert pd.isna(df.loc[0, 'do_temp_ratio'])


def test_conductance_buckets():
    df = prepare_us_features_for_prediction(conductance=900.0)
    assert df.loc[0, 'conductance_high'] == 1.0
    assert df.loc[0, 'conductance_medium'] == 0.0
    assert df.loc[0, 'conductance_low'] == 0.0


def test_do_temp_ratio():
    df = prepare_us_features_for_prediction(dissolved_oxygen=8.0, temperature=20.0)
    assert np.isclose(df.loc[0, 'do_temp_ratio'], 8.0 / 21.0)


def test_batch_wrapper_preserves_order():
    measurements = pd.DataFrame([
        {'ph': 7.0, 'dissolved_oxygen': 8.0, 'temperature': 18.0, 'nitrate': 2.0, 'conductance': 400.0},
        {'ph': 6.8, 'dissolved_oxygen': 7.5, 'temperature': 16.0, 'nitrate': 1.5, 'conductance': 500.0},
    ])
    batch = prepare_batch_us_features(measurements)
    assert list(batch.columns) == EXPECTED_COLUMNS
    assert batch.shape == (2, len(EXPECTED_COLUMNS))


def test_year_defaults_to_current_year():
    df = prepare_us_features_for_prediction(ph=7.0)
    assert df.loc[0, 'year'] == datetime.now().year
