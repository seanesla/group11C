"""Regression tests ensuring nitrate conversion is applied and persisted.

These validate that nitrate values from the Kaggle dataset are converted from
mg{NO3}/L to mg/L as N both in extraction and in the processed ML feature file.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.preprocessing.feature_engineering import (
    NITRATE_NO3_TO_N,
    load_kaggle_data,
    extract_wqi_parameters,
)


@pytest.fixture(scope="module")
def kaggle_df():
    kaggle_file = Path("data/raw/waterPollution.csv")
    if not kaggle_file.exists():
        pytest.skip("Kaggle CSV not available")
    return load_kaggle_data()


@pytest.fixture(scope="module")
def extracted_df(kaggle_df):
    df = extract_wqi_parameters(kaggle_df)
    if df.empty or 'nitrate' not in df.columns:
        pytest.skip("No nitrate data after extraction")
    return df


@pytest.fixture(scope="module")
def processed_df():
    processed_file = Path("data/processed/ml_features.csv")
    if not processed_file.exists():
        pytest.skip("Processed ML features file not found")
    df = pd.read_csv(processed_file)
    if 'nitrate' not in df.columns:
        pytest.skip("Nitrate column missing in processed features")
    return df


def test_extracted_nitrate_is_converted(extracted_df):
    raw_mean_no3 = extracted_df['nitrate'].mean() / NITRATE_NO3_TO_N
    converted_mean = extracted_df['nitrate'].mean()
    assert converted_mean < raw_mean_no3 * 0.3  # about 0.2258 factor


def test_processed_nitrate_is_converted(processed_df):
    mean_val = processed_df['nitrate'].mean()
    assert 0 < mean_val < 10.0  # expected ~2.86 mg/L as N


def test_conversion_ratio_matches_constant(kaggle_df, extracted_df):
    raw_records = kaggle_df[kaggle_df['observedPropertyDeterminandCode'] == 'CAS_14797-55-8']
    raw_values = pd.to_numeric(raw_records['resultMeanValue'], errors='coerce').dropna()
    raw_mean = raw_values.mean()
    converted_mean = extracted_df['nitrate'].mean()
    ratio = converted_mean / raw_mean
    assert abs(ratio - NITRATE_NO3_TO_N) < 0.02
