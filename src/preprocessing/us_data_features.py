"""
Prepare US Water Quality Portal data for ML model predictions.

This module converts the 6 WQI parameters from US data into the ~30 CORE features
that the ML models were trained on (water quality + temporal features only).

CORE FEATURE APPROACH:
The models are now trained on UNIVERSAL water quality features only, excluding
European-specific geographic, economic, and waste management features. This enables
better generalization to US water quality data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def prepare_us_features_for_prediction(
    ph: Optional[float] = None,
    dissolved_oxygen: Optional[float] = None,
    temperature: Optional[float] = None,
    turbidity: Optional[float] = None,
    nitrate: Optional[float] = None,
    conductance: Optional[float] = None,
    year: Optional[int] = None
) -> pd.DataFrame:
    """
    Convert US WQI parameters into CORE feature set matching ML model training.

    Creates ~30 CORE features from 6 WQI parameters by engineering:
    - Temporal features (year-based)
    - Water quality derived features (ratios, categories)
    - Missing value indicators
    - Interaction features

    EXCLUDES European-specific features (geographic, environmental, waste management)
    to enable better generalization to US water quality data.

    IMPORTANT: Feature order must match training data exactly.

    Args:
        ph: pH value (6.5-8.5 typical)
        dissolved_oxygen: DO in mg/L (5-11 typical)
        temperature: Water temperature in Celsius (0-30 typical)
        turbidity: Turbidity in NTU (0-10 typical for clean water)
        nitrate: Nitrate in mg/L (0-10 typical)
        conductance: Specific conductance in µS/cm (50-1500 typical)
        year: Year of measurement (defaults to current year)

    Returns:
        DataFrame with single row containing all CORE features in exact training order
    """
    from datetime import datetime

    if year is None:
        year = datetime.now().year

    # === BUILD CORE FEATURES IN EXACT ORDER EXPECTED BY MODEL ===
    features = {}

    # 1. Year (identifier - needed by model)
    features['year'] = year

    # 2. Raw WQI parameters
    features['ph'] = ph
    features['dissolved_oxygen'] = dissolved_oxygen
    features['temperature'] = temperature
    features['nitrate'] = nitrate
    features['conductance'] = conductance

    # 3. Temporal features
    features['years_since_1991'] = year - 1991
    features['decade'] = (year // 10) * 10
    features['is_1990s'] = float(1990 <= year < 2000)
    features['is_2000s'] = float(2000 <= year < 2010)
    features['is_2010s'] = float(2010 <= year < 2020)

    # 3. pH deviation
    features['ph_deviation_from_7'] = abs(ph - 7.0) if ph is not None else np.nan

    # 4. DO-temperature ratio
    if dissolved_oxygen is not None and temperature is not None:
        features['do_temp_ratio'] = dissolved_oxygen / (temperature + 1)
    else:
        features['do_temp_ratio'] = np.nan

    # 5. Conductance categories
    if conductance is not None:
        features['conductance_low'] = float(conductance < 200)
        features['conductance_medium'] = float(200 <= conductance < 800)
        features['conductance_high'] = float(conductance >= 800)
    else:
        features['conductance_low'] = np.nan
        features['conductance_medium'] = np.nan
        features['conductance_high'] = np.nan

    # 6. Missing value indicators
    features['ph_missing'] = int(ph is None)
    features['dissolved_oxygen_missing'] = int(dissolved_oxygen is None)
    features['temperature_missing'] = int(temperature is None)
    features['turbidity_missing'] = int(turbidity is None)
    features['nitrate_missing'] = int(nitrate is None)
    features['conductance_missing'] = int(conductance is None)

    # 7. Count of available parameters
    wqi_params = [ph, dissolved_oxygen, temperature, turbidity, nitrate, conductance]
    features['n_params_available'] = sum([int(p is not None) for p in wqi_params])

    # 8. Interaction features
    nitrate_val = nitrate if nitrate is not None else 0
    do_val = dissolved_oxygen if dissolved_oxygen is not None else 10
    features['pollution_stress'] = (nitrate_val / 50) * (1 - do_val / 10)

    temp_val = temperature if temperature is not None else 15
    features['temp_stress'] = abs(temp_val - 15) / 15

    # Convert to DataFrame with explicit column order
    # CRITICAL: Column order must match training data EXACTLY
    column_order = [
        # Year (identifier)
        'year',
        # Raw WQI parameters
        'ph', 'dissolved_oxygen', 'temperature', 'nitrate', 'conductance',
        # Temporal features
        'years_since_1991', 'decade', 'is_1990s', 'is_2000s', 'is_2010s',
        # Water quality derived
        'ph_deviation_from_7', 'do_temp_ratio',
        'conductance_low', 'conductance_medium', 'conductance_high',
        # Missing indicators
        'ph_missing', 'dissolved_oxygen_missing', 'temperature_missing',
        'turbidity_missing', 'nitrate_missing', 'conductance_missing',
        'n_params_available',
        # Interaction features
        'pollution_stress', 'temp_stress'
    ]

    df = pd.DataFrame([features], columns=column_order)

    # Verify feature count (should be 25 features to match model)
    expected_count = len(column_order)
    if len(df.columns) != expected_count:
        raise ValueError(f"Expected {expected_count} features, got {len(df.columns)}")

    # Verify all expected features are present
    missing_features = set(column_order) - set(features.keys())
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    logger.info(f"Prepared {len(features)} CORE features for US data prediction")
    logger.info(f"  WQI parameters provided: {features['n_params_available']}/6")
    logger.info(f"  Missing core parameter values: {df[['ph', 'dissolved_oxygen', 'temperature', 'nitrate', 'conductance']].isna().sum().sum()}")

    return df


def prepare_batch_us_features(
    measurements_df: pd.DataFrame,
    ph_col: str = 'ph',
    do_col: str = 'dissolved_oxygen',
    temp_col: str = 'temperature',
    turb_col: str = 'turbidity',
    nitrate_col: str = 'nitrate',
    cond_col: str = 'conductance',
    year_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Prepare features for multiple measurements at once.

    Args:
        measurements_df: DataFrame with WQI parameters (one row per measurement)
        ph_col: Name of pH column
        do_col: Name of dissolved oxygen column
        temp_col: Name of temperature column
        turb_col: Name of turbidity column
        nitrate_col: Name of nitrate column
        cond_col: Name of conductance column
        year_col: Name of year column (optional)

    Returns:
        DataFrame with features for all measurements
    """
    logger.info(f"Preparing features for {len(measurements_df)} measurements")

    feature_dfs = []

    for idx, row in measurements_df.iterrows():
        # Extract parameters (handle missing values)
        params = {
            'ph': row.get(ph_col),
            'dissolved_oxygen': row.get(do_col),
            'temperature': row.get(temp_col),
            'turbidity': row.get(turb_col),
            'nitrate': row.get(nitrate_col),
            'conductance': row.get(cond_col)
        }

        if year_col and year_col in row:
            params['year'] = row[year_col]

        # Prepare features for this measurement
        features_df = prepare_us_features_for_prediction(**params)
        feature_dfs.append(features_df)

    # Combine all measurements
    result = pd.concat(feature_dfs, ignore_index=True)

    logger.info(f"Prepared batch features: {result.shape[0]} rows × {result.shape[1]} columns")

    return result


if __name__ == "__main__":
    # Test with example US data
    logging.basicConfig(level=logging.INFO)

    print("Testing US feature preparation...")
    print("=" * 80)

    # Example: Washington DC water quality
    features = prepare_us_features_for_prediction(
        ph=7.2,
        dissolved_oxygen=8.5,
        temperature=15.0,
        turbidity=2.5,
        nitrate=3.2,
        conductance=450.0,
        year=2024
    )

    print(f"\nPrepared {len(features.columns)} features")
    print(f"Non-null features: {features.notna().sum().sum()}")
    print(f"Null features (will be imputed): {features.isna().sum().sum()}")
    print("\nSample of prepared features:")
    print(features.iloc[0, :15])  # Show first 15 features
