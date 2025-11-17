"""
Feature Engineering Module for Water Quality ML Models

This module transforms the Kaggle European water quality dataset into
features suitable for machine learning models. It extracts WQI parameters,
calculates labels, and creates comprehensive feature sets.

Dataset: data/raw/waterPollution.csv (20,000 rows × 29 columns)
Source: European water quality monitoring stations (1991-2017)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.wqi_calculator import WQICalculator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Nitrate unit conversion constant: NO3 (molecular form) → N (nitrogen form)
# Kaggle dataset uses mg{NO3}/L, but EPA standards and US APIs use mg/L as N
# Conversion factor = atomic_weight(N) / molecular_weight(NO3) = 14.0067 / 62.0049 = 0.2258
NITRATE_NO3_TO_N = 0.2258  # Multiply NO3 values by this to get N values


# Verified parameter code mappings from data analysis
PARAMETER_MAPPING = {
    'EEA_3152-01-0': 'ph',                    # pH, 1,157 records
    'EEA_3121-01-5': 'temperature',           # Temperature (Cel), 898 records
    'EEA_3133-01-5': 'dissolved_oxygen',      # DO (mg{O2}/L), 1,214 records
    'CAS_14797-55-8': 'nitrate',              # Nitrate (mg{NO3}/L), 1,289 records
    'EEA_3142-01-6': 'conductance',           # Conductivity (uS/cm), 738 records
}


def load_kaggle_data(file_path: str = "data/raw/waterPollution.csv") -> pd.DataFrame:
    """
    Load and validate the Kaggle water quality dataset.

    Args:
        file_path: Path to the CSV file

    Returns:
        Raw DataFrame with all original columns

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If data shape or structure is unexpected
    """
    logger.info(f"Loading Kaggle dataset from {file_path}")

    # Check file exists
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Kaggle dataset not found at {file_path}")

    # Load CSV
    df = pd.read_csv(file_path)

    # Validate shape
    expected_rows, expected_cols = 20000, 29
    if df.shape != (expected_rows, expected_cols):
        logger.warning(
            f"Expected shape ({expected_rows}, {expected_cols}), "
            f"got {df.shape}. Dataset may have changed."
        )

    # Validate required columns exist
    required_cols = [
        'observedPropertyDeterminandCode',
        'resultMeanValue',
        'phenomenonTimeReferenceYear',
        'waterBodyIdentifier',
        'Country'
    ]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    logger.info(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    logger.info(f"Years: {df['phenomenonTimeReferenceYear'].min()} - {df['phenomenonTimeReferenceYear'].max()}")
    logger.info(f"Countries: {df['Country'].nunique()} unique")

    return df


def extract_wqi_parameters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and pivot the 5 available WQI parameters from long to wide format.

    The Kaggle dataset is in long format (one parameter per row). This function:
    1. Filters for the 5 parameters we can map to WQI
    2. Pivots to wide format (one row per water body + year)
    3. Aggregates multiple measurements using mean
    4. Adds explicit None column for missing turbidity parameter

    Args:
        df: Raw Kaggle DataFrame

    Returns:
        DataFrame with columns:
        - waterBodyIdentifier
        - year
        - ph, temperature, dissolved_oxygen, nitrate, conductance
        - turbidity (always None)
        - Plus all other contextual columns (Country, GDP, etc.)
    """
    logger.info("Extracting WQI parameters from Kaggle dataset")

    # Filter for only the parameter codes we can map
    param_codes = list(PARAMETER_MAPPING.keys())
    df_filtered = df[df['observedPropertyDeterminandCode'].isin(param_codes)].copy()

    logger.info(f"Filtered to {len(df_filtered)} records with WQI parameters")
    logger.info(f"Parameter distribution:\n{df_filtered['observedPropertyDeterminandCode'].value_counts()}")

    # Map codes to parameter names
    df_filtered['parameter'] = df_filtered['observedPropertyDeterminandCode'].map(PARAMETER_MAPPING)

    # Get contextual columns (country, environmental, economic data)
    contextual_cols = [
        'waterBodyIdentifier',
        'phenomenonTimeReferenceYear',
        'Country',
        'parameterWaterBodyCategory',
        'PopulationDensity',
        'TerraMarineProtected_2016_2018',
        'TouristMean_1990_2020',
        'VenueCount',
        'netMigration_2011_2018',
        'droughts_floods_temperature',
        'literacyRate_2010_2018',
        'combustibleRenewables_2009_2014',
        'gdp',
        'composition_food_organic_waste_percent',
        'composition_glass_percent',
        'composition_metal_percent',
        'composition_other_percent',
        'composition_paper_cardboard_percent',
        'composition_plastic_percent',
        'composition_rubber_leather_percent',
        'composition_wood_percent',
        'composition_yard_garden_green_waste_percent',
        'waste_treatment_recycling_percent'
    ]

    # Keep only available contextual columns
    available_contextual = [col for col in contextual_cols if col in df_filtered.columns]

    # Pivot to wide format - aggregate multiple measurements with mean
    # Group by water body, year, and contextual data
    groupby_cols = ['waterBodyIdentifier', 'phenomenonTimeReferenceYear'] + [
        col for col in available_contextual if col not in ['waterBodyIdentifier', 'phenomenonTimeReferenceYear']
    ]

    # Aggregate measurements: mean for water quality, first for contextual data
    agg_dict = {'resultMeanValue': 'mean'}

    df_pivot = df_filtered.groupby(groupby_cols + ['parameter']).agg(agg_dict).reset_index()

    # Pivot parameter names to columns
    df_wide = df_pivot.pivot_table(
        index=groupby_cols,
        columns='parameter',
        values='resultMeanValue',
        aggfunc='mean'
    ).reset_index()

    # Rename year column for clarity
    df_wide = df_wide.rename(columns={'phenomenonTimeReferenceYear': 'year'})

    # CRITICAL: Convert nitrate from mg{NO3}/L to mg/L as N (EPA standard)
    # Kaggle dataset uses mg{NO3}/L but EPA/USGS standards use mg/L as N
    # Conversion factor = atomic_weight(N) / molecular_weight(NO3) = 14.0067 / 62.0049 = 0.2258
    if 'nitrate' in df_wide.columns:
        df_wide['nitrate'] = df_wide['nitrate'] * NITRATE_NO3_TO_N
        logger.info(f"Converted nitrate from mg{{NO3}}/L to mg/L as N (multiplied by {NITRATE_NO3_TO_N})")

    # Add turbidity column (always None - not available in Kaggle dataset)
    df_wide['turbidity'] = None

    logger.info(f"Created {len(df_wide)} unique water body-year combinations")
    logger.info(f"Parameters extracted: {[col for col in df_wide.columns if col in PARAMETER_MAPPING.values()]}")

    # Log missing value statistics for water quality parameters
    wqi_params = list(PARAMETER_MAPPING.values()) + ['turbidity']
    for param in wqi_params:
        if param in df_wide.columns:
            missing_pct = (df_wide[param].isna().sum() / len(df_wide)) * 100
            logger.info(f"  {param}: {missing_pct:.1f}% missing")

    return df_wide


def calculate_wqi_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate WQI scores and labels using the existing WQICalculator.

    This function applies the WQI calculation to each row and adds:
    - wqi_score: Continuous score 0-100
    - wqi_classification: Excellent/Good/Fair/Poor/Very Poor
    - is_safe: Binary (True if WQI >= 70)
    - parameter_scores: Individual parameter scores

    Args:
        df: DataFrame with WQI parameters (from extract_wqi_parameters)

    Returns:
        DataFrame with original columns plus WQI labels
    """
    logger.info("Calculating WQI labels using WQICalculator")

    calculator = WQICalculator()

    # Initialize result columns
    wqi_scores = []
    wqi_classifications = []
    is_safe_flags = []
    param_score_dicts = []

    # Calculate WQI for each row
    for idx, row in df.iterrows():
        try:
            # Extract parameters (handle None and NaN)
            # NOTE: Nitrate already converted to mg/L as N by extract_wqi_parameters() at line 182
            # Do NOT convert again here (previous bug: double conversion)
            params = {
                'ph': row.get('ph') if pd.notna(row.get('ph')) else None,
                'dissolved_oxygen': row.get('dissolved_oxygen') if pd.notna(row.get('dissolved_oxygen')) else None,
                'temperature': row.get('temperature') if pd.notna(row.get('temperature')) else None,
                'turbidity': row.get('turbidity') if pd.notna(row.get('turbidity')) else None,
                'nitrate': row.get('nitrate') if pd.notna(row.get('nitrate')) else None,  # Already in mg/L as N from extract_wqi_parameters()
                'conductance': row.get('conductance') if pd.notna(row.get('conductance')) else None,
            }

            # Calculate WQI (returns tuple: wqi, scores, classification)
            wqi, scores, classification = calculator.calculate_wqi(**params)

            wqi_scores.append(wqi)
            wqi_classifications.append(classification)
            is_safe_flags.append(wqi >= 70.0 if not pd.isna(wqi) else None)
            param_score_dicts.append(scores)

        except Exception as e:
            logger.warning(f"Failed to calculate WQI for row {idx}: {e}")
            wqi_scores.append(None)
            wqi_classifications.append(None)
            is_safe_flags.append(None)
            param_score_dicts.append({})

    # Add to dataframe
    df = df.copy()
    df['wqi_score'] = wqi_scores
    df['wqi_classification'] = wqi_classifications
    df['is_safe'] = is_safe_flags
    df['parameter_scores'] = param_score_dicts

    # Log statistics
    valid_wqi = df['wqi_score'].notna().sum()
    logger.info(f"Successfully calculated WQI for {valid_wqi}/{len(df)} rows ({valid_wqi/len(df)*100:.1f}%)")

    if valid_wqi > 0:
        logger.info(f"WQI statistics:")
        logger.info(f"  Mean: {df['wqi_score'].mean():.2f}")
        logger.info(f"  Median: {df['wqi_score'].median():.2f}")
        logger.info(f"  Min: {df['wqi_score'].min():.2f}")
        logger.info(f"  Max: {df['wqi_score'].max():.2f}")
        logger.info(f"  Safe (WQI >= 70): {df['is_safe'].sum()} ({df['is_safe'].sum()/valid_wqi*100:.1f}%)")
        logger.info(f"\nClassification distribution:\n{df['wqi_classification'].value_counts()}")

    return df


def create_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create comprehensive feature set for machine learning models.

    Features created:
    1. Water quality features: Raw parameters + derived ratios
    2. Temporal features: Year, season, trend indicators
    3. Geographic features: Country one-hot, water body type
    4. Environmental context: All 19 additional dataset columns
    5. Missing indicators: Flags for missing parameters
    6. Interaction features: Scientifically relevant combinations

    Args:
        df: DataFrame with WQI parameters and labels

    Returns:
        DataFrame ready for ML training with all features
    """
    logger.info("Creating ML features")

    df = df.copy()

    # === 1. TEMPORAL FEATURES ===
    logger.info("Creating temporal features")

    # Years since baseline
    df['years_since_1991'] = df['year'] - 1991

    # Decade bins
    df['decade'] = (df['year'] // 10) * 10

    # Time period indicators
    df['is_1990s'] = (df['year'] >= 1990) & (df['year'] < 2000)
    df['is_2000s'] = (df['year'] >= 2000) & (df['year'] < 2010)
    df['is_2010s'] = (df['year'] >= 2010) & (df['year'] < 2020)

    # === 2. WATER QUALITY DERIVED FEATURES ===
    logger.info("Creating water quality derived features")

    # pH deviation from neutral
    df['ph_deviation_from_7'] = np.abs(df['ph'] - 7.0)

    # DO saturation estimate (rough approximation based on temperature)
    # DO saturation decreases with temperature
    df['do_temp_ratio'] = df['dissolved_oxygen'] / (df['temperature'] + 1)  # +1 to avoid div by zero

    # Conductance categories (low/medium/high)
    df['conductance_low'] = (df['conductance'] < 200).astype(float)
    df['conductance_medium'] = ((df['conductance'] >= 200) & (df['conductance'] < 800)).astype(float)
    df['conductance_high'] = (df['conductance'] >= 800).astype(float)

    # Nitrate pollution level
    df['nitrate_pollution_level'] = pd.cut(
        df['nitrate'],
        bins=[-np.inf, 5, 10, 20, np.inf],
        labels=['low', 'moderate', 'high', 'very_high']
    )

    # === 3. MISSING VALUE INDICATORS ===
    logger.info("Creating missing value indicators")

    wqi_params = ['ph', 'dissolved_oxygen', 'temperature', 'turbidity', 'nitrate', 'conductance']
    for param in wqi_params:
        df[f'{param}_missing'] = df[param].isna().astype(int)

    # Count of available parameters
    df['n_params_available'] = sum([~df[param].isna() for param in wqi_params if param in df.columns])

    # === 4. GEOGRAPHIC FEATURES ===
    logger.info("Creating geographic features")

    # Water body type one-hot encoding
    if 'parameterWaterBodyCategory' in df.columns:
        water_body_dummies = pd.get_dummies(
            df['parameterWaterBodyCategory'],
            prefix='water_body'
        )
        df = pd.concat([df, water_body_dummies], axis=1)

    # Country one-hot encoding (top 10 countries, others grouped)
    if 'Country' in df.columns:
        top_countries = df['Country'].value_counts().head(10).index
        df['Country_grouped'] = df['Country'].apply(
            lambda x: x if x in top_countries else 'Other'
        )
        country_dummies = pd.get_dummies(df['Country_grouped'], prefix='country')
        df = pd.concat([df, country_dummies], axis=1)

    # === 5. ENVIRONMENTAL & ECONOMIC FEATURES ===
    logger.info("Adding environmental and economic features (already in dataset)")

    # These are already in the dataset, just log them
    env_features = [
        'PopulationDensity', 'TerraMarineProtected_2016_2018',
        'TouristMean_1990_2020', 'VenueCount', 'netMigration_2011_2018',
        'droughts_floods_temperature', 'literacyRate_2010_2018',
        'combustibleRenewables_2009_2014', 'gdp'
    ]
    available_env = [f for f in env_features if f in df.columns]
    logger.info(f"  Environmental features available: {len(available_env)}")

    # === 6. WASTE MANAGEMENT FEATURES ===
    waste_features = [col for col in df.columns if 'composition_' in col or 'waste_treatment' in col]
    logger.info(f"  Waste management features available: {len(waste_features)}")

    # === 7. INTERACTION FEATURES ===
    logger.info("Creating interaction features")

    # Pollution stress index (high nitrate + low DO)
    df['pollution_stress'] = (
        (df['nitrate'].fillna(0) / 50) * (1 - df['dissolved_oxygen'].fillna(10) / 10)
    )

    # Temperature stress (extreme cold or warm)
    df['temp_stress'] = np.abs(df['temperature'] - 15) / 15  # Optimal around 15°C

    # Economic-environmental interaction
    if 'gdp' in df.columns and 'PopulationDensity' in df.columns:
        df['gdp_per_capita_proxy'] = df['gdp'] / (df['PopulationDensity'] + 1)

    # === SUMMARY ===
    original_features = ['ph', 'dissolved_oxygen', 'temperature', 'nitrate', 'conductance']
    new_features = [col for col in df.columns if col not in original_features and col not in ['waterBodyIdentifier', 'year', 'Country', 'wqi_score', 'wqi_classification', 'is_safe', 'parameter_scores', 'turbidity']]

    logger.info(f"\nFeature engineering complete:")
    logger.info(f"  Original WQI parameters: {len(original_features)}")
    logger.info(f"  New features created: {len(new_features)}")
    logger.info(f"  Total features: {len(df.columns)}")

    return df


def create_core_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create CORE feature set for machine learning models (universal water quality only).

    This function creates features based ONLY on water quality parameters and temporal
    information, excluding European-specific geographic, economic, and waste management
    features. This enables better generalization to non-European water quality data.

    Features created:
    1. Water quality features: Raw parameters + derived ratios
    2. Temporal features: Year, season, trend indicators
    3. Missing indicators: Flags for missing parameters
    4. Interaction features: Scientifically relevant combinations

    Features EXCLUDED (vs create_ml_features):
    - Geographic features: Country, water body type one-hot encoding
    - Environmental/economic: Population density, GDP, tourism, migration, etc.
    - Waste management: Composition and treatment features
    - GDP interactions

    Args:
        df: DataFrame with WQI parameters and labels

    Returns:
        DataFrame ready for ML training with core features only (~25-30 features)
    """
    logger.info("Creating CORE ML features (water quality + temporal only)")

    df = df.copy()

    # === 1. TEMPORAL FEATURES ===
    logger.info("Creating temporal features")

    # Years since baseline
    df['years_since_1991'] = df['year'] - 1991

    # Decade bins
    df['decade'] = (df['year'] // 10) * 10

    # Time period indicators
    df['is_1990s'] = (df['year'] >= 1990) & (df['year'] < 2000)
    df['is_2000s'] = (df['year'] >= 2000) & (df['year'] < 2010)
    df['is_2010s'] = (df['year'] >= 2010) & (df['year'] < 2020)

    # === 2. WATER QUALITY DERIVED FEATURES ===
    logger.info("Creating water quality derived features")

    # pH deviation from neutral
    df['ph_deviation_from_7'] = np.abs(df['ph'] - 7.0)

    # DO saturation estimate (rough approximation based on temperature)
    # DO saturation decreases with temperature
    df['do_temp_ratio'] = df['dissolved_oxygen'] / (df['temperature'] + 1)  # +1 to avoid div by zero

    # Conductance categories (low/medium/high)
    df['conductance_low'] = (df['conductance'] < 200).astype(float)
    df['conductance_medium'] = ((df['conductance'] >= 200) & (df['conductance'] < 800)).astype(float)
    df['conductance_high'] = (df['conductance'] >= 800).astype(float)

    # Nitrate pollution level
    df['nitrate_pollution_level'] = pd.cut(
        df['nitrate'],
        bins=[-np.inf, 5, 10, 20, np.inf],
        labels=['low', 'moderate', 'high', 'very_high']
    )

    # === 3. MISSING VALUE INDICATORS ===
    logger.info("Creating missing value indicators")

    wqi_params = ['ph', 'dissolved_oxygen', 'temperature', 'turbidity', 'nitrate', 'conductance']
    for param in wqi_params:
        df[f'{param}_missing'] = df[param].isna().astype(int)

    # Count of available parameters
    df['n_params_available'] = sum([~df[param].isna() for param in wqi_params if param in df.columns])

    # === 4. INTERACTION FEATURES ===
    logger.info("Creating interaction features")

    # Pollution stress index (high nitrate + low DO)
    df['pollution_stress'] = (
        (df['nitrate'].fillna(0) / 50) * (1 - df['dissolved_oxygen'].fillna(10) / 10)
    )

    # Temperature stress (extreme cold or warm)
    df['temp_stress'] = np.abs(df['temperature'] - 15) / 15  # Optimal around 15°C

    # === SUMMARY ===
    original_features = ['ph', 'dissolved_oxygen', 'temperature', 'nitrate', 'conductance']
    new_features = [col for col in df.columns if col not in original_features and col not in ['waterBodyIdentifier', 'year', 'Country', 'wqi_score', 'wqi_classification', 'is_safe', 'parameter_scores', 'turbidity']]

    logger.info(f"\nCORE feature engineering complete:")
    logger.info(f"  Original WQI parameters: {len(original_features)}")
    logger.info(f"  New features created: {len(new_features)}")
    logger.info(f"  Total features: {len(df.columns)}")
    logger.info(f"  (Excluded: geographic, environmental, economic, waste management features)")

    return df


def prepare_ml_dataset(
    file_path: str = "data/raw/waterPollution.csv",
    save_processed: bool = True,
    core_params_only: bool = False
) -> pd.DataFrame:
    """
    Complete pipeline: Load → Extract → Calculate → Feature Engineering.

    This is the main entry point for preparing the ML dataset.

    Args:
        file_path: Path to raw Kaggle CSV
        save_processed: Whether to save processed data to data/processed/
        core_params_only: If True, use core water quality features only (~24 features)
                         If False, use full feature set including European-specific (59 features)

    Returns:
        Complete DataFrame ready for ML model training
    """
    logger.info("=" * 80)
    logger.info("Starting ML dataset preparation pipeline")
    if core_params_only:
        logger.info("Feature mode: CORE PARAMETERS ONLY (~24 features)")
    else:
        logger.info("Feature mode: FULL FEATURE SET (59 features)")
    logger.info("=" * 80)

    # Step 1: Load raw data
    df = load_kaggle_data(file_path)

    # Step 2: Extract WQI parameters
    df = extract_wqi_parameters(df)

    # Step 3: Calculate WQI labels
    df = calculate_wqi_labels(df)

    # Step 4: Create ML features (conditional based on mode)
    if core_params_only:
        df = create_core_ml_features(df)
    else:
        df = create_ml_features(df)

    # Step 5: Remove rows with invalid WQI scores
    valid_df = df[df['wqi_score'].notna()].copy()
    logger.info(f"\nFinal dataset: {len(valid_df)} valid samples (removed {len(df) - len(valid_df)} invalid)")

    # Step 6: Save processed data
    if save_processed:
        output_path = "data/processed/ml_features.csv"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        valid_df.to_csv(output_path, index=False)
        logger.info(f"Saved processed dataset to {output_path}")

    logger.info("=" * 80)
    logger.info("ML dataset preparation complete!")
    logger.info("=" * 80)

    return valid_df


if __name__ == "__main__":
    # Run the complete pipeline
    df = prepare_ml_dataset()
    print(f"\nDataset shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nColumn names:")
    print(df.columns.tolist())
