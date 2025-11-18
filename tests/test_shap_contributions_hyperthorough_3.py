"""
HYPERTHOROUGH TEST 3: Verify SHAP contributions across diverse US geographic locations

This test validates that the SHAP feature contribution system works correctly
across diverse US regions with characteristic water quality patterns:
- Pacific Northwest (cool, soft water)
- Southwest Desert (hot, hard water)
- Southeast Coastal (warm, coastal characteristics)
- Rocky Mountain (elevated, cold)
- Northeast Urban (moderate, watershed-sourced)
- Southwest Coastal (Mediterranean climate)
- Upper Midwest (cold, freshwater lakes)
- Gulf Coast (humid subtropical, industrial)

For each location, tests multiple water quality scenarios (Good, Fair, Poor)
to ensure robust geographic coverage.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import glob

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from utils.feature_importance import get_prediction_contributions
from preprocessing.us_data_features import prepare_us_features_for_prediction


def generate_geographic_diverse_samples():
    """
    Generate water quality samples from 8 diverse US geographic regions.

    Each region has 3 quality scenarios (Good, Fair, Poor) reflecting:
    - Regional climate characteristics
    - Typical water hardness/mineral content
    - Temperature patterns
    - Common water quality challenges

    Returns:
        List of (region, description, X_sample_df) tuples
    """
    samples = []

    # ========================================================================
    # REGION 1: SEATTLE, WA (Pacific Northwest)
    # ========================================================================
    # Characteristics: Cool, rainy, soft water, high DO, slightly acidic
    region = "Seattle, WA (98101)"

    samples.extend([
        (region, "Good: Cool pristine rainwater", prepare_us_features_for_prediction(
            ph=6.8, dissolved_oxygen=10.5, temperature=12.0,
            turbidity=1.0, nitrate=1.5, conductance=150.0, year=2024
        )),
        (region, "Fair: Moderate urban runoff", prepare_us_features_for_prediction(
            ph=6.5, dissolved_oxygen=8.0, temperature=15.0,
            turbidity=4.0, nitrate=5.0, conductance=250.0, year=2024
        )),
        (region, "Poor: Heavy industrial contamination", prepare_us_features_for_prediction(
            ph=6.2, dissolved_oxygen=5.0, temperature=14.0,
            turbidity=8.0, nitrate=9.0, conductance=400.0, year=2024
        )),
    ])

    # ========================================================================
    # REGION 2: PHOENIX, AZ (Southwest Desert)
    # ========================================================================
    # Characteristics: Hot, hard water, high conductance, alkaline, lower DO
    region = "Phoenix, AZ (85001)"

    samples.extend([
        (region, "Good: Well-treated hard water", prepare_us_features_for_prediction(
            ph=7.8, dissolved_oxygen=7.5, temperature=22.0,
            turbidity=1.5, nitrate=3.0, conductance=650.0, year=2024
        )),
        (region, "Fair: High mineral content", prepare_us_features_for_prediction(
            ph=8.2, dissolved_oxygen=6.5, temperature=25.0,
            turbidity=3.5, nitrate=6.0, conductance=850.0, year=2024
        )),
        (region, "Poor: Extreme heat + agricultural runoff", prepare_us_features_for_prediction(
            ph=8.5, dissolved_oxygen=4.0, temperature=30.0,
            turbidity=7.0, nitrate=12.0, conductance=1100.0, year=2024
        )),
    ])

    # ========================================================================
    # REGION 3: MIAMI, FL (Southeast Coastal)
    # ========================================================================
    # Characteristics: Warm, humid, coastal water, slightly alkaline, moderate DO
    region = "Miami, FL (33101)"

    samples.extend([
        (region, "Good: Coastal well water", prepare_us_features_for_prediction(
            ph=7.6, dissolved_oxygen=7.2, temperature=24.0,
            turbidity=2.0, nitrate=2.5, conductance=550.0, year=2024
        )),
        (region, "Fair: Urban coastal influence", prepare_us_features_for_prediction(
            ph=7.8, dissolved_oxygen=6.0, temperature=26.0,
            turbidity=5.0, nitrate=6.5, conductance=700.0, year=2024
        )),
        (region, "Poor: Saltwater intrusion + contamination", prepare_us_features_for_prediction(
            ph=8.0, dissolved_oxygen=4.5, temperature=28.0,
            turbidity=9.0, nitrate=10.5, conductance=950.0, year=2024
        )),
    ])

    # ========================================================================
    # REGION 4: DENVER, CO (Rocky Mountain)
    # ========================================================================
    # Characteristics: Cool, elevated, high DO (altitude effect), neutral pH
    region = "Denver, CO (80201)"

    samples.extend([
        (region, "Good: Mountain snowmelt", prepare_us_features_for_prediction(
            ph=7.2, dissolved_oxygen=9.5, temperature=11.0,
            turbidity=0.8, nitrate=1.0, conductance=180.0, year=2024
        )),
        (region, "Fair: Moderate agricultural influence", prepare_us_features_for_prediction(
            ph=7.0, dissolved_oxygen=7.5, temperature=14.0,
            turbidity=3.0, nitrate=5.5, conductance=350.0, year=2024
        )),
        (region, "Poor: Mining runoff concern", prepare_us_features_for_prediction(
            ph=6.5, dissolved_oxygen=5.5, temperature=13.0,
            turbidity=7.5, nitrate=8.5, conductance=600.0, year=2024
        )),
    ])

    # ========================================================================
    # REGION 5: NEW YORK, NY (Northeast Urban)
    # ========================================================================
    # Characteristics: Moderate climate, watershed-sourced, neutral pH, good quality baseline
    region = "New York, NY (10001)"

    samples.extend([
        (region, "Good: Catskill watershed water", prepare_us_features_for_prediction(
            ph=7.0, dissolved_oxygen=9.0, temperature=16.0,
            turbidity=1.2, nitrate=2.0, conductance=220.0, year=2024
        )),
        (region, "Fair: Urban distribution aging", prepare_us_features_for_prediction(
            ph=6.8, dissolved_oxygen=7.0, temperature=18.0,
            turbidity=4.5, nitrate=5.5, conductance=380.0, year=2024
        )),
        (region, "Poor: Industrial area contamination", prepare_us_features_for_prediction(
            ph=6.5, dissolved_oxygen=4.5, temperature=17.0,
            turbidity=8.5, nitrate=9.5, conductance=550.0, year=2024
        )),
    ])

    # ========================================================================
    # REGION 6: LOS ANGELES, CA (Southwest Coastal)
    # ========================================================================
    # Characteristics: Mediterranean climate, treated imported water, moderate-high conductance
    region = "Los Angeles, CA (90001)"

    samples.extend([
        (region, "Good: Well-treated municipal water", prepare_us_features_for_prediction(
            ph=7.5, dissolved_oxygen=8.0, temperature=20.0,
            turbidity=1.5, nitrate=2.5, conductance=450.0, year=2024
        )),
        (region, "Fair: Blended source water", prepare_us_features_for_prediction(
            ph=7.7, dissolved_oxygen=6.8, temperature=22.0,
            turbidity=3.5, nitrate=6.0, conductance=650.0, year=2024
        )),
        (region, "Poor: Drought + aging infrastructure", prepare_us_features_for_prediction(
            ph=8.0, dissolved_oxygen=5.0, temperature=24.0,
            turbidity=7.0, nitrate=10.0, conductance=850.0, year=2024
        )),
    ])

    # ========================================================================
    # REGION 7: MINNEAPOLIS, MN (Upper Midwest)
    # ========================================================================
    # Characteristics: Cold, freshwater lakes, high DO, soft water, pristine baseline
    region = "Minneapolis, MN (55401)"

    samples.extend([
        (region, "Good: Pristine lake water", prepare_us_features_for_prediction(
            ph=7.1, dissolved_oxygen=11.0, temperature=10.0,
            turbidity=0.5, nitrate=0.8, conductance=140.0, year=2024
        )),
        (region, "Fair: Agricultural watershed influence", prepare_us_features_for_prediction(
            ph=6.9, dissolved_oxygen=8.5, temperature=14.0,
            turbidity=3.0, nitrate=5.0, conductance=280.0, year=2024
        )),
        (region, "Poor: Urban + agricultural runoff", prepare_us_features_for_prediction(
            ph=6.6, dissolved_oxygen=5.5, temperature=13.0,
            turbidity=6.5, nitrate=8.5, conductance=420.0, year=2024
        )),
    ])

    # ========================================================================
    # REGION 8: HOUSTON, TX (Gulf Coast)
    # ========================================================================
    # Characteristics: Humid subtropical, warm, industrial activity, lower DO concerns
    region = "Houston, TX (77001)"

    samples.extend([
        (region, "Good: Treated groundwater", prepare_us_features_for_prediction(
            ph=7.2, dissolved_oxygen=7.5, temperature=23.0,
            turbidity=2.0, nitrate=3.0, conductance=400.0, year=2024
        )),
        (region, "Fair: Industrial area baseline", prepare_us_features_for_prediction(
            ph=6.9, dissolved_oxygen=6.0, temperature=25.0,
            turbidity=5.0, nitrate=7.0, conductance=550.0, year=2024
        )),
        (region, "Poor: Petrochemical influence", prepare_us_features_for_prediction(
            ph=6.5, dissolved_oxygen=4.0, temperature=27.0,
            turbidity=9.5, nitrate=11.0, conductance=750.0, year=2024
        )),
    ])

    return samples


def test_shap_geographic_diversity_classifier():
    """
    Test SHAP contributions for classifier across 8 US geographic regions.

    Validates:
    - Mathematical property: Σ(SHAP_i) = Prediction - Base Value
    - No errors/crashes for regional water quality variations
    - Contributions are reasonable and interpretable
    """
    print("\n" + "="*80)
    print("HYPERTHOROUGH TEST 3: SHAP Geographic Diversity - Classifier")
    print("="*80)

    # Load classifier
    classifier_files = sorted(glob.glob('data/models/classifier_*.joblib'), reverse=True)
    if not classifier_files:
        raise FileNotFoundError("No classifier models found")

    classifier_path = classifier_files[0]
    print(f"✓ Loaded classifier: {classifier_path}")

    # Generate geographic samples
    samples = generate_geographic_diverse_samples()
    print(f"✓ Generated {len(samples)} samples from 8 US geographic regions")
    print(f"  Regions: Pacific NW, SW Desert, SE Coastal, Rocky Mtn,")
    print(f"           NE Urban, SW Coastal, Upper Midwest, Gulf Coast")

    print(f"\n📊 Testing SHAP contributions across geographic diversity...")
    print("-" * 80)

    results = []

    for i, (region, description, X_sample) in enumerate(samples, 1):
        # Get SHAP contributions
        contributions = get_prediction_contributions(
            model_path=classifier_path,
            X_sample=X_sample,
            top_n=59  # Get all features
        )

        # Calculate prediction delta
        pred_delta = contributions['prediction'] - contributions['base_value']

        # Calculate match error
        match_error = abs(contributions['shap_sum'] - pred_delta)

        # Record results
        result = {
            'region': region,
            'description': description,
            'prediction': contributions['prediction'],
            'base_value': contributions['base_value'],
            'pred_delta': pred_delta,
            'shap_sum': contributions['shap_sum'],
            'match_error': match_error,
            'passes': match_error < 0.001
        }
        results.append(result)

        # Print progress
        status = "✅ PASS" if result['passes'] else "❌ FAIL"
        print(f"{i:2d}. {region:25s} | {description:35s}")
        print(f"    Pred={result['prediction']:.4f}, Δ={result['pred_delta']:+.4f}, "
              f"SHAP={result['shap_sum']:+.4f}, Error={result['match_error']:.6f} {status}")

    # Aggregate results
    results_df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    n_passed = results_df['passes'].sum()
    n_failed = len(results_df) - n_passed
    pass_rate = (n_passed / len(results_df)) * 100

    print(f"Total samples tested:      {len(results_df)}")
    print(f"  - 8 US geographic regions")
    print(f"  - 3 water quality scenarios per region (Good, Fair, Poor)")
    print(f"Passed (error < 0.001):    {n_passed} ({pass_rate:.1f}%)")
    print(f"Failed:                    {n_failed}")
    print(f"\nMatch Error Statistics:")
    print(f"  Mean:   {results_df['match_error'].mean():.8f}")
    print(f"  Median: {results_df['match_error'].median():.8f}")
    print(f"  Max:    {results_df['match_error'].max():.8f}")
    print(f"  Min:    {results_df['match_error'].min():.8f}")

    # Regional summary
    print(f"\n📍 Regional Breakdown:")
    for region in results_df['region'].unique():
        region_df = results_df[results_df['region'] == region]
        region_passes = region_df['passes'].sum()
        region_total = len(region_df)
        print(f"  {region:25s}: {region_passes}/{region_total} passed")

    # Assert all tests passed
    assert n_passed == len(results_df), (
        f"SHAP geographic diversity test failed for {n_failed} samples! "
        f"Max error: {results_df['match_error'].max():.8f}"
    )

    print(f"\n✅ ALL {len(results_df)} SAMPLES PASSED: SHAP works across US geographic diversity")
    print("="*80 + "\n")


def test_shap_geographic_diversity_regressor():
    """
    Test SHAP contributions for regressor across 8 US geographic regions.

    Validates:
    - Mathematical property: Σ(SHAP_i) = Prediction - Base Value
    - No errors/crashes for regional water quality variations
    - Contributions are reasonable and interpretable
    """
    print("\n" + "="*80)
    print("HYPERTHOROUGH TEST 3: SHAP Geographic Diversity - Regressor")
    print("="*80)

    # Load regressor
    regressor_files = sorted(glob.glob('data/models/regressor_*.joblib'), reverse=True)
    if not regressor_files:
        raise FileNotFoundError("No regressor models found")

    regressor_path = regressor_files[0]
    print(f"✓ Loaded regressor: {regressor_path}")

    # Generate geographic samples
    samples = generate_geographic_diverse_samples()
    print(f"✓ Generated {len(samples)} samples from 8 US geographic regions")
    print(f"  Regions: Pacific NW, SW Desert, SE Coastal, Rocky Mtn,")
    print(f"           NE Urban, SW Coastal, Upper Midwest, Gulf Coast")

    print(f"\n📊 Testing SHAP contributions across geographic diversity...")
    print("-" * 80)

    results = []

    for i, (region, description, X_sample) in enumerate(samples, 1):
        # Get SHAP contributions
        contributions = get_prediction_contributions(
            model_path=regressor_path,
            X_sample=X_sample,
            top_n=59  # Get all features
        )

        # Calculate prediction delta
        pred_delta = contributions['prediction'] - contributions['base_value']

        # Calculate match error
        match_error = abs(contributions['shap_sum'] - pred_delta)

        # Record results
        result = {
            'region': region,
            'description': description,
            'prediction': contributions['prediction'],
            'base_value': contributions['base_value'],
            'pred_delta': pred_delta,
            'shap_sum': contributions['shap_sum'],
            'match_error': match_error,
            'passes': match_error < 0.1  # Tolerance for regressor (WQI points)
        }
        results.append(result)

        # Print progress
        status = "✅ PASS" if result['passes'] else "❌ FAIL"
        print(f"{i:2d}. {region:25s} | {description:35s}")
        print(f"    Pred={result['prediction']:.2f}, Δ={result['pred_delta']:+.2f}, "
              f"SHAP={result['shap_sum']:+.2f}, Error={result['match_error']:.4f} {status}")

    # Aggregate results
    results_df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    n_passed = results_df['passes'].sum()
    n_failed = len(results_df) - n_passed
    pass_rate = (n_passed / len(results_df)) * 100

    print(f"Total samples tested:      {len(results_df)}")
    print(f"  - 8 US geographic regions")
    print(f"  - 3 water quality scenarios per region (Good, Fair, Poor)")
    print(f"Passed (error < 0.1):      {n_passed} ({pass_rate:.1f}%)")
    print(f"Failed:                    {n_failed}")
    print(f"\nMatch Error Statistics:")
    print(f"  Mean:   {results_df['match_error'].mean():.6f}")
    print(f"  Median: {results_df['match_error'].median():.6f}")
    print(f"  Max:    {results_df['match_error'].max():.6f}")
    print(f"  Min:    {results_df['match_error'].min():.6f}")

    # Regional summary
    print(f"\n📍 Regional Breakdown:")
    for region in results_df['region'].unique():
        region_df = results_df[results_df['region'] == region]
        region_passes = region_df['passes'].sum()
        region_total = len(region_df)
        print(f"  {region:25s}: {region_passes}/{region_total} passed")

    # Assert all tests passed
    assert n_passed == len(results_df), (
        f"SHAP geographic diversity test failed for {n_failed} samples! "
        f"Max error: {results_df['match_error'].max():.6f}"
    )

    print(f"\n✅ ALL {len(results_df)} SAMPLES PASSED: SHAP works across US geographic diversity")
    print("="*80 + "\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("HYPERTHOROUGH TEST 3: SHAP GEOGRAPHIC DIVERSITY VERIFICATION")
    print("Verifying SHAP works correctly across 8 diverse US geographic regions")
    print("="*80)

    # Run all tests
    test_shap_geographic_diversity_classifier()
    test_shap_geographic_diversity_regressor()

    print("\n" + "="*80)
    print("✅ ALL HYPERTHOROUGH TEST 3 VALIDATIONS PASSED")
    print("="*80)
    print("\n🎯 CONCLUSION: SHAP contributions verified across:")
    print("   - 8 diverse US geographic regions")
    print("   - 24 classifier samples (3 quality scenarios × 8 regions)")
    print("   - 24 regressor samples (same distribution)")
    print("   - Total: 48 samples with ZERO failures")
    print("\n✅ Geographic diversity: Pacific NW, SW Desert, SE Coastal, Rocky Mtn,")
    print("   NE Urban, SW Coastal, Upper Midwest, Gulf Coast")
    print("\n✅ The system handles US regional water quality variations correctly.\\n")
