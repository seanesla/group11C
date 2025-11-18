"""
HYPERTHOROUGH TEST 1: Verify SHAP contributions sum to prediction difference from base value

This test validates the fundamental mathematical property of SHAP values:
    Σ(SHAP_i) = Prediction - Base Value

Tests across diverse water quality samples to ensure the property holds universally.
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


def generate_diverse_water_samples():
    """
    Generate 25 diverse water quality samples covering different scenarios:
    - Excellent quality (all parameters optimal)
    - Good quality (most parameters good)
    - Fair quality (some parameters concerning)
    - Poor quality (several parameters bad)
    - Very poor quality (most parameters bad)

    Each scenario includes 5 variations with different parameter combinations.

    Returns:
        List of (description, X_sample_df) tuples
    """
    samples = []

    # EXCELLENT QUALITY (5 samples)
    samples.extend([
        ("Excellent #1: Perfect conditions", prepare_us_features_for_prediction(
            ph=7.0, dissolved_oxygen=10.0, temperature=15.0,
            turbidity=0.5, nitrate=1.0, conductance=200.0, year=2024
        )),
        ("Excellent #2: Near-perfect pH/DO", prepare_us_features_for_prediction(
            ph=7.2, dissolved_oxygen=9.5, temperature=14.0,
            turbidity=0.8, nitrate=1.5, conductance=250.0, year=2024
        )),
        ("Excellent #3: Cool pristine water", prepare_us_features_for_prediction(
            ph=7.1, dissolved_oxygen=11.0, temperature=10.0,
            turbidity=0.3, nitrate=0.5, conductance=150.0, year=2024
        )),
        ("Excellent #4: Moderate temp pristine", prepare_us_features_for_prediction(
            ph=7.3, dissolved_oxygen=9.0, temperature=18.0,
            turbidity=1.0, nitrate=2.0, conductance=300.0, year=2024
        )),
        ("Excellent #5: Low nitrate excellent", prepare_us_features_for_prediction(
            ph=7.0, dissolved_oxygen=9.8, temperature=16.0,
            turbidity=0.6, nitrate=0.8, conductance=220.0, year=2024
        )),
    ])

    # GOOD QUALITY (5 samples)
    samples.extend([
        ("Good #1: Slightly acidic", prepare_us_features_for_prediction(
            ph=6.5, dissolved_oxygen=8.0, temperature=20.0,
            turbidity=2.0, nitrate=4.0, conductance=400.0, year=2024
        )),
        ("Good #2: Slightly alkaline", prepare_us_features_for_prediction(
            ph=8.0, dissolved_oxygen=7.5, temperature=22.0,
            turbidity=2.5, nitrate=5.0, conductance=450.0, year=2024
        )),
        ("Good #3: Moderate turbidity", prepare_us_features_for_prediction(
            ph=7.2, dissolved_oxygen=8.5, temperature=17.0,
            turbidity=3.0, nitrate=3.5, conductance=350.0, year=2024
        )),
        ("Good #4: Higher nitrate", prepare_us_features_for_prediction(
            ph=7.1, dissolved_oxygen=8.2, temperature=19.0,
            turbidity=1.8, nitrate=6.0, conductance=380.0, year=2024
        )),
        ("Good #5: High conductance", prepare_us_features_for_prediction(
            ph=7.4, dissolved_oxygen=8.0, temperature=18.0,
            turbidity=2.2, nitrate=4.5, conductance=600.0, year=2024
        )),
    ])

    # FAIR QUALITY (5 samples)
    samples.extend([
        ("Fair #1: Low DO", prepare_us_features_for_prediction(
            ph=7.0, dissolved_oxygen=5.0, temperature=25.0,
            turbidity=5.0, nitrate=7.0, conductance=500.0, year=2024
        )),
        ("Fair #2: High nitrate", prepare_us_features_for_prediction(
            ph=7.2, dissolved_oxygen=7.0, temperature=20.0,
            turbidity=4.0, nitrate=8.5, conductance=550.0, year=2024
        )),
        ("Fair #3: Very acidic", prepare_us_features_for_prediction(
            ph=6.0, dissolved_oxygen=6.5, temperature=18.0,
            turbidity=4.5, nitrate=6.5, conductance=450.0, year=2024
        )),
        ("Fair #4: High turbidity", prepare_us_features_for_prediction(
            ph=7.1, dissolved_oxygen=6.8, temperature=21.0,
            turbidity=8.0, nitrate=7.5, conductance=480.0, year=2024
        )),
        ("Fair #5: Borderline safe", prepare_us_features_for_prediction(
            ph=6.8, dissolved_oxygen=6.0, temperature=23.0,
            turbidity=6.0, nitrate=8.0, conductance=520.0, year=2024
        )),
    ])

    # POOR QUALITY (5 samples)
    samples.extend([
        ("Poor #1: Very low DO", prepare_us_features_for_prediction(
            ph=6.5, dissolved_oxygen=3.0, temperature=28.0,
            turbidity=10.0, nitrate=9.5, conductance=700.0, year=2024
        )),
        ("Poor #2: EPA MCL nitrate", prepare_us_features_for_prediction(
            ph=7.0, dissolved_oxygen=5.5, temperature=24.0,
            turbidity=7.0, nitrate=10.0, conductance=650.0, year=2024
        )),
        ("Poor #3: Very alkaline", prepare_us_features_for_prediction(
            ph=9.0, dissolved_oxygen=4.5, temperature=26.0,
            turbidity=9.0, nitrate=9.0, conductance=600.0, year=2024
        )),
        ("Poor #4: Multiple issues", prepare_us_features_for_prediction(
            ph=6.2, dissolved_oxygen=4.0, temperature=27.0,
            turbidity=12.0, nitrate=11.0, conductance=750.0, year=2024
        )),
        ("Poor #5: High conductance", prepare_us_features_for_prediction(
            ph=6.8, dissolved_oxygen=4.8, temperature=25.0,
            turbidity=8.5, nitrate=9.8, conductance=850.0, year=2024
        )),
    ])

    # VERY POOR QUALITY (5 samples)
    samples.extend([
        ("Very Poor #1: Critically low DO", prepare_us_features_for_prediction(
            ph=5.5, dissolved_oxygen=1.5, temperature=30.0,
            turbidity=15.0, nitrate=15.0, conductance=900.0, year=2024
        )),
        ("Very Poor #2: Extreme pH", prepare_us_features_for_prediction(
            ph=5.0, dissolved_oxygen=2.0, temperature=29.0,
            turbidity=18.0, nitrate=12.0, conductance=800.0, year=2024
        )),
        ("Very Poor #3: High nitrate contamination", prepare_us_features_for_prediction(
            ph=6.0, dissolved_oxygen=2.5, temperature=28.0,
            turbidity=14.0, nitrate=20.0, conductance=950.0, year=2024
        )),
        ("Very Poor #4: All parameters bad", prepare_us_features_for_prediction(
            ph=5.8, dissolved_oxygen=1.8, temperature=32.0,
            turbidity=20.0, nitrate=18.0, conductance=1000.0, year=2024
        )),
        ("Very Poor #5: Extreme contamination", prepare_us_features_for_prediction(
            ph=5.2, dissolved_oxygen=1.0, temperature=31.0,
            turbidity=25.0, nitrate=25.0, conductance=1100.0, year=2024
        )),
    ])

    return samples


def test_shap_sum_property_classifier():
    """
    Test SHAP sum property for classifier across 25 diverse water quality samples.

    Mathematical property being tested:
        Σ(SHAP_i) ≈ Prediction - Base Value

    Tolerance: Match error must be < 0.001 for classifier
    """
    print("\n" + "="*80)
    print("HYPERTHOROUGH TEST 1: SHAP Sum Property - Classifier")
    print("="*80)

    # Load classifier
    classifier_files = sorted(glob.glob('data/models/classifier_*.joblib'), reverse=True)
    if not classifier_files:
        raise FileNotFoundError("No classifier models found")

    classifier_path = classifier_files[0]
    print(f"✓ Loaded classifier: {classifier_path}")

    # Generate diverse test samples
    samples = generate_diverse_water_samples()
    print(f"✓ Generated {len(samples)} diverse water quality samples")

    print(f"\n📊 Testing SHAP sum property on {len(samples)} samples...")
    print("-" * 80)

    results = []

    for i, (description, X_sample) in enumerate(samples, 1):
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
        print(f"{i:2d}. {description:35s}: Pred={result['prediction']:.4f}, "
              f"Δ={result['pred_delta']:+.4f}, SHAP={result['shap_sum']:+.4f}, "
              f"Error={result['match_error']:.6f} {status}")

    # Aggregate results
    results_df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    n_passed = results_df['passes'].sum()
    n_failed = len(results_df) - n_passed
    pass_rate = (n_passed / len(results_df)) * 100

    print(f"Total samples tested:  {len(results_df)}")
    print(f"Passed (error < 0.001): {n_passed} ({pass_rate:.1f}%)")
    print(f"Failed:                {n_failed}")
    print(f"\nMatch Error Statistics:")
    print(f"  Mean:   {results_df['match_error'].mean():.8f}")
    print(f"  Median: {results_df['match_error'].median():.8f}")
    print(f"  Max:    {results_df['match_error'].max():.8f}")
    print(f"  Min:    {results_df['match_error'].min():.8f}")

    # Assert all tests passed
    assert n_passed == len(results_df), (
        f"SHAP sum property violated for {n_failed} samples! "
        f"Max error: {results_df['match_error'].max():.8f}"
    )

    print(f"\n✅ ALL {len(results_df)} SAMPLES PASSED: SHAP sum property holds for classifier")
    print("="*80 + "\n")


def test_shap_sum_property_regressor():
    """
    Test SHAP sum property for regressor across 25 diverse water quality samples.

    Mathematical property being tested:
        Σ(SHAP_i) ≈ Prediction - Base Value

    Tolerance: Match error must be < 0.1 for regressor (WQI points)
    """
    print("\n" + "="*80)
    print("HYPERTHOROUGH TEST 1: SHAP Sum Property - Regressor")
    print("="*80)

    # Load regressor
    regressor_files = sorted(glob.glob('data/models/regressor_*.joblib'), reverse=True)
    if not regressor_files:
        raise FileNotFoundError("No regressor models found")

    regressor_path = regressor_files[0]
    print(f"✓ Loaded regressor: {regressor_path}")

    # Generate diverse test samples
    samples = generate_diverse_water_samples()
    print(f"✓ Generated {len(samples)} diverse water quality samples")

    print(f"\n📊 Testing SHAP sum property on {len(samples)} samples...")
    print("-" * 80)

    results = []

    for i, (description, X_sample) in enumerate(samples, 1):
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
            'description': description,
            'prediction': contributions['prediction'],
            'base_value': contributions['base_value'],
            'pred_delta': pred_delta,
            'shap_sum': contributions['shap_sum'],
            'match_error': match_error,
            'passes': match_error < 0.1  # Tolerance for regressor
        }
        results.append(result)

        # Print progress
        status = "✅ PASS" if result['passes'] else "❌ FAIL"
        print(f"{i:2d}. {description:35s}: Pred={result['prediction']:.2f}, "
              f"Δ={result['pred_delta']:+.2f}, SHAP={result['shap_sum']:+.2f}, "
              f"Error={result['match_error']:.4f} {status}")

    # Aggregate results
    results_df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    n_passed = results_df['passes'].sum()
    n_failed = len(results_df) - n_passed
    pass_rate = (n_passed / len(results_df)) * 100

    print(f"Total samples tested:    {len(results_df)}")
    print(f"Passed (error < 0.1):     {n_passed} ({pass_rate:.1f}%)")
    print(f"Failed:                  {n_failed}")
    print(f"\nMatch Error Statistics:")
    print(f"  Mean:   {results_df['match_error'].mean():.6f}")
    print(f"  Median: {results_df['match_error'].median():.6f}")
    print(f"  Max:    {results_df['match_error'].max():.6f}")
    print(f"  Min:    {results_df['match_error'].min():.6f}")

    # Assert all tests passed
    assert n_passed == len(results_df), (
        f"SHAP sum property violated for {n_failed} samples! "
        f"Max error: {results_df['match_error'].max():.6f}"
    )

    print(f"\n✅ ALL {len(results_df)} SAMPLES PASSED: SHAP sum property holds for regressor")
    print("="*80 + "\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("HYPERTHOROUGH TEST 1: SHAP MATHEMATICAL VERIFICATION")
    print("Verifying that Σ(SHAP_i) = Prediction - Base Value")
    print("="*80)

    # Run all tests
    test_shap_sum_property_classifier()
    test_shap_sum_property_regressor()

    print("\n" + "="*80)
    print("✅ ALL HYPERTHOROUGH TEST 1 VALIDATIONS PASSED")
    print("="*80)
    print("\n🎯 CONCLUSION: SHAP mathematical property verified across:")
    print("   - 25 diverse classifier samples (Excellent, Good, Fair, Poor, Very Poor)")
    print("   - 25 diverse regressor samples (same distribution)")
    print("   - Total: 50 samples with ZERO failures")
    print("\n✅ The get_prediction_contributions() function is mathematically sound.\n")
