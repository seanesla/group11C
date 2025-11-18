"""
HYPERTHOROUGH TEST 4: SHAP edge cases - missing parameters, extreme values, boundaries

This test validates that the SHAP feature contribution system handles edge cases robustly:
- Missing water quality parameters (1-5 parameters missing)
- Extreme parameter values (near-zero, super-saturated, contaminated)
- Boundary conditions (EPA MCL thresholds, critical DO levels, neutral pH)
- Numerical stability (very small and very large values)

The system should:
1. Handle missing data gracefully via imputation
2. Produce valid SHAP contributions for extreme values
3. Maintain mathematical accuracy: Σ(SHAP_i) = Prediction - Base Value
4. Not crash or produce NaN/Inf values
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


def generate_edge_case_samples():
    """
    Generate edge case water quality samples covering:
    - Missing parameters (1-5 missing)
    - Extreme values (acidic, alkaline, contaminated, super-saturated)
    - Boundary conditions (thresholds)
    - Numerical stability (very small/large values)

    Returns:
        List of (category, description, X_sample_df) tuples
    """
    samples = []

    # ========================================================================
    # CATEGORY 1: MISSING PARAMETERS
    # ========================================================================
    category = "Missing Parameters"

    # Baseline: All parameters present
    samples.append((category, "Baseline: All parameters present", prepare_us_features_for_prediction(
        ph=7.0, dissolved_oxygen=8.0, temperature=15.0,
        turbidity=2.0, nitrate=3.0, conductance=300.0, year=2024
    )))

    # 1 parameter missing
    samples.append((category, "1 missing: turbidity=NaN", prepare_us_features_for_prediction(
        ph=7.0, dissolved_oxygen=8.0, temperature=15.0,
        turbidity=np.nan, nitrate=3.0, conductance=300.0, year=2024
    )))

    # 2 parameters missing
    samples.append((category, "2 missing: turbidity + nitrate", prepare_us_features_for_prediction(
        ph=7.0, dissolved_oxygen=8.0, temperature=15.0,
        turbidity=np.nan, nitrate=np.nan, conductance=300.0, year=2024
    )))

    # 3 parameters missing
    samples.append((category, "3 missing: turbidity + nitrate + conductance", prepare_us_features_for_prediction(
        ph=7.0, dissolved_oxygen=8.0, temperature=15.0,
        turbidity=np.nan, nitrate=np.nan, conductance=np.nan, year=2024
    )))

    # 4 parameters missing (borderline usability)
    samples.append((category, "4 missing: Only pH + DO available", prepare_us_features_for_prediction(
        ph=7.0, dissolved_oxygen=8.0, temperature=np.nan,
        turbidity=np.nan, nitrate=np.nan, conductance=np.nan, year=2024
    )))

    # 5 parameters missing (minimal data)
    samples.append((category, "5 missing: Only pH available", prepare_us_features_for_prediction(
        ph=7.0, dissolved_oxygen=np.nan, temperature=np.nan,
        turbidity=np.nan, nitrate=np.nan, conductance=np.nan, year=2024
    )))

    # ========================================================================
    # CATEGORY 2: EXTREME VALUES - pH
    # ========================================================================
    category = "Extreme Values - pH"

    samples.append((category, "Extreme acidic: pH 4.0", prepare_us_features_for_prediction(
        ph=4.0, dissolved_oxygen=8.0, temperature=15.0,
        turbidity=2.0, nitrate=3.0, conductance=300.0, year=2024
    )))

    samples.append((category, "Extreme alkaline: pH 10.0", prepare_us_features_for_prediction(
        ph=10.0, dissolved_oxygen=8.0, temperature=15.0,
        turbidity=2.0, nitrate=3.0, conductance=300.0, year=2024
    )))

    samples.append((category, "Near-zero pH: 0.5", prepare_us_features_for_prediction(
        ph=0.5, dissolved_oxygen=8.0, temperature=15.0,
        turbidity=2.0, nitrate=3.0, conductance=300.0, year=2024
    )))

    samples.append((category, "Very high pH: 13.0", prepare_us_features_for_prediction(
        ph=13.0, dissolved_oxygen=8.0, temperature=15.0,
        turbidity=2.0, nitrate=3.0, conductance=300.0, year=2024
    )))

    # ========================================================================
    # CATEGORY 3: EXTREME VALUES - Dissolved Oxygen
    # ========================================================================
    category = "Extreme Values - DO"

    samples.append((category, "Near-zero DO: 0.5 mg/L", prepare_us_features_for_prediction(
        ph=7.0, dissolved_oxygen=0.5, temperature=15.0,
        turbidity=2.0, nitrate=3.0, conductance=300.0, year=2024
    )))

    samples.append((category, "Super-saturated DO: 15.0 mg/L", prepare_us_features_for_prediction(
        ph=7.0, dissolved_oxygen=15.0, temperature=15.0,
        turbidity=2.0, nitrate=3.0, conductance=300.0, year=2024
    )))

    samples.append((category, "Zero DO (anoxic): 0.0 mg/L", prepare_us_features_for_prediction(
        ph=7.0, dissolved_oxygen=0.0, temperature=15.0,
        turbidity=2.0, nitrate=3.0, conductance=300.0, year=2024
    )))

    samples.append((category, "Extreme super-saturation: 20.0 mg/L", prepare_us_features_for_prediction(
        ph=7.0, dissolved_oxygen=20.0, temperature=15.0,
        turbidity=2.0, nitrate=3.0, conductance=300.0, year=2024
    )))

    # ========================================================================
    # CATEGORY 4: EXTREME VALUES - Temperature
    # ========================================================================
    category = "Extreme Values - Temperature"

    samples.append((category, "Near-freezing: 1°C", prepare_us_features_for_prediction(
        ph=7.0, dissolved_oxygen=8.0, temperature=1.0,
        turbidity=2.0, nitrate=3.0, conductance=300.0, year=2024
    )))

    samples.append((category, "Very hot: 40°C", prepare_us_features_for_prediction(
        ph=7.0, dissolved_oxygen=8.0, temperature=40.0,
        turbidity=2.0, nitrate=3.0, conductance=300.0, year=2024
    )))

    samples.append((category, "Zero temperature: 0°C", prepare_us_features_for_prediction(
        ph=7.0, dissolved_oxygen=8.0, temperature=0.0,
        turbidity=2.0, nitrate=3.0, conductance=300.0, year=2024
    )))

    samples.append((category, "Extreme heat: 50°C", prepare_us_features_for_prediction(
        ph=7.0, dissolved_oxygen=8.0, temperature=50.0,
        turbidity=2.0, nitrate=3.0, conductance=300.0, year=2024
    )))

    # ========================================================================
    # CATEGORY 5: EXTREME VALUES - Turbidity
    # ========================================================================
    category = "Extreme Values - Turbidity"

    samples.append((category, "Near-zero turbidity: 0.01 NTU", prepare_us_features_for_prediction(
        ph=7.0, dissolved_oxygen=8.0, temperature=15.0,
        turbidity=0.01, nitrate=3.0, conductance=300.0, year=2024
    )))

    samples.append((category, "Extreme turbidity: 100 NTU", prepare_us_features_for_prediction(
        ph=7.0, dissolved_oxygen=8.0, temperature=15.0,
        turbidity=100.0, nitrate=3.0, conductance=300.0, year=2024
    )))

    samples.append((category, "Zero turbidity: 0.0 NTU", prepare_us_features_for_prediction(
        ph=7.0, dissolved_oxygen=8.0, temperature=15.0,
        turbidity=0.0, nitrate=3.0, conductance=300.0, year=2024
    )))

    samples.append((category, "Ultra-high turbidity: 500 NTU", prepare_us_features_for_prediction(
        ph=7.0, dissolved_oxygen=8.0, temperature=15.0,
        turbidity=500.0, nitrate=3.0, conductance=300.0, year=2024
    )))

    # ========================================================================
    # CATEGORY 6: EXTREME VALUES - Nitrate
    # ========================================================================
    category = "Extreme Values - Nitrate"

    samples.append((category, "Near-zero nitrate: 0.01 mg/L", prepare_us_features_for_prediction(
        ph=7.0, dissolved_oxygen=8.0, temperature=15.0,
        turbidity=2.0, nitrate=0.01, conductance=300.0, year=2024
    )))

    samples.append((category, "EPA MCL: 10.0 mg/L", prepare_us_features_for_prediction(
        ph=7.0, dissolved_oxygen=8.0, temperature=15.0,
        turbidity=2.0, nitrate=10.0, conductance=300.0, year=2024
    )))

    samples.append((category, "Extreme contamination: 50 mg/L", prepare_us_features_for_prediction(
        ph=7.0, dissolved_oxygen=8.0, temperature=15.0,
        turbidity=2.0, nitrate=50.0, conductance=300.0, year=2024
    )))

    samples.append((category, "Ultra-high nitrate: 100 mg/L", prepare_us_features_for_prediction(
        ph=7.0, dissolved_oxygen=8.0, temperature=15.0,
        turbidity=2.0, nitrate=100.0, conductance=300.0, year=2024
    )))

    # ========================================================================
    # CATEGORY 7: EXTREME VALUES - Conductance
    # ========================================================================
    category = "Extreme Values - Conductance"

    samples.append((category, "Very low conductance: 50 µS/cm", prepare_us_features_for_prediction(
        ph=7.0, dissolved_oxygen=8.0, temperature=15.0,
        turbidity=2.0, nitrate=3.0, conductance=50.0, year=2024
    )))

    samples.append((category, "Very high conductance: 2000 µS/cm", prepare_us_features_for_prediction(
        ph=7.0, dissolved_oxygen=8.0, temperature=15.0,
        turbidity=2.0, nitrate=3.0, conductance=2000.0, year=2024
    )))

    samples.append((category, "Near-zero conductance: 10 µS/cm", prepare_us_features_for_prediction(
        ph=7.0, dissolved_oxygen=8.0, temperature=15.0,
        turbidity=2.0, nitrate=3.0, conductance=10.0, year=2024
    )))

    samples.append((category, "Extreme conductance: 5000 µS/cm", prepare_us_features_for_prediction(
        ph=7.0, dissolved_oxygen=8.0, temperature=15.0,
        turbidity=2.0, nitrate=3.0, conductance=5000.0, year=2024
    )))

    # ========================================================================
    # CATEGORY 8: BOUNDARY CONDITIONS
    # ========================================================================
    category = "Boundary Conditions"

    samples.append((category, "Neutral pH: exactly 7.0", prepare_us_features_for_prediction(
        ph=7.0, dissolved_oxygen=8.0, temperature=15.0,
        turbidity=2.0, nitrate=3.0, conductance=300.0, year=2024
    )))

    samples.append((category, "Critical DO: exactly 5.0 mg/L", prepare_us_features_for_prediction(
        ph=7.0, dissolved_oxygen=5.0, temperature=15.0,
        turbidity=2.0, nitrate=3.0, conductance=300.0, year=2024
    )))

    samples.append((category, "EPA MCL nitrate: exactly 10.0 mg/L", prepare_us_features_for_prediction(
        ph=7.0, dissolved_oxygen=8.0, temperature=15.0,
        turbidity=2.0, nitrate=10.0, conductance=300.0, year=2024
    )))

    samples.append((category, "Optimal temperature: exactly 15.0°C", prepare_us_features_for_prediction(
        ph=7.0, dissolved_oxygen=8.0, temperature=15.0,
        turbidity=2.0, nitrate=3.0, conductance=300.0, year=2024
    )))

    # ========================================================================
    # CATEGORY 9: COMBINED EXTREME CONDITIONS
    # ========================================================================
    category = "Combined Extremes"

    samples.append((category, "All optimal values", prepare_us_features_for_prediction(
        ph=7.0, dissolved_oxygen=10.0, temperature=15.0,
        turbidity=0.5, nitrate=1.0, conductance=200.0, year=2024
    )))

    samples.append((category, "All worst-case values", prepare_us_features_for_prediction(
        ph=5.0, dissolved_oxygen=1.0, temperature=35.0,
        turbidity=100.0, nitrate=50.0, conductance=2000.0, year=2024
    )))

    samples.append((category, "Mixed: optimal DO + worst pH", prepare_us_features_for_prediction(
        ph=4.0, dissolved_oxygen=11.0, temperature=15.0,
        turbidity=2.0, nitrate=3.0, conductance=300.0, year=2024
    )))

    samples.append((category, "Mixed: worst DO + optimal pH", prepare_us_features_for_prediction(
        ph=7.0, dissolved_oxygen=0.5, temperature=15.0,
        turbidity=2.0, nitrate=3.0, conductance=300.0, year=2024
    )))

    return samples


def test_shap_edge_cases_classifier():
    """
    Test SHAP contributions for classifier with edge cases.

    Validates:
    - Mathematical property: Σ(SHAP_i) = Prediction - Base Value
    - No crashes with missing data, extreme values, boundaries
    - No NaN or Inf in SHAP values
    - Contributions remain interpretable
    """
    print("\n" + "="*80)
    print("HYPERTHOROUGH TEST 4: SHAP Edge Cases - Classifier")
    print("="*80)

    # Load classifier
    classifier_files = sorted(glob.glob('data/models/classifier_*.joblib'), reverse=True)
    if not classifier_files:
        raise FileNotFoundError("No classifier models found")

    classifier_path = classifier_files[0]
    print(f"✓ Loaded classifier: {classifier_path}")

    # Generate edge case samples
    samples = generate_edge_case_samples()
    print(f"✓ Generated {len(samples)} edge case samples")
    print(f"  Categories: Missing Parameters, Extreme Values (pH/DO/Temp/Turbidity/Nitrate/Conductance),")
    print(f"              Boundary Conditions, Combined Extremes")

    print(f"\n📊 Testing SHAP contributions on edge cases...")
    print("-" * 80)

    results = []
    errors = []

    for i, (category, description, X_sample) in enumerate(samples, 1):
        try:
            # Get SHAP contributions
            contributions = get_prediction_contributions(
                model_path=classifier_path,
                X_sample=X_sample,
                top_n=59  # Get all features
            )

            # Check for NaN/Inf in SHAP values
            has_nan = np.isnan(contributions['shap_sum'])
            has_inf = np.isinf(contributions['shap_sum'])

            # Calculate prediction delta
            pred_delta = contributions['prediction'] - contributions['base_value']

            # Calculate match error
            match_error = abs(contributions['shap_sum'] - pred_delta)

            # Record results
            result = {
                'category': category,
                'description': description,
                'prediction': contributions['prediction'],
                'base_value': contributions['base_value'],
                'pred_delta': pred_delta,
                'shap_sum': contributions['shap_sum'],
                'match_error': match_error,
                'has_nan': has_nan,
                'has_inf': has_inf,
                'passes': match_error < 0.001 and not has_nan and not has_inf
            }
            results.append(result)

            # Print progress
            status = "✅ PASS" if result['passes'] else "❌ FAIL"
            nan_warning = " [NaN!]" if has_nan else ""
            inf_warning = " [Inf!]" if has_inf else ""
            print(f"{i:2d}. {category:25s} | {description:40s}")
            print(f"    Pred={result['prediction']:.4f}, Δ={result['pred_delta']:+.4f}, "
                  f"SHAP={result['shap_sum']:+.4f}, Error={result['match_error']:.6f} "
                  f"{status}{nan_warning}{inf_warning}")

        except Exception as e:
            # Record error
            error = {
                'category': category,
                'description': description,
                'error': str(e)
            }
            errors.append(error)
            print(f"{i:2d}. {category:25s} | {description:40s}")
            print(f"    ❌ EXCEPTION: {str(e)}")

    # Aggregate results
    results_df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    n_passed = results_df['passes'].sum()
    n_failed = len(results_df) - n_passed
    n_errors = len(errors)
    pass_rate = (n_passed / len(samples)) * 100 if len(samples) > 0 else 0

    print(f"Total samples tested:      {len(samples)}")
    print(f"Successful evaluations:    {len(results_df)}")
    print(f"Passed (error < 0.001):    {n_passed} ({pass_rate:.1f}%)")
    print(f"Failed:                    {n_failed}")
    print(f"Exceptions/Crashes:        {n_errors}")

    if len(results_df) > 0:
        print(f"\nMatch Error Statistics:")
        print(f"  Mean:   {results_df['match_error'].mean():.8f}")
        print(f"  Median: {results_df['match_error'].median():.8f}")
        print(f"  Max:    {results_df['match_error'].max():.8f}")
        print(f"  Min:    {results_df['match_error'].min():.8f}")

        # Category breakdown
        print(f"\n📊 Category Breakdown:")
        for cat in results_df['category'].unique():
            cat_df = results_df[results_df['category'] == cat]
            cat_passes = cat_df['passes'].sum()
            cat_total = len(cat_df)
            print(f"  {cat:30s}: {cat_passes}/{cat_total} passed")

    # Show errors if any
    if errors:
        print(f"\n❌ EXCEPTIONS:")
        for err in errors:
            print(f"  {err['category']} | {err['description']}")
            print(f"    Error: {err['error']}")

    # Assert all tests passed
    assert n_errors == 0, f"SHAP edge case test encountered {n_errors} exceptions!"
    assert n_passed == len(results_df), (
        f"SHAP edge case test failed for {n_failed} samples! "
        f"Max error: {results_df['match_error'].max():.8f}"
    )

    print(f"\n✅ ALL {len(results_df)} EDGE CASES PASSED: Classifier handles edge cases robustly")
    print("="*80 + "\n")


def test_shap_edge_cases_regressor():
    """
    Test SHAP contributions for regressor with edge cases.

    Validates:
    - Mathematical property: Σ(SHAP_i) = Prediction - Base Value
    - No crashes with missing data, extreme values, boundaries
    - No NaN or Inf in SHAP values
    - Contributions remain interpretable
    """
    print("\n" + "="*80)
    print("HYPERTHOROUGH TEST 4: SHAP Edge Cases - Regressor")
    print("="*80)

    # Load regressor
    regressor_files = sorted(glob.glob('data/models/regressor_*.joblib'), reverse=True)
    if not regressor_files:
        raise FileNotFoundError("No regressor models found")

    regressor_path = regressor_files[0]
    print(f"✓ Loaded regressor: {regressor_path}")

    # Generate edge case samples
    samples = generate_edge_case_samples()
    print(f"✓ Generated {len(samples)} edge case samples")
    print(f"  Categories: Missing Parameters, Extreme Values (pH/DO/Temp/Turbidity/Nitrate/Conductance),")
    print(f"              Boundary Conditions, Combined Extremes")

    print(f"\n📊 Testing SHAP contributions on edge cases...")
    print("-" * 80)

    results = []
    errors = []

    for i, (category, description, X_sample) in enumerate(samples, 1):
        try:
            # Get SHAP contributions
            contributions = get_prediction_contributions(
                model_path=regressor_path,
                X_sample=X_sample,
                top_n=59  # Get all features
            )

            # Check for NaN/Inf in SHAP values
            has_nan = np.isnan(contributions['shap_sum'])
            has_inf = np.isinf(contributions['shap_sum'])

            # Calculate prediction delta
            pred_delta = contributions['prediction'] - contributions['base_value']

            # Calculate match error
            match_error = abs(contributions['shap_sum'] - pred_delta)

            # Record results
            result = {
                'category': category,
                'description': description,
                'prediction': contributions['prediction'],
                'base_value': contributions['base_value'],
                'pred_delta': pred_delta,
                'shap_sum': contributions['shap_sum'],
                'match_error': match_error,
                'has_nan': has_nan,
                'has_inf': has_inf,
                'passes': match_error < 0.1 and not has_nan and not has_inf  # Tolerance for regressor
            }
            results.append(result)

            # Print progress
            status = "✅ PASS" if result['passes'] else "❌ FAIL"
            nan_warning = " [NaN!]" if has_nan else ""
            inf_warning = " [Inf!]" if has_inf else ""
            print(f"{i:2d}. {category:25s} | {description:40s}")
            print(f"    Pred={result['prediction']:.2f}, Δ={result['pred_delta']:+.2f}, "
                  f"SHAP={result['shap_sum']:+.2f}, Error={result['match_error']:.4f} "
                  f"{status}{nan_warning}{inf_warning}")

        except Exception as e:
            # Record error
            error = {
                'category': category,
                'description': description,
                'error': str(e)
            }
            errors.append(error)
            print(f"{i:2d}. {category:25s} | {description:40s}")
            print(f"    ❌ EXCEPTION: {str(e)}")

    # Aggregate results
    results_df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    n_passed = results_df['passes'].sum()
    n_failed = len(results_df) - n_passed
    n_errors = len(errors)
    pass_rate = (n_passed / len(samples)) * 100 if len(samples) > 0 else 0

    print(f"Total samples tested:      {len(samples)}")
    print(f"Successful evaluations:    {len(results_df)}")
    print(f"Passed (error < 0.1):      {n_passed} ({pass_rate:.1f}%)")
    print(f"Failed:                    {n_failed}")
    print(f"Exceptions/Crashes:        {n_errors}")

    if len(results_df) > 0:
        print(f"\nMatch Error Statistics:")
        print(f"  Mean:   {results_df['match_error'].mean():.6f}")
        print(f"  Median: {results_df['match_error'].median():.6f}")
        print(f"  Max:    {results_df['match_error'].max():.6f}")
        print(f"  Min:    {results_df['match_error'].min():.6f}")

        # Category breakdown
        print(f"\n📊 Category Breakdown:")
        for cat in results_df['category'].unique():
            cat_df = results_df[results_df['category'] == cat]
            cat_passes = cat_df['passes'].sum()
            cat_total = len(cat_df)
            print(f"  {cat:30s}: {cat_passes}/{cat_total} passed")

    # Show errors if any
    if errors:
        print(f"\n❌ EXCEPTIONS:")
        for err in errors:
            print(f"  {err['category']} | {err['description']}")
            print(f"    Error: {err['error']}")

    # Assert all tests passed
    assert n_errors == 0, f"SHAP edge case test encountered {n_errors} exceptions!"
    assert n_passed == len(results_df), (
        f"SHAP edge case test failed for {n_failed} samples! "
        f"Max error: {results_df['match_error'].max():.6f}"
    )

    print(f"\n✅ ALL {len(results_df)} EDGE CASES PASSED: Regressor handles edge cases robustly")
    print("="*80 + "\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("HYPERTHOROUGH TEST 4: SHAP EDGE CASE VALIDATION")
    print("Verifying SHAP handles missing data, extreme values, and boundary conditions")
    print("="*80)

    # Run all tests
    test_shap_edge_cases_classifier()
    test_shap_edge_cases_regressor()

    print("\n" + "="*80)
    print("✅ ALL HYPERTHOROUGH TEST 4 VALIDATIONS PASSED")
    print("="*80)
    print("\n🎯 CONCLUSION: SHAP contributions verified for edge cases:")
    print("   - Missing parameters (1-5 missing): Handled via imputation")
    print("   - Extreme pH (0.5-13.0): No crashes, valid contributions")
    print("   - Extreme DO (0.0-20.0 mg/L): No crashes, valid contributions")
    print("   - Extreme Temperature (0-50°C): No crashes, valid contributions")
    print("   - Extreme Turbidity (0-500 NTU): No crashes, valid contributions")
    print("   - Extreme Nitrate (0.01-100 mg/L): No crashes, valid contributions")
    print("   - Extreme Conductance (10-5000 µS/cm): No crashes, valid contributions")
    print("   - Boundary conditions (EPA MCL, critical thresholds): Handled correctly")
    print("   - Combined extremes: No crashes, mathematically valid")
    print("\n✅ The system is robust to edge cases and maintains mathematical integrity.\\n")
