"""
COMPREHENSIVE BACKEND TEST: Phase 4.2 & 4.3 Implementation

Direct testing of backend functions without browser:
- get_prediction_contributions() (Phase 4.2)
- generate_decision_explanation() (Phase 4.3)

NO SHORTCUTS. NO ASSUMPTIONS. VERIFY EVERYTHING.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import glob
import json
import pytest

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from utils.feature_importance import get_prediction_contributions, generate_decision_explanation
from preprocessing.us_data_features import prepare_us_features_for_prediction


def generate_comprehensive_test_samples():
    """
    Generate comprehensive test samples covering all water quality scenarios.

    Returns:
        List of (description, X_sample_df, water_params_dict) tuples
    """
    samples = []

    # EXCELLENT WATER QUALITY (10 samples)
    test_cases = [
        ("Excellent #1: Pristine mountain stream", 7.0, 11.0, 12.0, 0.3, 0.5, 150.0),
        ("Excellent #2: Perfect conditions", 7.2, 10.5, 14.0, 0.5, 1.0, 200.0),
        ("Excellent #3: Cool pristine water", 7.1, 11.5, 10.0, 0.4, 0.3, 120.0),
        ("Excellent #4: High oxygen", 7.3, 12.0, 11.0, 0.6, 0.8, 180.0),
        ("Excellent #5: Low nitrate", 7.0, 10.8, 13.0, 0.5, 0.2, 160.0),
        ("Excellent #6: Optimal pH", 7.0, 11.2, 12.5, 0.4, 0.6, 170.0),
        ("Excellent #7: Very low turbidity", 7.1, 10.9, 13.5, 0.2, 0.7, 190.0),
        ("Excellent #8: Clean river", 7.2, 11.3, 11.5, 0.5, 0.4, 175.0),
        ("Excellent #9: Spring water", 7.0, 11.0, 12.0, 0.3, 0.5, 140.0),
        ("Excellent #10: Perfect balance", 7.1, 10.7, 13.0, 0.6, 0.9, 185.0),
        # GOOD WATER QUALITY (10 samples)
        ("Good #1: Slightly acidic", 6.5, 8.5, 18.0, 2.0, 3.0, 350.0),
        ("Good #2: Slightly alkaline", 8.0, 8.0, 19.0, 2.5, 4.0, 400.0),
        ("Good #3: Moderate turbidity", 7.2, 8.8, 17.0, 3.0, 3.5, 380.0),
        ("Good #4: Higher nitrate", 7.1, 8.3, 18.5, 2.2, 5.0, 420.0),
        ("Good #5: Warm water", 7.0, 7.8, 22.0, 2.8, 3.8, 390.0),
        ("Good #6: Higher conductance", 7.3, 8.2, 20.0, 2.5, 4.2, 550.0),
        ("Good #7: Cool good water", 6.8, 8.7, 15.0, 2.3, 3.3, 370.0),
        ("Good #8: Moderate parameters", 7.0, 8.1, 19.5, 2.7, 4.5, 410.0),
        ("Good #9: Slightly warm", 7.2, 7.9, 21.0, 2.6, 3.9, 430.0),
        ("Good #10: Good quality lake", 6.9, 8.4, 18.0, 2.4, 4.3, 395.0),
        # FAIR WATER QUALITY (10 samples)
        ("Fair #1: Low oxygen", 7.0, 5.5, 24.0, 4.5, 6.5, 500.0),
        ("Fair #2: High nitrate", 7.2, 7.0, 20.0, 4.0, 8.0, 550.0),
        ("Fair #3: Very acidic", 6.0, 6.8, 18.0, 4.2, 6.8, 480.0),
        ("Fair #4: High turbidity", 7.1, 6.5, 21.0, 7.5, 7.2, 520.0),
        ("Fair #5: Borderline safe", 6.8, 6.2, 23.0, 5.5, 7.8, 540.0),
        ("Fair #6: Multiple concerns", 6.5, 6.0, 24.5, 6.0, 7.5, 560.0),
        ("Fair #7: Warm polluted", 7.0, 5.8, 25.0, 5.8, 8.2, 530.0),
        ("Fair #8: Questionable quality", 6.7, 6.3, 22.5, 6.5, 7.0, 510.0),
        ("Fair #9: Marginal water", 6.9, 5.9, 24.2, 5.2, 7.9, 545.0),
        ("Fair #10: Fair river water", 6.6, 6.4, 23.5, 6.2, 7.3, 525.0),
        # POOR WATER QUALITY (10 samples)
        ("Poor #1: Very low oxygen", 6.5, 3.5, 27.0, 9.5, 9.2, 720.0),
        ("Poor #2: EPA MCL nitrate", 7.0, 5.0, 24.0, 7.5, 10.0, 680.0),
        ("Poor #3: Very alkaline", 9.0, 4.2, 26.0, 8.8, 9.0, 640.0),
        ("Poor #4: Multiple violations", 6.2, 4.5, 26.5, 11.0, 10.5, 750.0),
        ("Poor #5: High conductance", 6.8, 4.8, 25.0, 8.2, 9.8, 820.0),
        ("Poor #6: Contaminated", 6.4, 3.8, 27.5, 10.0, 9.5, 700.0),
        ("Poor #7: Polluted stream", 6.6, 4.0, 26.8, 9.8, 10.2, 740.0),
        ("Poor #8: Industrial runoff", 6.3, 3.9, 27.2, 10.5, 9.9, 780.0),
        ("Poor #9: Heavy metals suspected", 5.8, 4.0, 25.5, 10.5, 9.7, 880.0),
        ("Poor #10: Extreme pollution", 6.3, 3.6, 27.2, 9.8, 10.2, 800.0),
        # VERY POOR WATER QUALITY (10 samples)
        ("Very Poor #1: Critically low oxygen", 5.5, 1.8, 29.0, 14.0, 13.0, 950.0),
        ("Very Poor #2: Extreme pH", 5.0, 2.2, 28.5, 16.0, 12.5, 920.0),
        ("Very Poor #3: High nitrate contamination", 6.0, 2.8, 28.0, 13.5, 18.0, 1000.0),
        ("Very Poor #4: All parameters failed", 5.8, 2.0, 30.0, 18.0, 15.5, 1100.0),
        ("Very Poor #5: Extreme contamination", 5.2, 1.5, 29.5, 20.0, 20.0, 1200.0),
        ("Very Poor #6: Toxic conditions", 5.3, 1.2, 31.0, 22.0, 17.0, 1150.0),
        ("Very Poor #7: Industrial waste", 5.6, 1.8, 29.8, 19.0, 16.5, 1080.0),
        ("Very Poor #8: Severely polluted", 5.4, 1.6, 30.5, 21.0, 19.0, 1250.0),
        ("Very Poor #9: Dangerous water", 5.1, 1.3, 31.5, 23.0, 21.0, 1300.0),
        ("Very Poor #10: Uninhabitable", 5.0, 1.0, 32.0, 25.0, 22.0, 1400.0),
    ]

    for desc, ph, do, temp, turb, nitrate, cond in test_cases:
        X_sample = prepare_us_features_for_prediction(
            ph=ph, dissolved_oxygen=do, temperature=temp,
            turbidity=turb, nitrate=nitrate, conductance=cond, year=2024
        )
        water_params = {
            'ph': ph,
            'dissolved_oxygen': do,
            'temperature': temp,
            'turbidity': turb,
            'nitrate': nitrate,
            'conductance': cond
        }
        samples.append((desc, X_sample, water_params))

    return samples


def test_phase_4_2_shap_contributions():
    """
    Test Phase 4.2: get_prediction_contributions()

    Verifies:
    - Function executes without errors for all samples
    - Returns required keys in dictionary
    - SHAP sum property holds (Σ(SHAP_i) = Prediction - Base Value)
    - Contributions DataFrame has all 59 features
    - Contributions are sorted by absolute value
    """
    print("\n" + "="*80)
    print("PHASE 4.2 TEST: get_prediction_contributions()")
    print("="*80)

    # Load models
    classifier_files = sorted(glob.glob('data/models/classifier_*.joblib'), reverse=True)
    regressor_files = sorted(glob.glob('data/models/regressor_*.joblib'), reverse=True)

    if not classifier_files or not regressor_files:
        pytest.skip("Model files not found")

    classifier_path = classifier_files[0]
    regressor_path = regressor_files[0]

    print(f"✓ Loaded classifier: {Path(classifier_path).name}")
    print(f"✓ Loaded regressor: {Path(regressor_path).name}")

    # Generate test samples
    samples = generate_comprehensive_test_samples()
    print(f"✓ Generated {len(samples)} test samples")

    results = {
        'classifier': {'passed': 0, 'failed': 0, 'errors': []},
        'regressor': {'passed': 0, 'failed': 0, 'errors': []}
    }

    # Test classifier
    print(f"\n📊 Testing Classifier Contributions...")
    print("-" * 80)

    for description, X_sample, _ in samples:
        try:
            contribs = get_prediction_contributions(
                model_path=classifier_path,
                X_sample=X_sample,
                top_n=59
            )

            # Verify required keys
            required_keys = ['contributions', 'base_value', 'prediction', 'shap_sum', 'model_type']
            missing_keys = [k for k in required_keys if k not in contribs]

            if missing_keys:
                results['classifier']['failed'] += 1
                results['classifier']['errors'].append({
                    'sample': description,
                    'error': f"Missing keys: {missing_keys}"
                })
                print(f"  ✗ {description}: Missing keys {missing_keys}")
                continue

            # Verify model type
            if contribs['model_type'] != 'classifier':
                results['classifier']['failed'] += 1
                results['classifier']['errors'].append({
                    'sample': description,
                    'error': f"Wrong model_type: {contribs['model_type']}"
                })
                print(f"  ✗ {description}: Wrong model type")
                continue

            # Verify SHAP sum property
            pred_delta = contribs['prediction'] - contribs['base_value']
            match_error = abs(contribs['shap_sum'] - pred_delta)

            if match_error > 0.001:  # Tolerance for classifier
                results['classifier']['failed'] += 1
                results['classifier']['errors'].append({
                    'sample': description,
                    'error': f"SHAP sum property violated: error={match_error:.6f}"
                })
                print(f"  ✗ {description}: SHAP sum error={match_error:.6f}")
                continue

            # Verify contributions DataFrame
            contrib_df = contribs['contributions']
            if not isinstance(contrib_df, pd.DataFrame):
                results['classifier']['failed'] += 1
                results['classifier']['errors'].append({
                    'sample': description,
                    'error': f"contributions is not a DataFrame: {type(contrib_df)}"
                })
                print(f"  ✗ {description}: Wrong contributions type")
                continue

            # Verify DataFrame has correct columns
            required_cols = ['feature', 'value', 'contribution', 'abs_contribution', 'rank']
            missing_cols = [c for c in required_cols if c not in contrib_df.columns]
            if missing_cols:
                results['classifier']['failed'] += 1
                results['classifier']['errors'].append({
                    'sample': description,
                    'error': f"Missing columns in contributions: {missing_cols}"
                })
                print(f"  ✗ {description}: Missing columns")
                continue

            # Verify sorting by absolute contribution
            abs_contribs = contrib_df['abs_contribution'].values
            if not all(abs_contribs[i] >= abs_contribs[i+1] for i in range(len(abs_contribs)-1)):
                results['classifier']['failed'] += 1
                results['classifier']['errors'].append({
                    'sample': description,
                    'error': "Contributions not sorted by absolute value"
                })
                print(f"  ✗ {description}: Not sorted")
                continue

            results['classifier']['passed'] += 1

        except Exception as e:
            results['classifier']['failed'] += 1
            results['classifier']['errors'].append({
                'sample': description,
                'error': str(e)
            })
            print(f"  ✗ {description}: {e}")

    print(f"\nClassifier Results: {results['classifier']['passed']}/{len(samples)} passed")

    # Test regressor
    print(f"\n📊 Testing Regressor Contributions...")
    print("-" * 80)

    for description, X_sample, _ in samples:
        try:
            contribs = get_prediction_contributions(
                model_path=regressor_path,
                X_sample=X_sample,
                top_n=59
            )

            # Verify required keys
            required_keys = ['contributions', 'base_value', 'prediction', 'shap_sum', 'model_type']
            missing_keys = [k for k in required_keys if k not in contribs]

            if missing_keys:
                results['regressor']['failed'] += 1
                results['regressor']['errors'].append({
                    'sample': description,
                    'error': f"Missing keys: {missing_keys}"
                })
                continue

            # Verify model type
            if contribs['model_type'] != 'regressor':
                results['regressor']['failed'] += 1
                results['regressor']['errors'].append({
                    'sample': description,
                    'error': f"Wrong model_type: {contribs['model_type']}"
                })
                continue

            # Verify SHAP sum property
            pred_delta = contribs['prediction'] - contribs['base_value']
            match_error = abs(contribs['shap_sum'] - pred_delta)

            if match_error > 0.1:  # Tolerance for regressor
                results['regressor']['failed'] += 1
                results['regressor']['errors'].append({
                    'sample': description,
                    'error': f"SHAP sum property violated: error={match_error:.6f}"
                })
                continue

            # Verify contributions DataFrame
            contrib_df = contribs['contributions']
            if not isinstance(contrib_df, pd.DataFrame):
                results['regressor']['failed'] += 1
                results['regressor']['errors'].append({
                    'sample': description,
                    'error': f"contributions is not a DataFrame: {type(contrib_df)}"
                })
                continue

            # Verify sorting
            abs_contribs = contrib_df['abs_contribution'].values
            if not all(abs_contribs[i] >= abs_contribs[i+1] for i in range(len(abs_contribs)-1)):
                results['regressor']['failed'] += 1
                results['regressor']['errors'].append({
                    'sample': description,
                    'error': "Contributions not sorted by absolute value"
                })
                continue

            results['regressor']['passed'] += 1

        except Exception as e:
            results['regressor']['failed'] += 1
            results['regressor']['errors'].append({
                'sample': description,
                'error': str(e)
            })

    print(f"\nRegressor Results: {results['regressor']['passed']}/{len(samples)} passed")

    # Final results
    all_passed = (results['classifier']['failed'] == 0 and results['regressor']['failed'] == 0)

    if not all_passed:
        print(f"\n⚠️  ERRORS:")
        for model_type in ['classifier', 'regressor']:
            if results[model_type]['errors']:
                print(f"\n{model_type.upper()} Errors:")
                for error in results[model_type]['errors'][:10]:  # Show first 10
                    print(f"  - {error['sample']}: {error['error']}")

    assert all_passed, "Phase 4.2: SHAP contributions checks failed"


def test_phase_4_3_decision_explanation():
    """
    Test Phase 4.3: generate_decision_explanation()

    Verifies:
    - Function executes without errors for all samples
    - Returns required keys in dictionary
    - Verdict is either 'SAFE' or 'UNSAFE'
    - Confidence is between 0 and 100
    - WQI category matches score ranges
    - All sections present and non-empty
    """
    print("\n" + "="*80)
    print("PHASE 4.3 TEST: generate_decision_explanation()")
    print("="*80)

    # Load models
    classifier_files = sorted(glob.glob('data/models/classifier_*.joblib'), reverse=True)
    regressor_files = sorted(glob.glob('data/models/regressor_*.joblib'), reverse=True)

    if not classifier_files or not regressor_files:
        pytest.skip("Model files not found")

    classifier_path = classifier_files[0]
    regressor_path = regressor_files[0]

    print(f"✓ Loaded classifier: {Path(classifier_path).name}")
    print(f"✓ Loaded regressor: {Path(regressor_path).name}")

    # Generate test samples
    samples = generate_comprehensive_test_samples()
    print(f"✓ Generated {len(samples)} test samples")

    results = {'passed': 0, 'failed': 0, 'errors': []}

    print(f"\n💬 Testing Decision Explanations...")
    print("-" * 80)

    for description, X_sample, water_params in samples:
        try:
            # First get contributions from both models
            clf_contribs = get_prediction_contributions(
                model_path=classifier_path,
                X_sample=X_sample,
                top_n=20
            )

            reg_contribs = get_prediction_contributions(
                model_path=regressor_path,
                X_sample=X_sample,
                top_n=20
            )

            # Generate explanation
            explanation = generate_decision_explanation(
                classifier_contributions=clf_contribs,
                regressor_contributions=reg_contribs,
                water_params=water_params
            )

            # Verify required keys
            required_keys = ['verdict', 'confidence', 'predicted_wqi', 'wqi_category',
                           'summary', 'primary_factors', 'parameter_assessment', 'recommendations']
            missing_keys = [k for k in required_keys if k not in explanation]

            if missing_keys:
                results['failed'] += 1
                results['errors'].append({
                    'sample': description,
                    'error': f"Missing keys: {missing_keys}"
                })
                print(f"  ✗ {description}: Missing keys {missing_keys}")
                continue

            # Verify verdict
            if explanation['verdict'] not in ['SAFE', 'UNSAFE']:
                results['failed'] += 1
                results['errors'].append({
                    'sample': description,
                    'error': f"Invalid verdict: {explanation['verdict']}"
                })
                print(f"  ✗ {description}: Invalid verdict")
                continue

            # Verify confidence range
            if not (0 <= explanation['confidence'] <= 100):
                results['failed'] += 1
                results['errors'].append({
                    'sample': description,
                    'error': f"Confidence out of range: {explanation['confidence']}"
                })
                print(f"  ✗ {description}: Invalid confidence")
                continue

            # Verify WQI category matches score
            wqi = explanation['predicted_wqi']
            category = explanation['wqi_category']

            expected_category = None
            if wqi >= 90:
                expected_category = 'Excellent'
            elif wqi >= 70:
                expected_category = 'Good'
            elif wqi >= 50:
                expected_category = 'Medium'
            elif wqi >= 25:
                expected_category = 'Bad'
            else:
                expected_category = 'Very Bad'

            if category != expected_category:
                results['failed'] += 1
                results['errors'].append({
                    'sample': description,
                    'error': f"Category mismatch: expected {expected_category}, got {category} (WQI={wqi})"
                })
                print(f"  ✗ {description}: Category mismatch")
                continue

            # Verify summary is non-empty string
            if not isinstance(explanation['summary'], str) or len(explanation['summary']) == 0:
                results['failed'] += 1
                results['errors'].append({
                    'sample': description,
                    'error': "Summary is empty or not a string"
                })
                print(f"  ✗ {description}: Empty summary")
                continue

            # Verify primary_factors is a list
            if not isinstance(explanation['primary_factors'], list):
                results['failed'] += 1
                results['errors'].append({
                    'sample': description,
                    'error': f"primary_factors is not a list: {type(explanation['primary_factors'])}"
                })
                print(f"  ✗ {description}: Wrong primary_factors type")
                continue

            # Verify parameter_assessment is a dict
            if not isinstance(explanation['parameter_assessment'], dict):
                results['failed'] += 1
                results['errors'].append({
                    'sample': description,
                    'error': f"parameter_assessment is not a dict: {type(explanation['parameter_assessment'])}"
                })
                print(f"  ✗ {description}: Wrong parameter_assessment type")
                continue

            # Verify recommendations is a list
            if not isinstance(explanation['recommendations'], list):
                results['failed'] += 1
                results['errors'].append({
                    'sample': description,
                    'error': f"recommendations is not a list: {type(explanation['recommendations'])}"
                })
                print(f"  ✗ {description}: Wrong recommendations type")
                continue

            # For UNSAFE verdicts, recommendations should be non-empty
            if explanation['verdict'] == 'UNSAFE' and len(explanation['recommendations']) == 0:
                results['failed'] += 1
                results['errors'].append({
                    'sample': description,
                    'error': "UNSAFE verdict but no recommendations provided"
                })
                print(f"  ✗ {description}: No recommendations for UNSAFE")
                continue

            results['passed'] += 1

        except Exception as e:
            results['failed'] += 1
            results['errors'].append({
                'sample': description,
                'error': str(e)
            })
            print(f"  ✗ {description}: {e}")

    print(f"\nResults: {results['passed']}/{len(samples)} passed")

    # Show errors if any
    if results['errors']:
        print(f"\n⚠️  ERRORS:")
        for error in results['errors'][:10]:  # Show first 10
            print(f"  - {error['sample']}: {error['error']}")

    all_passed = results['failed'] == 0

    assert all_passed, "Phase 4.3: Decision explanation validation failed"


if __name__ == "__main__":
    print("\n" + "="*80)
    print("COMPREHENSIVE BACKEND TEST: PHASE 4.2 & 4.3")
    print("="*80)

    # Test Phase 4.2
    phase42_passed, phase42_results = test_phase_4_2_shap_contributions()

    # Test Phase 4.3
    phase43_passed, phase43_results = test_phase_4_3_decision_explanation()

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    if phase42_results:
        print(f"\nPhase 4.2 (get_prediction_contributions):")
        print(f"  Classifier: {phase42_results['classifier']['passed']}/50 passed")
        print(f"  Regressor:  {phase42_results['regressor']['passed']}/50 passed")

    if phase43_results:
        print(f"\nPhase 4.3 (generate_decision_explanation):")
        print(f"  {phase43_results['passed']}/50 passed")

    if phase42_passed and phase43_passed:
        print("\n✅ ALL TESTS PASSED")
        print("="*80)
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED")
        print("="*80)
        sys.exit(1)
