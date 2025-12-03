"""
Environmental Justice Analysis: Complete System Testing (WQI + ML Models).

Tests NSF-WQI calculator, ML classifier, and ML regressor on Flint/Jackson crisis
scenarios. All components show 100% false negative rate on lead contamination.

See docs/ENVIRONMENTAL_JUSTICE_ANALYSIS.md for full context and methodology.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from src.utils.wqi_calculator import WQICalculator
from src.models.model_utils import load_latest_models
from src.preprocessing.us_data_features import prepare_us_features_for_prediction


# Define contaminated water scenarios
FLINT_JACKSON_SCENARIOS = [
    {
        "name": "Flint Scenario 1: High Lead, Good Other Parameters",
        "location": "Flint, MI",
        "description": "Typical Flint home during crisis - corrosive water leached lead from pipes",
        "lead_ppb": 100,  # 6.7× EPA action level (15 ppb)
        "lead_health_risk": "SEVERE - Permanent neurological damage in children",
        "params": {
            "ph": 6.8,
            "dissolved_oxygen": 8.5,
            "temperature": 18.0,
            "turbidity": 4.0,
            "nitrate": 2.0,
            "conductance": 450
        },
        "year": 2015  # Peak of crisis
    },
    {
        "name": "Flint Scenario 2: Extreme Lead, Perfect Other Parameters",
        "location": "Flint, MI",
        "description": "Worst-case Flint home with very high lead levels",
        "lead_ppb": 150,  # 10× EPA action level
        "lead_health_risk": "EXTREME - Medical emergency, immediate health threat",
        "params": {
            "ph": 7.0,
            "dissolved_oxygen": 9.0,
            "temperature": 20.0,
            "turbidity": 3.0,
            "nitrate": 1.0,
            "conductance": 400
        },
        "year": 2015
    },
    {
        "name": "Flint Scenario 3: Moderate Lead with Corrosion Indicators",
        "location": "Flint, MI",
        "description": "Early crisis detection - lead with low pH from corrosive water",
        "lead_ppb": 25,  # 1.7× EPA action level
        "lead_health_risk": "HIGH - Unsafe for children and pregnant women",
        "params": {
            "ph": 6.2,  # Low (corrosive, leaches lead)
            "dissolved_oxygen": 7.8,
            "temperature": 17.0,
            "turbidity": 8.0,
            "nitrate": 3.0,
            "conductance": 520
        },
        "year": 2014  # Early crisis
    },
    {
        "name": "Jackson Scenario 1: Lead + Turbidity (Treatment Plant Failure)",
        "location": "Jackson, MS",
        "description": "Post-flood treatment failure with discolored water and lead",
        "lead_ppb": 35,  # 2.3× EPA action level
        "lead_health_risk": "HIGH - Unsafe, especially for children",
        "params": {
            "ph": 6.9,
            "dissolved_oxygen": 7.5,
            "temperature": 22.0,
            "turbidity": 65.0,  # High - visible discoloration
            "nitrate": 2.5,
            "conductance": 480
        },
        "year": 2022  # Crisis year
    },
    {
        "name": "Jackson Scenario 2: Lead with Normal Appearance",
        "location": "Jackson, MS",
        "description": "Clear-looking water with hidden lead contamination from pipes",
        "lead_ppb": 20,  # 1.3× EPA action level
        "lead_health_risk": "MODERATE-HIGH - Exceeds safety threshold",
        "params": {
            "ph": 7.1,
            "dissolved_oxygen": 8.2,
            "temperature": 20.0,
            "turbidity": 6.0,  # Appears clear
            "nitrate": 1.8,
            "conductance": 420
        },
        "year": 2023
    },
    {
        "name": "Jackson Scenario 3: Chronic Infrastructure Decay",
        "location": "Jackson, MS",
        "description": "Low-level lead from aging 100-year-old infrastructure",
        "lead_ppb": 18,  # 1.2× EPA action level
        "lead_health_risk": "MODERATE - Above action level, requires intervention",
        "params": {
            "ph": 6.7,
            "dissolved_oxygen": 7.0,
            "temperature": 21.0,
            "turbidity": 28.0,  # Moderate discoloration
            "nitrate": 3.2,
            "conductance": 510
        },
        "year": 2023
    }
]


def test_wqi_calculator(scenarios):
    """Test NSF-WQI Calculator on contaminated scenarios."""
    print("\n" + "=" * 80)
    print("COMPONENT 1: NSF-WQI CALCULATOR TESTING")
    print("=" * 80)
    print("\nTesting the mathematical WQI formula on contaminated water scenarios...")
    print()

    calculator = WQICalculator()
    results = []

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n[{i}/6] {scenario['name']}")
        print("-" * 80)
        print(f"Lead Level: {scenario['lead_ppb']} ppb (EPA action level: 15 ppb)")
        print(f"Health Risk: {scenario['lead_health_risk']}")

        # Calculate WQI (IGNORES lead!)
        wqi, scores, classification = calculator.calculate_wqi(**scenario['params'])

        print(f"\nWQI Prediction: {wqi} ({classification})")
        print(f"Reality: UNSAFE due to lead contamination")

        false_negative = classification in ['Good', 'Excellent']
        if false_negative:
            print("⚠️  FALSE NEGATIVE: Predicts SAFE when UNSAFE")
        else:
            print("⚠️  May detect some issues, but MISSES lead contamination")

        results.append({
            'scenario': scenario['name'],
            'component': 'WQI Calculator',
            'lead_ppb': scenario['lead_ppb'],
            'predicted_wqi': wqi,
            'predicted_class': classification,
            'false_negative': false_negative
        })

    return results


def test_ml_models(scenarios):
    """Test ML Classifier and Regressor on contaminated scenarios."""
    print("\n\n" + "=" * 80)
    print("COMPONENT 2 & 3: ML MODELS TESTING")
    print("=" * 80)
    print("\nLoading production ML models...")

    # Load models
    classifier, regressor = load_latest_models()

    if classifier is None or regressor is None:
        print("\n❌ ERROR: Models not found. Run train_models.py first.")
        return []

    print(f"✓ Classifier loaded ({len(classifier.feature_names)} features)")
    print(f"✓ Regressor loaded ({len(regressor.feature_names)} features)")
    print()

    results = []

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n[{i}/6] {scenario['name']}")
        print("-" * 80)
        print(f"Lead Level: {scenario['lead_ppb']} ppb (EPA action level: 15 ppb)")
        print(f"Health Risk: {scenario['lead_health_risk']}")

        # Prepare features
        features_df = prepare_us_features_for_prediction(
            ph=scenario['params']['ph'],
            dissolved_oxygen=scenario['params']['dissolved_oxygen'],
            temperature=scenario['params']['temperature'],
            turbidity=scenario['params']['turbidity'],
            nitrate=scenario['params']['nitrate'],
            conductance=scenario['params']['conductance'],
            year=scenario['year']
        )

        # ML Classifier prediction
        is_safe_pred = classifier.predict(features_df)[0]
        safe_proba = classifier.predict_proba(features_df)[0][1]  # Probability of SAFE

        # ML Regressor prediction
        wqi_pred = regressor.predict(features_df)[0]

        print(f"\nML Classifier: {'SAFE' if is_safe_pred else 'UNSAFE'} (P(safe) = {safe_proba:.2%})")
        print(f"ML Regressor: WQI = {wqi_pred:.2f}")
        print(f"Reality: UNSAFE due to lead contamination")

        classifier_fn = is_safe_pred == 1  # Predicts SAFE when actually UNSAFE
        regressor_fn = wqi_pred >= 70  # Predicts good WQI when actually UNSAFE

        if classifier_fn:
            print("⚠️  CLASSIFIER FALSE NEGATIVE: Predicts SAFE when UNSAFE")
        if regressor_fn:
            print("⚠️  REGRESSOR FALSE NEGATIVE: Predicts WQI ≥ 70 when UNSAFE")

        results.append({
            'scenario': scenario['name'],
            'component': 'ML Classifier',
            'lead_ppb': scenario['lead_ppb'],
            'predicted_safe': is_safe_pred,
            'safe_probability': safe_proba,
            'false_negative': classifier_fn
        })

        results.append({
            'scenario': scenario['name'],
            'component': 'ML Regressor',
            'lead_ppb': scenario['lead_ppb'],
            'predicted_wqi': wqi_pred,
            'false_negative': regressor_fn
        })

    return results


def analyze_combined_results(wqi_results, ml_results):
    """Analyze false negative rates across all components."""
    print("\n\n" + "=" * 80)
    print("COMPREHENSIVE FALSE NEGATIVE ANALYSIS")
    print("=" * 80)

    # WQI Calculator
    wqi_df = pd.DataFrame(wqi_results)
    wqi_fn_rate = wqi_df['false_negative'].mean() * 100

    # ML Classifier
    clf_df = pd.DataFrame([r for r in ml_results if r['component'] == 'ML Classifier'])
    clf_fn_rate = clf_df['false_negative'].mean() * 100

    # ML Regressor
    reg_df = pd.DataFrame([r for r in ml_results if r['component'] == 'ML Regressor'])
    reg_fn_rate = reg_df['false_negative'].mean() * 100

    print(f"\nFalse Negative Rates (Predicting SAFE when UNSAFE):")
    print(f"  WQI Calculator:  {wqi_fn_rate:.1f}% ({wqi_df['false_negative'].sum()}/6 scenarios)")
    print(f"  ML Classifier:   {clf_fn_rate:.1f}% ({clf_df['false_negative'].sum()}/6 scenarios)")
    print(f"  ML Regressor:    {reg_fn_rate:.1f}% ({reg_df['false_negative'].sum()}/6 scenarios)")
    print()

    # Detailed comparison
    print("Detailed Results by Scenario:")
    print()
    print(f"{'Scenario':<50} {'Lead (ppb)':<12} {'WQI Calc':<15} {'ML Clf':<15} {'ML Reg':<15}")
    print("-" * 110)

    for i, scenario in enumerate(FLINT_JACKSON_SCENARIOS):
        name = scenario['name'][:47] + "..." if len(scenario['name']) > 50 else scenario['name']
        lead = scenario['lead_ppb']

        wqi_result = wqi_results[i]
        clf_result = [r for r in ml_results if r['scenario'] == scenario['name'] and r['component'] == 'ML Classifier'][0]
        reg_result = [r for r in ml_results if r['scenario'] == scenario['name'] and r['component'] == 'ML Regressor'][0]

        wqi_status = "❌ FN" if wqi_result['false_negative'] else "✓ Detect"
        clf_status = "❌ FN" if clf_result['false_negative'] else "✓ Detect"
        reg_status = "❌ FN" if reg_result['false_negative'] else "✓ Detect"

        print(f"{name:<50} {lead:<12} {wqi_status:<15} {clf_status:<15} {reg_status:<15}")

    print()

    # Key findings
    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print()

    if wqi_fn_rate == 100 and clf_fn_rate == 100 and reg_fn_rate == 100:
        print("⚠️  CRITICAL: ALL THREE COMPONENTS show 100% false negative rate!")
        print("   None of the models detect lead contamination.")
        print()
    elif wqi_fn_rate == 100:
        print(f"1. WQI Calculator: 100% false negative rate (as expected)")
        print(f"2. ML Classifier: {clf_fn_rate:.0f}% false negative rate")
        print(f"3. ML Regressor: {reg_fn_rate:.0f}% false negative rate")
        print()
        if clf_fn_rate < 100 or reg_fn_rate < 100:
            print("✓ ML models show SOME improvement over WQI calculator")
        else:
            print("⚠️  ML models inherit WQI limitation (trained on WQI labels)")

    print("ROOT CAUSE:")
    print("  - NSF-WQI methodology does NOT include lead as a parameter")
    print("  - ML models are trained on WQI-derived labels (is_safe = WQI ≥ 70)")
    print("  - If WQI can't detect lead, neither can models trained on it")
    print()

    print("ENVIRONMENTAL JUSTICE IMPACT:")
    print("  - Flint residents: 53% Black, 41% poverty")
    print("  - Jackson residents: 83% Black, 25% poverty")
    print("  - Lead causes permanent IQ damage in children")
    print("  - System limitation disproportionately harms vulnerable communities")
    print()

    return {
        'wqi_fn_rate': wqi_fn_rate,
        'classifier_fn_rate': clf_fn_rate,
        'regressor_fn_rate': reg_fn_rate
    }


def main():
    """Run complete environmental justice analysis on all system components."""
    print()
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  COMPLETE ENVIRONMENTAL JUSTICE ANALYSIS".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("║" + "  Testing ALL THREE Production System Components".center(78) + "║")
    print("║" + "  on Flint, MI and Jackson, MS Water Crises".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")

    # Test 1: WQI Calculator
    wqi_results = test_wqi_calculator(FLINT_JACKSON_SCENARIOS)

    # Test 2 & 3: ML Models
    ml_results = test_ml_models(FLINT_JACKSON_SCENARIOS)

    # Combined analysis
    fn_rates = analyze_combined_results(wqi_results, ml_results)

    # Save results
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    wqi_df = pd.DataFrame(wqi_results)
    ml_df = pd.DataFrame(ml_results)

    wqi_df.to_csv(output_dir / "environmental_justice_wqi_results.csv", index=False)
    ml_df.to_csv(output_dir / "environmental_justice_ml_results.csv", index=False)

    print("RESULTS SAVED:")
    print(f"  {output_dir / 'environmental_justice_wqi_results.csv'}")
    print(f"  {output_dir / 'environmental_justice_ml_results.csv'}")
    print()

    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Add lead-specific warning to Streamlit app")
    print("2. Add data coverage indicators")
    print("3. Document existing system disclaimers")
    print("4. Write comprehensive environmental justice analysis (10+ pages)")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
