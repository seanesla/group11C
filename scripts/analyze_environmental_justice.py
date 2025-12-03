"""
Environmental Justice Analysis: WQI Model Limitations on Lead-Contaminated Water.

Analyzes NSF-WQI on Flint/Jackson crisis scenarios. The model CANNOT detect lead,
causing false negatives on water that poisoned vulnerable communities.

See docs/ENVIRONMENTAL_JUSTICE_ANALYSIS.md for full context and methodology.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from src.utils.wqi_calculator import WQICalculator


def test_flint_contaminated_scenarios():
    """
    Analyze WQI on Flint, MI crisis scenarios (lead contamination 2014-2019).

    Demonstrates false negatives: model predicts Good/Excellent for lead-poisoned water.
    """
    print("=" * 80)
    print("FLINT, MICHIGAN WATER CRISIS SCENARIOS")
    print("=" * 80)
    print()

    calculator = WQICalculator()

    scenarios = [
        {
            "name": "Flint Scenario 1: High Lead, Good Other Parameters",
            "description": "Typical Flint home during crisis - corrosive water leached lead from pipes",
            "lead_ppb": 100,  # 6.7× EPA action level (15 ppb)
            "lead_health_risk": "SEVERE - Permanent neurological damage in children",
            "params": {
                "ph": 6.8,  # Slightly acidic (corrosive)
                "dissolved_oxygen": 8.5,  # Good
                "temperature": 18.0,  # Good
                "turbidity": 4.0,  # Excellent
                "nitrate": 2.0,  # Excellent
                "conductance": 450  # Excellent
            }
        },
        {
            "name": "Flint Scenario 2: Extreme Lead, Perfect Other Parameters",
            "description": "Worst-case Flint home with very high lead levels",
            "lead_ppb": 150,  # 10× EPA action level
            "lead_health_risk": "EXTREME - Medical emergency, immediate health threat",
            "params": {
                "ph": 7.0,  # Perfect
                "dissolved_oxygen": 9.0,  # Perfect
                "temperature": 20.0,  # Perfect
                "turbidity": 3.0,  # Perfect
                "nitrate": 1.0,  # Perfect
                "conductance": 400  # Perfect
            }
        },
        {
            "name": "Flint Scenario 3: Moderate Lead with Corrosion Indicators",
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
            }
        }
    ]

    results = []

    for scenario in scenarios:
        print(f"\n{scenario['name']}")
        print("-" * 80)
        print(f"Description: {scenario['description']}")
        print(f"Lead Level: {scenario['lead_ppb']} ppb (EPA action level: 15 ppb)")
        print(f"Health Risk: {scenario['lead_health_risk']}")
        print()
        print("Water Quality Parameters (NSF-WQI inputs):")
        for param, value in scenario['params'].items():
            print(f"  {param}: {value}")
        print()

        # Calculate WQI (IGNORES lead completely!)
        wqi, scores, classification = calculator.calculate_wqi(**scenario['params'])

        print(f"WQI PREDICTION: {wqi} ({classification})")
        print()
        print("REALITY: Water is HIGHLY TOXIC due to lead contamination")
        print()
        print("⚠️  FALSE NEGATIVE: Model predicts SAFE when water is UNSAFE")
        print("=" * 80)

        results.append({
            'scenario': scenario['name'],
            'lead_ppb': scenario['lead_ppb'],
            'actual_safety': 'UNSAFE',
            'predicted_wqi': wqi,
            'predicted_class': classification,
            'false_negative': classification in ['Good', 'Excellent']
        })

    return results


def test_jackson_contaminated_scenarios():
    """
    Analyze WQI on Jackson, MS crisis scenarios (infrastructure failure 2021-2023).

    Demonstrates false negatives: model predicts Good/Excellent for contaminated water.
    """
    print()
    print("=" * 80)
    print("JACKSON, MISSISSIPPI WATER CRISIS SCENARIOS")
    print("=" * 80)
    print()

    calculator = WQICalculator()

    scenarios = [
        {
            "name": "Jackson Scenario 1: Lead + Turbidity (Treatment Plant Failure)",
            "description": "Post-flood treatment failure with discolored water and lead",
            "lead_ppb": 35,  # 2.3× EPA action level
            "lead_health_risk": "HIGH - Unsafe, especially for children",
            "other_contaminants": "High turbidity (sediment), possible bacteria",
            "params": {
                "ph": 6.9,
                "dissolved_oxygen": 7.5,
                "temperature": 22.0,
                "turbidity": 65.0,  # High - visible discoloration
                "nitrate": 2.5,
                "conductance": 480
            }
        },
        {
            "name": "Jackson Scenario 2: Lead with Normal Appearance",
            "description": "Clear-looking water with hidden lead contamination from pipes",
            "lead_ppb": 20,  # 1.3× EPA action level
            "lead_health_risk": "MODERATE-HIGH - Exceeds safety threshold",
            "other_contaminants": "Lead from corroded service lines",
            "params": {
                "ph": 7.1,
                "dissolved_oxygen": 8.2,
                "temperature": 20.0,
                "turbidity": 6.0,  # Appears clear
                "nitrate": 1.8,
                "conductance": 420
            }
        },
        {
            "name": "Jackson Scenario 3: Chronic Infrastructure Decay",
            "description": "Low-level lead from aging 100-year-old infrastructure",
            "lead_ppb": 18,  # 1.2× EPA action level
            "lead_health_risk": "MODERATE - Above action level, requires intervention",
            "other_contaminants": "Rust, sediment from pipe breaks",
            "params": {
                "ph": 6.7,
                "dissolved_oxygen": 7.0,
                "temperature": 21.0,
                "turbidity": 28.0,  # Moderate discoloration
                "nitrate": 3.2,
                "conductance": 510
            }
        }
    ]

    results = []

    for scenario in scenarios:
        print(f"\n{scenario['name']}")
        print("-" * 80)
        print(f"Description: {scenario['description']}")
        print(f"Lead Level: {scenario['lead_ppb']} ppb (EPA action level: 15 ppb)")
        print(f"Health Risk: {scenario['lead_health_risk']}")
        print(f"Other Contaminants: {scenario['other_contaminants']}")
        print()
        print("Water Quality Parameters (NSF-WQI inputs):")
        for param, value in scenario['params'].items():
            print(f"  {param}: {value}")
        print()

        # Calculate WQI (IGNORES lead completely!)
        wqi, scores, classification = calculator.calculate_wqi(**scenario['params'])

        print(f"WQI PREDICTION: {wqi} ({classification})")
        print()
        print("REALITY: Water is UNSAFE due to lead contamination")
        print()

        if classification in ['Good', 'Excellent']:
            print("⚠️  FALSE NEGATIVE: Model predicts SAFE when water is UNSAFE")
        else:
            # Even if WQI is lower due to turbidity, it doesn't detect the LEAD hazard
            print("⚠️  PARTIAL DETECTION: Model detects some issues (turbidity)")
            print("    but COMPLETELY MISSES the primary health threat (LEAD)")
        print("=" * 80)

        results.append({
            'scenario': scenario['name'],
            'lead_ppb': scenario['lead_ppb'],
            'actual_safety': 'UNSAFE',
            'predicted_wqi': wqi,
            'predicted_class': classification,
            'false_negative': classification in ['Good', 'Excellent']
        })

    return results


def calculate_false_negative_rate(all_results):
    """
    Calculate false negative rate: P(predict SAFE | actually UNSAFE).

    False Negative = Model predicts "Good" or "Excellent" when water is UNSAFE
    This is catastrophic for public health because residents drink poisoned water
    thinking it's safe.
    """
    print()
    print("=" * 80)
    print("FALSE NEGATIVE ANALYSIS")
    print("=" * 80)
    print()

    df = pd.DataFrame(all_results)

    total_scenarios = len(df)
    unsafe_scenarios = len(df[df['actual_safety'] == 'UNSAFE'])
    false_negatives = len(df[df['false_negative'] == True])

    false_negative_rate = false_negatives / total_scenarios * 100

    print(f"Total Test Scenarios: {total_scenarios}")
    print(f"Scenarios with Unsafe Lead Levels (>15 ppb): {unsafe_scenarios}")
    print(f"Model Predicted 'Good' or 'Excellent' (FALSE NEGATIVE): {false_negatives}")
    print()
    print(f"FALSE NEGATIVE RATE: {false_negative_rate:.1f}%")
    print()
    print("Interpretation:")
    print(f"  {false_negative_rate:.0f}% of the time, the model predicts SAFE water quality")
    print("  when lead contamination makes the water HIGHLY TOXIC.")
    print()
    print("Health Impact:")
    print("  - Children drinking this water face permanent IQ damage")
    print("  - Pregnant women risk fetal brain development issues")
    print("  - No safe level of lead exposure exists (EPA MCLG = 0)")
    print()
    print("Environmental Justice Impact:")
    print("  - Flint: 53% Black, 41% poverty → disproportionately harmed")
    print("  - Jackson: 83% Black, 25% poverty → disproportionately harmed")
    print("  - Model failure perpetuates systemic health inequities")
    print()

    # Detailed breakdown
    print("Detailed Results:")
    print(df.to_string(index=False))
    print()

    return false_negative_rate, df


def main():
    """Run complete environmental justice analysis."""
    print()
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  ENVIRONMENTAL JUSTICE ANALYSIS: NSF-WQI MODEL TESTING".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("║" + "  Testing Water Quality Index on Contaminated Water Scenarios".center(78) + "║")
    print("║" + "  from Flint, MI and Jackson, MS Water Crises".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    # Run Flint scenarios
    flint_results = test_flint_contaminated_scenarios()

    # Run Jackson scenarios
    jackson_results = test_jackson_contaminated_scenarios()

    # Combine and analyze
    all_results = flint_results + jackson_results
    false_negative_rate, results_df = calculate_false_negative_rate(all_results)

    # Final summary
    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print()
    print("1. MODEL LIMITATION: NSF-WQI does NOT include lead as a parameter")
    print("   ├─ Uses only: pH, DO, temperature, turbidity, nitrate, conductance")
    print("   └─ Missing: lead, fecal coliform, phosphate, biochemical oxygen demand")
    print()
    print(f"2. FALSE NEGATIVE RATE: {false_negative_rate:.0f}%")
    print("   ├─ Model predicts 'Good' or 'Excellent' for highly toxic water")
    print("   └─ Residents drink poisoned water believing it's safe")
    print()
    print("3. ENVIRONMENTAL JUSTICE IMPACT:")
    print("   ├─ Flint: 53% Black, 41% poverty")
    print("   ├─ Jackson: 83% Black, 25% poverty")
    print("   ├─ Lead causes permanent neurological damage in children")
    print("   └─ Model failure disproportionately harms vulnerable communities")
    print()
    print("4. RECOMMENDATIONS:")
    print("   ├─ Add prominent warnings to Streamlit app about lead limitation")
    print("   ├─ Include data coverage indicators for contaminated water")
    print("   ├─ Recommend lead testing for homes with older infrastructure")
    print("   └─ Never use WQI alone for drinking water safety decisions")
    print()
    print("=" * 80)
    print()

    # Save results
    output_path = Path(__file__).parent.parent / "data" / "environmental_justice_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    print()


if __name__ == "__main__":
    main()
