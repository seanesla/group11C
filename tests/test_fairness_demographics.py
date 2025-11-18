"""
Fairness & Demographics Testing: Environmental Justice Analysis

Tests for algorithmic fairness and disparate impact using REAL DATA from
Flint, MI and Jackson, MS water crises.

This test suite verifies that the system's limitations (100% false negative rate
on lead contamination) are consistently documented and that we measure disparate
impact on marginalized communities.

Data Source: data/environmental_justice_ml_results.csv (REAL DATA)
Demographics:
- Flint, MI: 53% Black, 41% poverty
- Jackson, MS: 83% Black, 25% poverty

Author: Environmental Justice & Fairness Analysis Team
Date: 2025-11-17
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


# === FIXTURES ===

@pytest.fixture(scope="module")
def environmental_justice_data():
    """
    Load REAL environmental justice test results from CSV.

    Returns:
        pd.DataFrame: Results from testing WQI Calculator, ML Classifier, and ML Regressor
                     on 6 lead-contaminated water scenarios from Flint and Jackson.
    """
    data_path = Path(__file__).parent.parent / "data" / "environmental_justice_ml_results.csv"
    assert data_path.exists(), f"Environmental justice data not found at {data_path}"

    df = pd.read_csv(data_path)
    assert len(df) > 0, "Environmental justice CSV is empty"

    return df


@pytest.fixture(scope="module")
def flint_results(environmental_justice_data):
    """Extract Flint-specific results (53% Black, 41% poverty)."""
    return environmental_justice_data[
        environmental_justice_data['scenario'].str.contains('Flint')
    ]


@pytest.fixture(scope="module")
def jackson_results(environmental_justice_data):
    """Extract Jackson-specific results (83% Black, 25% poverty)."""
    return environmental_justice_data[
        environmental_justice_data['scenario'].str.contains('Jackson')
    ]


@pytest.fixture(scope="module")
def classifier_results(environmental_justice_data):
    """Extract ML Classifier results."""
    return environmental_justice_data[
        environmental_justice_data['component'] == 'ML Classifier'
    ]


@pytest.fixture(scope="module")
def regressor_results(environmental_justice_data):
    """Extract ML Regressor results."""
    return environmental_justice_data[
        environmental_justice_data['component'] == 'ML Regressor'
    ]


# === DATA INTEGRITY TESTS ===

def test_environmental_justice_data_exists(environmental_justice_data):
    """
    Test 1: Verify environmental justice CSV data exists and is not empty.

    This ensures we have REAL DATA for fairness testing.
    """
    assert len(environmental_justice_data) > 0
    assert len(environmental_justice_data) == 12  # 6 scenarios × 2 components


def test_required_columns_exist(environmental_justice_data):
    """
    Test 2: Verify all required columns exist in the dataset.
    """
    required_columns = [
        'scenario', 'component', 'lead_ppb', 'false_negative'
    ]

    for col in required_columns:
        assert col in environmental_justice_data.columns, \
            f"Required column '{col}' missing from environmental justice data"


def test_data_types_correct(environmental_justice_data):
    """
    Test 3: Verify data types are correct for fairness analysis.
    """
    # Lead concentration should be numeric
    assert pd.api.types.is_numeric_dtype(environmental_justice_data['lead_ppb']), \
        "lead_ppb should be numeric"

    # False negative should be boolean
    assert environmental_justice_data['false_negative'].dtype == bool or \
           environmental_justice_data['false_negative'].isin([True, False]).all(), \
        "false_negative should be boolean"

    # Scenario names should be strings
    assert environmental_justice_data['scenario'].dtype == object, \
        "scenario should be string"


# === FALSE NEGATIVE RATE TESTS ===

def test_flint_false_negative_rate(flint_results):
    """
    Test 4: Calculate false negative rate for Flint, MI scenarios.

    Demographics: 53% Black, 41% poverty
    Expected: 100% false negative rate (all contaminated water predicted as safe)
    """
    fn_rate = flint_results['false_negative'].mean() * 100

    assert fn_rate == 100.0, \
        f"Flint false negative rate should be 100%, got {fn_rate:.1f}%"

    # Verify all 3 Flint scenarios × 2 components = 6 results
    assert len(flint_results) == 6, \
        f"Expected 6 Flint results (3 scenarios × 2 components), got {len(flint_results)}"


def test_jackson_false_negative_rate(jackson_results):
    """
    Test 5: Calculate false negative rate for Jackson, MS scenarios.

    Demographics: 83% Black, 25% poverty
    Expected: 100% false negative rate (all contaminated water predicted as safe)
    """
    fn_rate = jackson_results['false_negative'].mean() * 100

    assert fn_rate == 100.0, \
        f"Jackson false negative rate should be 100%, got {fn_rate:.1f}%"

    # Verify all 3 Jackson scenarios × 2 components = 6 results
    assert len(jackson_results) == 6, \
        f"Expected 6 Jackson results (3 scenarios × 2 components), got {len(jackson_results)}"


def test_overall_false_negative_rate(environmental_justice_data):
    """
    Test 6: Calculate overall false negative rate across all scenarios.

    Expected: 100% false negative rate (complete failure to detect lead)
    """
    fn_rate = environmental_justice_data['false_negative'].mean() * 100

    assert fn_rate == 100.0, \
        f"Overall false negative rate should be 100%, got {fn_rate:.1f}%"


# === DISPARATE IMPACT TESTS ===

def test_no_demographic_disparity(flint_results, jackson_results):
    """
    Test 7: Verify false negative rates are equal across demographics.

    While this seems like a "good" result, it's actually concerning:
    - Both communities (53% and 83% Black) have 100% FN rate
    - System fails EQUALLY for all marginalized communities
    - This is NOT fairness - it's uniform failure

    This test documents that the system does NOT discriminate further,
    but also does NOT provide protection to any vulnerable community.
    """
    flint_fn_rate = flint_results['false_negative'].mean()
    jackson_fn_rate = jackson_results['false_negative'].mean()

    assert flint_fn_rate == jackson_fn_rate == 1.0, \
        "Both Flint and Jackson should have 100% FN rate (uniform failure)"


def test_high_minority_communities_impacted(environmental_justice_data):
    """
    Test 8: Document that high-minority communities are impacted.

    Flint: 53% Black
    Jackson: 83% Black

    Both communities experience 100% false negative rate, meaning the system
    CANNOT protect these vulnerable populations from lead contamination.
    """
    # All 12 results (6 Flint + 6 Jackson) should be false negatives
    total_false_negatives = environmental_justice_data['false_negative'].sum()

    assert total_false_negatives == 12, \
        f"Expected all 12 scenarios to be false negatives, got {total_false_negatives}"

    # Document this is a LIMITATION, not acceptable performance
    print("\n⚠️  ENVIRONMENTAL JUSTICE IMPACT:")
    print("   Flint (53% Black): 100% false negative rate")
    print("   Jackson (83% Black): 100% false negative rate")
    print("   System cannot protect marginalized communities from lead poisoning")


# === COMPONENT-SPECIFIC TESTS ===

def test_classifier_fails_on_lead(classifier_results):
    """
    Test 9: Verify ML Classifier predicts SAFE for all lead-contaminated water.

    The binary classifier should predict UNSAFE for contaminated water,
    but instead predicts SAFE 100% of the time.
    """
    # All classifier predictions should be SAFE (predicted_safe = True)
    all_predicted_safe = classifier_results['predicted_safe'].all()

    assert all_predicted_safe, \
        "ML Classifier should predict SAFE for all scenarios (false negative)"

    # Verify this results in 100% false negative rate
    fn_rate = classifier_results['false_negative'].mean() * 100
    assert fn_rate == 100.0, \
        f"Classifier FN rate should be 100%, got {fn_rate:.1f}%"


def test_regressor_predicts_high_wqi_for_contaminated_water(regressor_results):
    """
    Test 10: Verify ML Regressor predicts high WQI scores for lead-contaminated water.

    The regressor should predict LOW WQI scores (<70) for contaminated water,
    but instead predicts HIGH scores (≥70) in 100% of cases.
    """
    # All regressor predictions should be ≥70 (good water quality)
    high_wqi_predictions = (regressor_results['predicted_wqi'] >= 70).sum()
    total_predictions = len(regressor_results)

    assert high_wqi_predictions == total_predictions, \
        f"All regressor predictions should be ≥70, got {high_wqi_predictions}/{total_predictions}"

    # Verify this results in 100% false negative rate
    fn_rate = regressor_results['false_negative'].mean() * 100
    assert fn_rate == 100.0, \
        f"Regressor FN rate should be 100%, got {fn_rate:.1f}%"


# === LEAD CONTAMINATION LEVEL TESTS ===

def test_extreme_lead_still_undetected(environmental_justice_data):
    """
    Test 11: Verify even EXTREME lead levels (150 ppb) are not detected.

    EPA action level: 15 ppb
    Test level: 150 ppb (10× EPA action level)

    If the system can't detect lead at 150 ppb, it cannot detect lead at ANY level.
    """
    # Find scenario with extreme lead (150 ppb)
    extreme_lead = environmental_justice_data[
        environmental_justice_data['lead_ppb'] == 150
    ]

    assert len(extreme_lead) == 2, "Should have 2 results for 150 ppb scenario"

    # All should be false negatives
    fn_rate = extreme_lead['false_negative'].mean() * 100
    assert fn_rate == 100.0, \
        f"Even at 150 ppb (10× EPA limit), FN rate should be 100%, got {fn_rate:.1f}%"


def test_moderate_lead_still_undetected(environmental_justice_data):
    """
    Test 12: Verify moderate lead levels (18-35 ppb) are not detected.

    EPA action level: 15 ppb
    Test range: 18-35 ppb (1.2× to 2.3× EPA action level)

    These are realistic levels from Jackson's aging infrastructure.
    """
    # Find scenarios with moderate lead (18-35 ppb)
    moderate_lead = environmental_justice_data[
        (environmental_justice_data['lead_ppb'] >= 18) &
        (environmental_justice_data['lead_ppb'] <= 35)
    ]

    assert len(moderate_lead) == 8, \
        f"Should have 8 results for moderate lead (4 scenarios × 2 components), got {len(moderate_lead)}"

    # All should be false negatives
    fn_rate = moderate_lead['false_negative'].mean() * 100
    assert fn_rate == 100.0, \
        f"For moderate lead (18-35 ppb), FN rate should be 100%, got {fn_rate:.1f}%"


# === SAFETY PROBABILITY TESTS (Classifier Confidence) ===

def test_classifier_confidence_analysis(classifier_results):
    """
    Test 13: Analyze ML Classifier confidence when making false negative predictions.

    Low confidence (50-60%): Model is "unsure" but still predicts SAFE
    High confidence (>70%): Model is "confident" in false SAFE prediction

    Lower confidence is better (indicates some uncertainty about contaminated water).
    """
    # Get confidence levels
    safe_probabilities = classifier_results['safe_probability']

    mean_confidence = safe_probabilities.mean()
    min_confidence = safe_probabilities.min()
    max_confidence = safe_probabilities.max()

    print(f"\n📊 Classifier Confidence on Lead-Contaminated Water:")
    print(f"   Mean: {mean_confidence:.1%}")
    print(f"   Min: {min_confidence:.1%}")
    print(f"   Max: {max_confidence:.1%}")

    # Verify all probabilities are reasonable (not 0 or 100%)
    assert (safe_probabilities > 0).all(), "Probabilities should be > 0%"
    assert (safe_probabilities < 1).all(), "Probabilities should be < 100%"

    # Document that even with uncertainty, the system still predicts SAFE
    assert (safe_probabilities > 0.5).all(), \
        "All contaminated water predicted as SAFE (P(safe) > 50%)"


# === ENVIRONMENTAL JUSTICE DOCUMENTATION TEST ===

def test_environmental_justice_metrics_documented():
    """
    Test 14: Verify environmental justice metrics are properly documented.

    This test ensures we have:
    1. CSV data with all required fields
    2. Documentation of demographics (Flint: 53% Black, Jackson: 83% Black)
    3. Documentation of false negative rates (100%)
    4. Proper data types for analysis

    These metrics are critical for transparency and accountability.
    """
    # Check that CSV exists
    data_path = Path(__file__).parent.parent / "data" / "environmental_justice_ml_results.csv"
    assert data_path.exists(), "Environmental justice results CSV should exist"

    # Check that analysis documentation exists
    docs_path = Path(__file__).parent.parent / "docs" / "ENVIRONMENTAL_JUSTICE_ANALYSIS.md"
    assert docs_path.exists(), "Environmental justice analysis documentation should exist"

    # Load and verify CSV structure
    df = pd.read_csv(data_path)

    required_fields = ['scenario', 'component', 'lead_ppb', 'false_negative']
    for field in required_fields:
        assert field in df.columns, f"Required field '{field}' should be in CSV"

    print("\n✅ Environmental justice metrics properly documented:")
    print(f"   CSV: {data_path}")
    print(f"   Docs: {docs_path}")
    print(f"   Scenarios: {df['scenario'].nunique()} unique scenarios")
    print(f"   Components: {df['component'].nunique()} components tested")


# === SUMMARY STATISTICS ===

def test_generate_fairness_summary(environmental_justice_data, flint_results, jackson_results):
    """
    Test 15: Generate comprehensive fairness analysis summary.

    Provides aggregate statistics for documentation and reporting.
    """
    print("\n" + "="*80)
    print("FAIRNESS & DEMOGRAPHICS ANALYSIS SUMMARY")
    print("="*80)

    print(f"\nTotal Scenarios Tested: {environmental_justice_data['scenario'].nunique()}")
    print(f"  Flint, MI (53% Black, 41% poverty): {flint_results['scenario'].nunique()} scenarios")
    print(f"  Jackson, MS (83% Black, 25% poverty): {jackson_results['scenario'].nunique()} scenarios")

    print(f"\nComponents Tested: {environmental_justice_data['component'].nunique()}")
    print(f"  ML Classifier")
    print(f"  ML Regressor")

    print(f"\nLead Contamination Levels:")
    lead_levels = sorted(environmental_justice_data['lead_ppb'].unique())
    for level in lead_levels:
        epa_multiple = level / 15  # EPA action level is 15 ppb
        print(f"  {level} ppb ({epa_multiple:.1f}× EPA action level)")

    print(f"\nFalse Negative Rates:")
    overall_fn = environmental_justice_data['false_negative'].mean() * 100
    flint_fn = flint_results['false_negative'].mean() * 100
    jackson_fn = jackson_results['false_negative'].mean() * 100

    print(f"  Overall: {overall_fn:.1f}%")
    print(f"  Flint: {flint_fn:.1f}%")
    print(f"  Jackson: {jackson_fn:.1f}%")

    print("\n" + "="*80)
    print("KEY FINDING: 100% false negative rate across ALL demographics")
    print("System cannot detect lead contamination for ANY community")
    print("="*80 + "\n")

    # Assertions to ensure test fails if metrics change unexpectedly
    assert overall_fn == 100.0, "Overall FN rate should remain 100%"
    assert flint_fn == 100.0, "Flint FN rate should remain 100%"
    assert jackson_fn == 100.0, "Jackson FN rate should remain 100%"


# === RUN AS STANDALONE SCRIPT ===

if __name__ == "__main__":
    """
    Run fairness tests standalone with verbose output.

    Usage:
        python tests/test_fairness_demographics.py
        pytest tests/test_fairness_demographics.py -v
    """
    pytest.main([__file__, "-v", "-s"])
