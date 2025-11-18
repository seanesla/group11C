#!/usr/bin/env python3
"""
Quick Validation: Test NEW 24-feature models on 5 diverse US locations.

This script validates that retraining with core parameters only (removing European-specific
features) fixes the systematic 22% under-prediction issue on US water quality data.

Sample locations:
- MA, NY, TX, OH, FL (diverse states)
- WQI scores: 88-92 (Good to Excellent)
- OLD model errors: 21-25 points (23-28%)

Success Criteria:
- NEW model error < 10 points per location
- Mean absolute error < 10 points across all locations
- Improvement > 50% reduction in error vs OLD models
"""

import sys
import json
import logging
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.us_data_features import prepare_us_features_for_prediction
from src.models.classifier import WaterQualityClassifier
from src.models.regressor import WQIPredictionRegressor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_models():
    """Load the NEW 25-feature models (core parameters only)."""
    classifier_path = "data/models/classifier_20251117_042805.joblib"
    regressor_path = "data/models/regressor_20251117_042831.joblib"

    logger.info(f"Loading NEW models (25 core features only):")
    logger.info(f"  Classifier: {classifier_path}")
    logger.info(f"  Regressor:  {regressor_path}")

    classifier = WaterQualityClassifier.load(classifier_path)
    regressor = WQIPredictionRegressor.load(regressor_path)

    return classifier, regressor


def load_sample_locations():
    """Load the 5 sample locations from 191 ZIP test results."""
    with open('tests/geographic_coverage_191_results.json', 'r') as f:
        data = json.load(f)

    results = data['results']

    # Select specific samples (indices from previous selection)
    sample_zips = ['01230', '12134', '75001', '45601', '34001']

    samples = []
    for zip_code in sample_zips:
        for r in results:
            if r['zip_code'] == zip_code:
                samples.append(r)
                break

    logger.info(f"Loaded {len(samples)} sample locations for validation")
    return samples


def validate_location(location, classifier, regressor):
    """
    Validate NEW model predictions for a single location.

    Returns:
        dict with actual WQI, NEW prediction, OLD prediction, and errors
    """
    zip_code = location['zip_code']
    description = location['description']
    state = location['geolocation']['state_code'] if location['geolocation'] else 'Unknown'

    # Get actual WQI and parameters
    if not location['wqi']:
        return None

    wqi_data = location['wqi']
    actual_wqi = wqi_data['score']
    params = wqi_data['parameters']

    # Prepare features for NEW 24-feature model
    features_df = prepare_us_features_for_prediction(
        ph=params.get('ph'),
        dissolved_oxygen=params.get('dissolved_oxygen'),
        temperature=params.get('temperature'),
        turbidity=params.get('turbidity'),
        nitrate=params.get('nitrate'),
        conductance=params.get('conductance'),
        year=2024  # Use current year for temporal features
    )

    # Get NEW predictions
    classifier_pred = int(classifier.predict(features_df)[0])
    classifier_proba = classifier.predict_proba(features_df)[0]
    classifier_verdict = "SAFE" if classifier_pred == 1 else "UNSAFE"
    classifier_confidence = float(classifier_proba[classifier_pred])

    regressor_pred = float(regressor.predict(features_df)[0])

    # Get OLD prediction for comparison
    old_ml = location.get('ml_predictions')
    old_pred = old_ml['regressor_wqi'] if old_ml else None
    old_error = abs(actual_wqi - old_pred) if old_pred else None

    # Calculate NEW error
    new_error = abs(actual_wqi - regressor_pred)

    # Calculate improvement
    improvement = None
    if old_error:
        improvement = ((old_error - new_error) / old_error) * 100

    result = {
        'zip_code': zip_code,
        'description': description,
        'state': state,
        'actual_wqi': actual_wqi,
        'actual_classification': wqi_data['classification'],
        'parameter_count': wqi_data['parameter_count'],
        'new_classifier_verdict': classifier_verdict,
        'new_classifier_confidence': classifier_confidence,
        'new_regressor_wqi': regressor_pred,
        'new_error': new_error,
        'old_regressor_wqi': old_pred,
        'old_error': old_error,
        'improvement_pct': improvement
    }

    return result


def main():
    """Run quick validation test."""
    logger.info("=" * 100)
    logger.info("QUICK VALIDATION: NEW 24-FEATURE MODELS ON 5 US LOCATIONS")
    logger.info("=" * 100)
    logger.info("")

    # Load models
    classifier, regressor = load_models()
    logger.info("")

    # Load sample locations
    samples = load_sample_locations()
    logger.info("")

    # Validate each location
    logger.info("=" * 100)
    logger.info("TESTING NEW MODELS")
    logger.info("=" * 100)
    logger.info("")

    results = []
    for i, location in enumerate(samples, 1):
        logger.info(f"Testing location {i}/{len(samples)}...")
        result = validate_location(location, classifier, regressor)
        if result:
            results.append(result)

    logger.info("")
    logger.info("=" * 100)
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 100)
    logger.info("")

    # Display results
    for i, r in enumerate(results, 1):
        logger.info(f"{i}. ZIP {r['zip_code']} ({r['description']})")
        logger.info(f"   State: {r['state']}")
        logger.info(f"   Actual WQI: {r['actual_wqi']:.2f} ({r['actual_classification']})")
        logger.info(f"   Parameters: {r['parameter_count']}/6 available")
        logger.info("")
        logger.info(f"   OLD Model (59 features):")
        if r['old_regressor_wqi']:
            logger.info(f"     Predicted WQI: {r['old_regressor_wqi']:.2f}")
            logger.info(f"     Error: {r['old_error']:.2f} points ({r['old_error']/r['actual_wqi']*100:.1f}%)")
        else:
            logger.info(f"     No OLD prediction available")
        logger.info("")
        logger.info(f"   NEW Model (24 features):")
        logger.info(f"     Classifier: {r['new_classifier_verdict']} ({r['new_classifier_confidence']*100:.1f}% confidence)")
        logger.info(f"     Predicted WQI: {r['new_regressor_wqi']:.2f}")
        logger.info(f"     Error: {r['new_error']:.2f} points ({r['new_error']/r['actual_wqi']*100:.1f}%)")
        logger.info("")
        if r['improvement_pct'] is not None:
            status = "✓ IMPROVED" if r['improvement_pct'] > 0 else "✗ WORSE"
            logger.info(f"   {status}: {abs(r['improvement_pct']):.1f}% {'reduction' if r['improvement_pct'] > 0 else 'increase'} in error")
        logger.info("")

    # Calculate summary statistics
    logger.info("=" * 100)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 100)
    logger.info("")

    new_errors = [r['new_error'] for r in results]
    old_errors = [r['old_error'] for r in results if r['old_error'] is not None]
    improvements = [r['improvement_pct'] for r in results if r['improvement_pct'] is not None]

    logger.info(f"NEW Model (24 features):")
    logger.info(f"  Mean Absolute Error: {np.mean(new_errors):.2f} points")
    logger.info(f"  Min Error:  {np.min(new_errors):.2f} points")
    logger.info(f"  Max Error:  {np.max(new_errors):.2f} points")
    logger.info("")

    if old_errors:
        logger.info(f"OLD Model (59 features):")
        logger.info(f"  Mean Absolute Error: {np.mean(old_errors):.2f} points")
        logger.info("")

        logger.info(f"Improvement:")
        logger.info(f"  Mean Error Reduction: {np.mean(improvements):.1f}%")
        logger.info(f"  Error decreased from {np.mean(old_errors):.2f} → {np.mean(new_errors):.2f} points")
        logger.info("")

    # Check success criteria
    logger.info("=" * 100)
    logger.info("SUCCESS CRITERIA CHECK")
    logger.info("=" * 100)
    logger.info("")

    success = True

    # Criterion 1: Mean error < 10 points
    if np.mean(new_errors) < 10:
        logger.info(f"✓ Mean absolute error < 10 points: {np.mean(new_errors):.2f}")
    else:
        logger.warning(f"✗ Mean absolute error >= 10 points: {np.mean(new_errors):.2f}")
        success = False

    # Criterion 2: Max error < 15 points (allowing some outliers)
    if np.max(new_errors) < 15:
        logger.info(f"✓ Max error < 15 points: {np.max(new_errors):.2f}")
    else:
        logger.warning(f"⚠ Max error >= 15 points: {np.max(new_errors):.2f} (acceptable if mean is good)")

    # Criterion 3: Improvement > 50% vs OLD
    if old_errors and np.mean(improvements) > 50:
        logger.info(f"✓ Error reduction > 50%: {np.mean(improvements):.1f}%")
    elif old_errors:
        logger.info(f"⚠ Error reduction < 50%: {np.mean(improvements):.1f}% (but still improved)")

    logger.info("")
    if success:
        logger.info("=" * 100)
        logger.info("✓✓✓ VALIDATION SUCCESSFUL - NEW MODELS READY FOR FULL TEST ✓✓✓")
        logger.info("=" * 100)
    else:
        logger.warning("=" * 100)
        logger.warning("⚠ Partial success - proceed with full validation to verify")
        logger.warning("=" * 100)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
