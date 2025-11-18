"""
Diagnostic test to verify components work before running full 191 ZIP code test.
"""
import sys
from pathlib import Path

def test_imports():
    """Test that all required imports work."""
    print("\n=== TEST 1: IMPORTS ===")

    from src.geolocation.zipcode_mapper import ZipCodeMapper
    print("✓ ZipCodeMapper imported")

    from src.data_collection.wqp_client import WQPClient
    print("✓ WQPClient imported")

    from src.utils.wqi_calculator import WQICalculator
    print("✓ WQICalculator imported")

    from src.preprocessing.us_data_features import prepare_us_features_for_prediction
    print("✓ prepare_us_features_for_prediction imported")

    from src.models.model_utils import load_latest_models
    print("✓ load_latest_models imported")

    print("✅ All imports successful\n")


def test_model_loading():
    """Test that ML models load correctly."""
    print("\n=== TEST 2: MODEL LOADING ===")

    from src.models.model_utils import load_latest_models

    print("Loading models...")
    classifier, regressor = load_latest_models()

    assert classifier is not None, "Classifier failed to load"
    print(f"✓ Classifier loaded: {type(classifier).__name__}")

    assert regressor is not None, "Regressor failed to load"
    print(f"✓ Regressor loaded: {type(regressor).__name__}")

    print("✅ Models loaded successfully\n")

    return classifier, regressor


def test_single_zip_code(classifier, regressor):
    """Test complete pipeline for a single ZIP code."""
    print("\n=== TEST 3: SINGLE ZIP CODE PIPELINE ===")

    from src.geolocation.zipcode_mapper import ZipCodeMapper
    from src.data_collection.wqp_client import WQPClient
    from src.utils.wqi_calculator import WQICalculator
    from src.preprocessing.us_data_features import prepare_us_features_for_prediction
    from datetime import datetime, timedelta
    import pandas as pd

    zip_code = "01230"  # Great Barrington MA
    print(f"Testing ZIP code: {zip_code}")

    # Step 1: Geolocation
    mapper = ZipCodeMapper()
    coords = mapper.get_coordinates(zip_code)
    assert coords is not None, f"Failed to get coordinates for {zip_code}"
    lat, lon = coords
    print(f"✓ Geolocation: ({lat:.4f}, {lon:.4f})")

    # Step 2: WQP Data (might be empty, which is OK)
    client = WQPClient()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*5)

    print("  Fetching WQP data (this takes ~2 seconds due to rate limiting)...")
    wq_data = client.get_data_by_location(
        latitude=lat,
        longitude=lon,
        radius_miles=25,
        start_date=start_date,
        end_date=end_date,
        characteristics=['pH', 'Dissolved oxygen (DO)', 'Temperature, water',
                       'Turbidity', 'Nitrate', 'Specific conductance']
    )

    if wq_data.empty:
        print("  ⓘ No WQP data available for this location (not an error)")
        print("✅ Pipeline test complete (no data to process)\n")
        return

    print(f"✓ WQP Data retrieved: {len(wq_data)} measurements")

    # Step 3: Aggregate and calculate WQI
    calculator = WQICalculator()

    # Simple aggregation for testing
    param_mapping = {
        'pH': 'ph',
        'Dissolved oxygen (DO)': 'dissolved_oxygen',
        'Temperature, water': 'temperature',
        'Turbidity': 'turbidity',
        'Nitrate': 'nitrate',
        'Specific conductance': 'conductance'
    }

    aggregated = {}
    for char_name, param_key in param_mapping.items():
        mask = wq_data['CharacteristicName'] == char_name
        values = pd.to_numeric(wq_data.loc[mask, 'ResultMeasureValue'], errors='coerce').dropna()
        if len(values) > 0:
            aggregated[param_key] = float(values.median())

    if len(aggregated) < 6:
        print(f"  ⓘ Only {len(aggregated)}/6 parameters available, skipping ML predictions")
        print("✅ Pipeline test complete (insufficient parameters for ML)\n")
        return

    wqi, scores, classification = calculator.calculate_wqi(**aggregated)
    print(f"✓ WQI calculated: {wqi:.2f} ({classification})")

    # Step 4: ML Predictions
    X_features = prepare_us_features_for_prediction(
        ph=aggregated['ph'],
        dissolved_oxygen=aggregated['dissolved_oxygen'],
        temperature=aggregated['temperature'],
        turbidity=aggregated['turbidity'],
        nitrate=aggregated['nitrate'],
        conductance=aggregated['conductance'],
        year=datetime.now().year
    )
    print(f"✓ Features prepared: {X_features.shape}")

    clf_pred = classifier.predict(X_features)[0]
    clf_prob = classifier.predict_proba(X_features)[0]
    verdict = "SAFE" if clf_pred == 1 else "UNSAFE"
    confidence = max(clf_prob) * 100
    print(f"✓ Classifier prediction: {verdict} ({confidence:.1f}% confidence)")

    reg_pred = regressor.predict(X_features)[0]
    print(f"✓ Regressor prediction: WQI={reg_pred:.2f}")

    print("✅ Complete pipeline successful!\n")


if __name__ == "__main__":
    try:
        # Run diagnostic tests
        test_imports()
        classifier, regressor = test_model_loading()
        test_single_zip_code(classifier, regressor)

        print("="*60)
        print("✅ ALL DIAGNOSTIC TESTS PASSED")
        print("="*60)
        print("\nReady to run full 191 ZIP code test.")

    except Exception as e:
        print(f"\n❌ DIAGNOSTIC TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
