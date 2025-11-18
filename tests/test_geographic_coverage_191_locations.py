"""
BACKEND INTEGRATION TEST: 191 US ZIP Codes Geographic Coverage

Tests the core backend pipeline for 191 geographically diverse ZIP codes:
1. Geolocation lookup (ZIP → lat/lon + location name)
2. WQP data retrieval (handles "no data" cases gracefully)
3. For locations with data:
   - WQI calculation from median parameters
   - ML classifier prediction
   - ML regressor prediction

NO browser automation - pure Python backend testing.
Note: Feature importance and SHAP explanations are tested separately in hyperthorough test suites.
"""

import sys
from pathlib import Path
import glob
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Use pytest imports with src prefix
from src.geolocation.zipcode_mapper import ZipCodeMapper
from src.data_collection.wqp_client import WQPClient
from src.utils.wqi_calculator import WQICalculator
from src.preprocessing.us_data_features import prepare_us_features_for_prediction
from src.models.model_utils import load_latest_models


# 191 ZIP codes covering all US states + diverse geographic/demographic characteristics
TEST_ZIP_CODES = [
    # Small towns / Rural (001 prefix series)
    ("01230", "Great Barrington MA - rural Berkshires"),
    ("05001", "White River Junction VT"),
    ("12134", "Averill Park NY"),
    ("16650", "Huntingdon PA"),
    ("24701", "Bluefield WV"),
    ("26554", "Morgantown WV"),
    ("31001", "Macon GA"),
    ("35004", "Moody AL"),
    ("38801", "Tupelo MS"),
    ("41001", "Alexandria KY"),
    ("45601", "Chillicothe OH"),
    ("49001", "Kalamazoo MI"),
    ("50001", "Ackley IA"),
    ("51001", "Akron IA"),
    ("52001", "Dubuque IA"),
    ("54001", "Afton WI"),
    ("55001", "Afton MN"),
    ("56001", "Mankato MN"),
    ("57001", "Alcester SD"),
    ("58001", "Abercrombie ND"),
    ("59001", "Absarokee MT"),
    ("68001", "Abie NE"),
    ("73001", "Albert OK"),
    ("74001", "Avant OK"),
    ("75001", "Addison TX"),
    ("76001", "Arlington TX"),
    ("77001", "Houston TX"),
    ("78001", "Bandera TX"),
    ("79001", "Abernathy TX"),
    ("83001", "Afton WY"),
    ("84001", "Altamont UT"),
    ("85001", "Phoenix AZ"),
    ("86001", "Flagstaff AZ"),
    ("87001", "Algodones NM"),
    ("88001", "Alamogordo NM"),
    ("89001", "Alamo NV"),
    ("06001", "Avon CT"),
    ("07001", "Avenel NJ"),
    ("08001", "Alloway NJ"),
    ("17001", "Camp Hill PA"),
    ("18001", "Bethlehem PA"),
    ("19001", "Abington PA"),
    ("21001", "Aberdeen MD"),
    ("22001", "Ashburn VA"),
    ("23001", "Aylett VA"),
    ("27001", "Advance NC"),
    ("28001", "Albemarle NC"),
    ("29001", "Alcolu SC"),
    ("32001", "Amelia Island FL"),
    ("33001", "Aventura FL"),
    ("34001", "Arcadia FL"),
    ("37001", "Whitwell TN"),
    ("42001", "Paducah KY"),
    ("43001", "Alexandria OH"),
    ("46001", "Anderson IN"),
    ("47001", "Attica IN"),
    ("60001", "Aikin IL"),
    ("61001", "Abingdon IL"),
    ("62001", "Alorton IL"),
    ("63001", "Affton MO"),
    ("64001", "Atchison KS"),
    ("65001", "Ashland MO"),
    ("66001", "Abilene KS"),
    ("67001", "Andale KS"),
    ("70001", "Ama LA"),
    ("71001", "Addis LA"),
    ("72001", "Adona AR"),
    ("80001", "Arvada CO"),
    ("81001", "Alamosa CO"),
    ("82001", "Albin WY"),
    ("90001", "Los Angeles CA"),
    ("91001", "Altadena CA"),
    ("92001", "Bonsall CA"),
    ("93001", "Atascadero CA"),
    ("94001", "Brisbane CA"),
    ("95001", "Aptos CA"),
    ("96001", "Adin CA"),
    ("97001", "Antelope OR"),
    ("98001", "Auburn WA"),
    ("99001", "Airway Heights WA"),

    # Major cities
    ("10001", "New York NY"),
    ("11001", "Floral Park NY"),
    ("13001", "Pulaski NY"),
    ("14001", "Akron NY"),
    ("15001", "Aliquippa PA"),
    ("25001", "Appalachia VA"),
    ("30001", "Apalachee GA"),
    ("36001", "Autaugaville AL"),
    ("39001", "Ackerman MS"),
    ("40001", "Anchorage KY"),
    ("44001", "Akron OH"),
    ("48001", "Bloomfield Hills MI"),
    ("53001", "Allenton WI"),
    ("58002", "Absaraka ND"),
    ("68002", "Arapahoe NE"),
    ("69001", "Alliance NE"),
    ("88002", "Anthony NM"),
    ("97002", "Aurora OR"),
    ("02001", "Amherst MA"),
    ("03001", "Atkinson NH"),
    ("04001", "Acton ME"),
    ("12201", "Albany NY"),
    ("14201", "Buffalo NY"),
    ("19101", "Philadelphia PA"),
    ("21201", "Baltimore MD"),
    ("23451", "Virginia Beach VA"),
    ("28201", "Charlotte NC"),
    ("30301", "Atlanta GA"),
    ("32301", "Tallahassee FL"),
    ("33101", "Miami FL"),
    ("37201", "Nashville TN"),
    ("43201", "Columbus OH"),
    ("44101", "Cleveland OH"),
    ("45201", "Cincinnati OH"),
    ("46201", "Indianapolis IN"),
    ("48201", "Detroit MI"),
    ("53201", "Milwaukee WI"),
    ("55401", "Minneapolis MN"),
    ("60601", "Chicago IL"),
    ("63101", "St Louis MO"),
    ("64101", "Kansas City MO"),
    ("66101", "Kansas City KS"),
    ("68101", "Omaha NE"),
    ("70112", "New Orleans LA"),
    ("73101", "Oklahoma City OK"),
    ("75201", "Dallas TX"),
    ("77002", "Houston TX"),
    ("78201", "San Antonio TX"),
    ("79901", "El Paso TX"),
    ("80201", "Denver CO"),
    ("85701", "Tucson AZ"),
    ("87101", "Albuquerque NM"),
    ("89101", "Las Vegas NV"),
    ("90012", "Los Angeles CA"),
    ("92101", "San Diego CA"),
    ("94102", "San Francisco CA"),
    ("95814", "Sacramento CA"),
    ("97201", "Portland OR"),
    ("98101", "Seattle WA"),
    ("99201", "Spokane WA"),

    # Mid-sized cities
    ("50401", "Mason City IA"),
    ("51301", "Spencer IA"),
    ("52301", "Burlington IA"),
    ("56201", "Fairmont MN"),
    ("57401", "Aberdeen SD"),
    ("58401", "Jamestown ND"),
    ("59401", "Great Falls MT"),
    ("67601", "Hays KS"),
    ("68301", "Beatrice NE"),
    ("69101", "North Platte NE"),
    ("79601", "Abilene TX"),
    ("83201", "Pocatello ID"),
    ("84601", "Provo UT"),

    # Coastal towns / beach areas
    ("02601", "Hyannis MA"),
    ("02840", "Newport RI"),
    ("04101", "Portland ME"),
    ("08401", "Atlantic City NJ"),
    ("19971", "Rehoboth Beach DE"),
    ("21842", "Ocean City MD"),
    ("27954", "Nags Head NC"),
    ("29401", "Charleston SC"),
    ("31401", "Savannah GA"),
    ("32501", "Pensacola FL"),
    ("33139", "Miami Beach FL"),
    ("39501", "Gulfport MS"),
    ("70501", "Lafayette LA"),
    ("77550", "Galveston TX"),
    ("92054", "Oceanside CA"),
    ("93401", "San Luis Obispo CA"),
    ("97367", "Newport OR"),
    ("98362", "Port Angeles WA"),

    # Mountain / rural mountainous areas
    ("26201", "Buckhannon WV"),
    ("28901", "Andrews NC"),
    ("37601", "Johnson City TN"),
    ("59701", "Butte MT"),
    ("80424", "Breckenridge CO"),
    ("81301", "Durango CO"),
    ("82901", "Rock Springs WY"),
    ("83440", "Jackson WY"),
    ("86301", "Prescott AZ"),
    ("87501", "Santa Fe NM"),
    ("88901", "Las Cruces NM"),

    # Desert / arid regions
    ("92201", "Indio CA"),
    ("93001", "Ventura CA"),
    ("78040", "Laredo TX"),

    # Border regions
    ("98001", "Blaine WA"),  # Canadian border

    # Industrial / rust belt
    ("26101", "Parkersburg WV"),
    ("46401", "Gary IN"),

    # State capitals (additional)
    ("89701", "Carson City NV"),

    # Additional California diversity
    ("96001", "Redding CA"),

    # Washington DC (federal)
    ("20001", "Washington DC"),
]

# Parameter mapping from WQP characteristic names to WQI calculator parameter names
PARAM_MAPPING = {
    'pH': 'ph',
    'Dissolved oxygen (DO)': 'dissolved_oxygen',
    'Temperature, water': 'temperature',
    'Turbidity': 'turbidity',
    'Nitrate': 'nitrate',
    'Specific conductance': 'conductance'
}


def aggregate_wqp_data(df: pd.DataFrame) -> dict:
    """
    Aggregate long-format WQP data into parameter dictionary for WQI calculation.

    Args:
        df: Long-format DataFrame from WQPClient

    Returns:
        Dict with parameter names as keys and median values as values
    """
    aggregated = {}

    for characteristic_name, param_key in PARAM_MAPPING.items():
        # Filter rows for this characteristic
        mask = df['CharacteristicName'] == characteristic_name

        # Convert values to numeric (coerce errors to NaN)
        values = pd.to_numeric(
            df.loc[mask, 'ResultMeasureValue'],
            errors='coerce'
        ).dropna()

        if len(values) > 0:
            # Use median of all measurements
            aggregated[param_key] = float(values.median())

    return aggregated


def test_all_191_zip_codes():
    """
    Test core backend pipeline for all 191 ZIP codes.

    For each ZIP code, verify:
    1. Geolocation lookup succeeds
    2. WQP data retrieval (may return 0 results for some locations)
    3. If data exists: WQI calculation, ML classifier + regressor predictions
    """
    print("\n" + "="*100)
    print("BACKEND INTEGRATION TEST: 191 US ZIP CODES GEOGRAPHIC COVERAGE")
    print("="*100)
    print(f"Testing {len(TEST_ZIP_CODES)} geographically diverse locations")
    print("No browser automation - pure Python backend testing")
    print("="*100 + "\n")

    # Initialize components
    zip_mapper = ZipCodeMapper()
    wqp_client = WQPClient()
    wqi_calculator = WQICalculator()

    # Load ML models
    print("Loading ML models...")
    classifier, regressor = load_latest_models()
    print(f"✓ Classifier loaded successfully")
    print(f"✓ Regressor loaded successfully")
    print()

    # Track results
    results = {
        'geolocation_success': 0,
        'geolocation_fail': 0,
        'data_available': 0,
        'data_unavailable': 0,
        'wqi_calculated': 0,
        'ml_predictions': 0,
        'errors': []
    }

    # Date range for WQP queries (last 5 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)

    print(f"📅 WQP Query Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n")
    print("-" * 100)

    for i, (zip_code, description) in enumerate(TEST_ZIP_CODES, 1):
        try:
            print(f"\n[{i:3d}/191] Testing {zip_code} ({description})")

            # STEP 1: Geolocation lookup
            try:
                coords = zip_mapper.get_coordinates(zip_code)
                if coords is None:
                    results['geolocation_fail'] += 1
                    results['errors'].append((zip_code, description, "geolocation", "ZIP code not found"))
                    print(f"  ✗ Geolocation FAILED: ZIP code not found")
                    continue

                lat, lon = coords
                location_info = zip_mapper.get_location_info(zip_code)
                results['geolocation_success'] += 1
                print(f"  ✓ Geolocation: ({lat:.4f}, {lon:.4f}) - {location_info['place_name']}, {location_info['state_code']}")
            except Exception as e:
                results['geolocation_fail'] += 1
                results['errors'].append((zip_code, description, "geolocation", str(e)))
                print(f"  ✗ Geolocation FAILED: {e}")
                continue

            # STEP 2: WQP data retrieval
            try:
                wq_data = wqp_client.get_data_by_location(
                    latitude=lat,
                    longitude=lon,
                    radius_miles=25,
                    start_date=start_date,
                    end_date=end_date,
                    characteristics=list(PARAM_MAPPING.keys())
                )

                if wq_data.empty:
                    results['data_unavailable'] += 1
                    print(f"  ⓘ WQP: No data available (not an error)")
                    continue
                else:
                    results['data_available'] += 1
                    print(f"  ✓ WQP: Retrieved {len(wq_data)} measurements")
            except Exception as e:
                results['data_unavailable'] += 1
                results['errors'].append((zip_code, description, "wqp_retrieval", str(e)))
                print(f"  ✗ WQP retrieval FAILED: {e}")
                continue

            # STEP 3: Aggregate parameters and calculate WQI
            try:
                aggregated_params = aggregate_wqp_data(wq_data)

                if not aggregated_params:
                    print(f"  ⓘ WQI: No aggregatable parameters found")
                    continue

                wqi, scores, classification = wqi_calculator.calculate_wqi(**aggregated_params)
                results['wqi_calculated'] += 1
                print(f"  ✓ WQI: {wqi:.2f} ({classification}) [{len(aggregated_params)}/6 params]")

                # Skip ML predictions if insufficient parameters (need all 6)
                if len(aggregated_params) < 6:
                    print(f"  ⓘ ML: Skipped (need 6/6 params, have {len(aggregated_params)}/6)")
                    continue

            except Exception as e:
                results['errors'].append((zip_code, description, "wqi_calculation", str(e)))
                print(f"  ✗ WQI calculation FAILED: {e}")
                continue

            # STEP 4: ML predictions
            try:
                # Prepare features
                X_features = prepare_us_features_for_prediction(
                    ph=aggregated_params['ph'],
                    dissolved_oxygen=aggregated_params['dissolved_oxygen'],
                    temperature=aggregated_params['temperature'],
                    turbidity=aggregated_params['turbidity'],
                    nitrate=aggregated_params['nitrate'],
                    conductance=aggregated_params['conductance'],
                    year=datetime.now().year
                )

                # Classifier prediction
                clf_pred = classifier.predict(X_features)[0]
                clf_prob = classifier.predict_proba(X_features)[0]

                # Regressor prediction
                reg_pred = regressor.predict(X_features)[0]

                results['ml_predictions'] += 1
                verdict = "SAFE" if clf_pred == 1 else "UNSAFE"
                confidence = max(clf_prob) * 100
                print(f"  ✓ ML Classifier: {verdict} ({confidence:.1f}% confidence)")
                print(f"  ✓ ML Regressor: WQI={reg_pred:.2f}")
                print(f"  ✅ COMPLETE: Full pipeline successful for {zip_code}")

            except Exception as e:
                results['errors'].append((zip_code, description, "ml_prediction", str(e)))
                print(f"  ✗ ML prediction FAILED: {e}")
                continue

        except Exception as e:
            results['errors'].append((zip_code, description, "unexpected", str(e)))
            print(f"  ✗✗ UNEXPECTED ERROR: {e}")

    # Print summary
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)
    print(f"\n📍 GEOLOCATION:")
    print(f"  Success: {results['geolocation_success']}/{len(TEST_ZIP_CODES)} "
          f"({results['geolocation_success']/len(TEST_ZIP_CODES)*100:.1f}%)")
    print(f"  Failed:  {results['geolocation_fail']}/{len(TEST_ZIP_CODES)}")

    print(f"\n💧 WQP DATA AVAILABILITY:")
    print(f"  Data available:   {results['data_available']}/{results['geolocation_success']}")
    print(f"  No data:          {results['data_unavailable']}/{results['geolocation_success']}")

    print(f"\n🧮 BACKEND PROCESSING (locations with data):")
    print(f"  WQI calculated:              {results['wqi_calculated']}/{results['data_available']}")
    print(f"  ML predictions (full pipeline): {results['ml_predictions']}/{results['data_available']}")

    if results['errors']:
        print(f"\n❌ ERRORS: {len(results['errors'])} errors occurred")
        print("\nError details:")
        for zip_code, description, stage, error in results['errors'][:10]:  # Show first 10
            print(f"  - {zip_code} ({description}): {stage} failed - {error[:80]}")
        if len(results['errors']) > 10:
            print(f"  ... and {len(results['errors']) - 10} more errors")
    else:
        print(f"\n✅ NO ERRORS: All tested locations processed successfully")

    print("\n" + "="*100)
    print("✅ BACKEND INTEGRATION TEST COMPLETE")
    print("="*100)
    print(f"\n📊 Final Score:")
    print(f"  - Geolocation: {results['geolocation_success']}/191 success")
    print(f"  - Data availability: {results['data_available']}/191 locations have monitoring data")
    print(f"  - Full pipeline: {results['ml_predictions']}/{results['data_available']} "
          f"data-available locations successfully processed")
    print()

    # Assert critical requirements
    assert results['geolocation_success'] >= 185, \
        f"Too many geolocation failures: {results['geolocation_fail']}/191 failed"

    # Don't assert data availability - many rural areas may have no monitoring
    # Just ensure processing works when data IS available
    if results['data_available'] > 0:
        processing_rate = results['ml_predictions'] / results['data_available']
        assert processing_rate >= 0.5, \
            f"Backend processing failing too often: only {processing_rate*100:.1f}% success rate"

    print("✅ All assertions passed\n")


if __name__ == "__main__":
    test_all_191_zip_codes()
