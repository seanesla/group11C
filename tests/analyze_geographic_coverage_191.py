"""
Geographic Coverage Analysis for 191 US ZIP Codes

This script collects data and saves results to a JSON file for analysis.
NO automated pass/fail - just data collection for manual review.
"""
import json
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

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
    """Aggregate long-format WQP data into parameter dictionary."""
    aggregated = {}
    for characteristic_name, param_key in PARAM_MAPPING.items():
        mask = df['CharacteristicName'] == characteristic_name
        values = pd.to_numeric(df.loc[mask, 'ResultMeasureValue'], errors='coerce').dropna()
        if len(values) > 0:
            aggregated[param_key] = float(values.median())
    return aggregated


def collect_geographic_data():
    """
    Collect data for all 191 ZIP codes and save to JSON for analysis.

    Returns path to saved JSON file.
    """
    print("\n" + "="*100)
    print("GEOGRAPHIC COVERAGE DATA COLLECTION: 191 US ZIP CODES")
    print("="*100)
    print(f"Collecting data from {len(TEST_ZIP_CODES)} locations")
    print("This will take 5-10+ minutes due to WQP API rate limiting...")
    print("="*100 + "\n")

    # Initialize components
    zip_mapper = ZipCodeMapper()
    wqp_client = WQPClient()
    wqi_calculator = WQICalculator()

    print("Loading ML models...")
    classifier, regressor = load_latest_models()
    print(f"✓ Models loaded\n")

    # Date range for WQP queries (last 5 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)

    results = []

    for i, (zip_code, description) in enumerate(TEST_ZIP_CODES, 1):
        result = {
            'index': i,
            'zip_code': zip_code,
            'description': description,
            'geolocation': None,
            'wqp_data': None,
            'wqi': None,
            'ml_predictions': None,
            'errors': []
        }

        print(f"[{i:3d}/191] {zip_code} ({description})")

        # STEP 1: Geolocation
        try:
            coords = zip_mapper.get_coordinates(zip_code)
            if coords:
                lat, lon = coords
                location_info = zip_mapper.get_location_info(zip_code)
                result['geolocation'] = {
                    'latitude': lat,
                    'longitude': lon,
                    'place_name': location_info.get('place_name'),
                    'state_code': location_info.get('state_code')
                }
                print(f"  ✓ Geolocation: ({lat:.4f}, {lon:.4f})")
            else:
                result['errors'].append('Geolocation failed: ZIP not found')
                print(f"  ✗ Geolocation failed")
                results.append(result)
                continue
        except Exception as e:
            result['errors'].append(f'Geolocation error: {str(e)}')
            print(f"  ✗ Geolocation error: {e}")
            results.append(result)
            continue

        # STEP 2: WQP Data
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
                result['wqp_data'] = {'measurement_count': 0, 'has_data': False}
                print(f"  ⓘ No WQP data")
                results.append(result)
                continue
            else:
                result['wqp_data'] = {
                    'measurement_count': len(wq_data),
                    'has_data': True
                }
                print(f"  ✓ WQP: {len(wq_data)} measurements")
        except Exception as e:
            result['errors'].append(f'WQP error: {str(e)}')
            print(f"  ✗ WQP error: {e}")
            results.append(result)
            continue

        # STEP 3: WQI Calculation
        try:
            aggregated_params = aggregate_wqp_data(wq_data)

            if len(aggregated_params) == 0:
                print(f"  ⓘ No parameters to aggregate")
                results.append(result)
                continue

            wqi, scores, classification = wqi_calculator.calculate_wqi(**aggregated_params)
            result['wqi'] = {
                'score': wqi,
                'classification': classification,
                'parameter_count': len(aggregated_params),
                'parameters': aggregated_params
            }
            print(f"  ✓ WQI: {wqi:.2f} ({classification}) [{len(aggregated_params)}/6 params]")

            if len(aggregated_params) < 6:
                print(f"  ⓘ Insufficient params for ML")
                results.append(result)
                continue

        except Exception as e:
            result['errors'].append(f'WQI error: {str(e)}')
            print(f"  ✗ WQI error: {e}")
            results.append(result)
            continue

        # STEP 4: ML Predictions
        try:
            X_features = prepare_us_features_for_prediction(
                ph=aggregated_params['ph'],
                dissolved_oxygen=aggregated_params['dissolved_oxygen'],
                temperature=aggregated_params['temperature'],
                turbidity=aggregated_params['turbidity'],
                nitrate=aggregated_params['nitrate'],
                conductance=aggregated_params['conductance'],
                year=datetime.now().year
            )

            clf_pred = classifier.predict(X_features)[0]
            clf_prob = classifier.predict_proba(X_features)[0]
            reg_pred = regressor.predict(X_features)[0]

            result['ml_predictions'] = {
                'classifier_prediction': int(clf_pred),
                'classifier_verdict': 'SAFE' if clf_pred == 1 else 'UNSAFE',
                'classifier_confidence': float(max(clf_prob)),
                'regressor_wqi': float(reg_pred)
            }

            verdict = result['ml_predictions']['classifier_verdict']
            confidence = result['ml_predictions']['classifier_confidence'] * 100
            print(f"  ✓ ML: {verdict} ({confidence:.1f}%), WQI={reg_pred:.2f}")

        except Exception as e:
            result['errors'].append(f'ML error: {str(e)}')
            print(f"  ✗ ML error: {e}")

        results.append(result)
        print()  # Blank line between locations

    # Save results
    output_file = Path('tests/geographic_coverage_191_results.json')
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_locations': len(TEST_ZIP_CODES),
            'date_range': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'results': results
        }, f, indent=2)

    print("="*100)
    print(f"✅ DATA COLLECTION COMPLETE")
    print(f"Results saved to: {output_file}")
    print("="*100)

    return output_file


if __name__ == "__main__":
    output_file = collect_geographic_data()
    print(f"\nAnalyze results in: {output_file}")
