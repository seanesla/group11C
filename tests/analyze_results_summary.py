"""
Analyze the geographic coverage results and produce summary statistics.
"""
import json
from collections import Counter
from pathlib import Path

# Load results
with open('tests/geographic_coverage_191_results.json', 'r') as f:
    data = json.load(f)

results = data['results']
total = len(results)

print("=" * 100)
print("GEOGRAPHIC COVERAGE ANALYSIS: 191 US ZIP CODES")
print("=" * 100)
print(f"Data collection timestamp: {data['timestamp']}")
print(f"Date range: {data['date_range']['start'][:10]} to {data['date_range']['end'][:10]}")
print(f"Total locations tested: {total}")
print()

# 1. GEOLOCATION SUCCESS
print("=" * 100)
print("1. GEOLOCATION SUCCESS RATE")
print("=" * 100)
geo_success = sum(1 for r in results if r['geolocation'] is not None)
geo_failed = total - geo_success
print(f"✓ Successful geolocation: {geo_success}/{total} ({geo_success/total*100:.1f}%)")
print(f"✗ Failed geolocation: {geo_failed}/{total} ({geo_failed/total*100:.1f}%)")
print()

# 2. WQP DATA AVAILABILITY
print("=" * 100)
print("2. WATER QUALITY DATA AVAILABILITY")
print("=" * 100)
wqp_has_data = sum(1 for r in results if r['wqp_data'] and r['wqp_data']['has_data'])
wqp_no_data = sum(1 for r in results if r['wqp_data'] and not r['wqp_data']['has_data'])
wqp_not_attempted = total - wqp_has_data - wqp_no_data

print(f"✓ Locations WITH WQP data: {wqp_has_data}/{total} ({wqp_has_data/total*100:.1f}%)")
print(f"⊘ Locations WITHOUT WQP data: {wqp_no_data}/{total} ({wqp_no_data/total*100:.1f}%)")
print(f"⊗ Not attempted (geo failed): {wqp_not_attempted}/{total}")
print()

# Distribution of measurement counts
measurement_counts = [r['wqp_data']['measurement_count'] for r in results
                      if r['wqp_data'] and r['wqp_data']['has_data']]
if measurement_counts:
    print(f"Measurement count statistics (for {len(measurement_counts)} locations with data):")
    print(f"  Min:    {min(measurement_counts):,}")
    print(f"  Max:    {max(measurement_counts):,}")
    print(f"  Mean:   {sum(measurement_counts)/len(measurement_counts):,.1f}")
    print(f"  Median: {sorted(measurement_counts)[len(measurement_counts)//2]:,}")
    print()

# 3. WQI CALCULATION SUCCESS
print("=" * 100)
print("3. WQI CALCULATION SUCCESS")
print("=" * 100)
wqi_calculated = sum(1 for r in results if r['wqi'] is not None)
wqi_not_calculated = total - wqi_calculated
print(f"✓ WQI calculated: {wqi_calculated}/{total} ({wqi_calculated/total*100:.1f}%)")
print(f"✗ WQI not calculated: {wqi_not_calculated}/{total} ({wqi_not_calculated/total*100:.1f}%)")
print()

# WQI score distribution
if wqi_calculated > 0:
    wqi_scores = [r['wqi']['score'] for r in results if r['wqi']]
    wqi_classifications = [r['wqi']['classification'] for r in results if r['wqi']]

    print(f"WQI score statistics (n={len(wqi_scores)}):")
    print(f"  Min:    {min(wqi_scores):.2f}")
    print(f"  Max:    {max(wqi_scores):.2f}")
    print(f"  Mean:   {sum(wqi_scores)/len(wqi_scores):.2f}")
    print(f"  Median: {sorted(wqi_scores)[len(wqi_scores)//2]:.2f}")
    print()

    print("WQI classifications:")
    classification_counts = Counter(wqi_classifications)
    for classification in ['Excellent', 'Good', 'Medium', 'Bad', 'Very Bad']:
        count = classification_counts.get(classification, 0)
        pct = count / len(wqi_classifications) * 100
        print(f"  {classification:12s}: {count:3d} ({pct:5.1f}%)")
    print()

    # Parameter coverage
    param_counts = [r['wqi']['parameter_count'] for r in results if r['wqi']]
    print("Parameter coverage (out of 6 core parameters):")
    param_distribution = Counter(param_counts)
    for n in range(1, 7):
        count = param_distribution.get(n, 0)
        pct = count / len(param_counts) * 100 if param_counts else 0
        print(f"  {n}/6 params: {count:3d} ({pct:5.1f}%)")
    print()

# 4. ML PREDICTIONS
print("=" * 100)
print("4. ML PREDICTION SUCCESS")
print("=" * 100)
ml_success = sum(1 for r in results if r['ml_predictions'] is not None)
ml_failed = total - ml_success
print(f"✓ ML predictions made: {ml_success}/{total} ({ml_success/total*100:.1f}%)")
print(f"✗ ML predictions not made: {ml_failed}/{total} ({ml_failed/total*100:.1f}%)")
print()

if ml_success > 0:
    ml_verdicts = [r['ml_predictions']['classifier_verdict'] for r in results if r['ml_predictions']]
    ml_confidences = [r['ml_predictions']['classifier_confidence'] for r in results if r['ml_predictions']]
    ml_wqi_preds = [r['ml_predictions']['regressor_wqi'] for r in results if r['ml_predictions']]

    print(f"Classifier verdicts (n={len(ml_verdicts)}):")
    verdict_counts = Counter(ml_verdicts)
    for verdict in ['SAFE', 'UNSAFE']:
        count = verdict_counts.get(verdict, 0)
        pct = count / len(ml_verdicts) * 100
        print(f"  {verdict:6s}: {count:3d} ({pct:5.1f}%)")
    print()

    print(f"Classifier confidence statistics:")
    print(f"  Min:    {min(ml_confidences)*100:.1f}%")
    print(f"  Max:    {max(ml_confidences)*100:.1f}%")
    print(f"  Mean:   {sum(ml_confidences)/len(ml_confidences)*100:.1f}%")
    print(f"  Median: {sorted(ml_confidences)[len(ml_confidences)//2]*100:.1f}%")
    print()

    print(f"Regressor WQI predictions:")
    print(f"  Min:    {min(ml_wqi_preds):.2f}")
    print(f"  Max:    {max(ml_wqi_preds):.2f}")
    print(f"  Mean:   {sum(ml_wqi_preds)/len(ml_wqi_preds):.2f}")
    print(f"  Median: {sorted(ml_wqi_preds)[len(ml_wqi_preds)//2]:.2f}")
    print()

# 5. ERRORS ENCOUNTERED
print("=" * 100)
print("5. ERRORS ENCOUNTERED")
print("=" * 100)
locations_with_errors = sum(1 for r in results if r['errors'])
total_errors = sum(len(r['errors']) for r in results)
print(f"Locations with errors: {locations_with_errors}/{total} ({locations_with_errors/total*100:.1f}%)")
print(f"Total errors: {total_errors}")
print()

if total_errors > 0:
    error_types = []
    for r in results:
        for error in r['errors']:
            error_type = error.split(':')[0]
            error_types.append(error_type)

    print("Error types:")
    error_counts = Counter(error_types)
    for error_type, count in error_counts.most_common():
        print(f"  {error_type}: {count}")
    print()

# 6. GEOGRAPHIC DISTRIBUTION
print("=" * 100)
print("6. GEOGRAPHIC DISTRIBUTION (BY STATE)")
print("=" * 100)

state_success = {}
for r in results:
    if r['geolocation'] and r['geolocation']['state_code']:
        state = r['geolocation']['state_code']
        if state not in state_success:
            state_success[state] = {'total': 0, 'has_data': 0, 'has_wqi': 0, 'has_ml': 0}
        state_success[state]['total'] += 1
        if r['wqp_data'] and r['wqp_data']['has_data']:
            state_success[state]['has_data'] += 1
        if r['wqi'] is not None:
            state_success[state]['has_wqi'] += 1
        if r['ml_predictions'] is not None:
            state_success[state]['has_ml'] += 1

print(f"States represented: {len(state_success)}")
print()
print("Top 10 states by number of test locations:")
for state, counts in sorted(state_success.items(), key=lambda x: x[1]['total'], reverse=True)[:10]:
    print(f"  {state}: {counts['total']} locations ({counts['has_data']} with data, {counts['has_wqi']} with WQI, {counts['has_ml']} with ML)")
print()

# States without WQP data
states_no_data = [state for state, counts in state_success.items() if counts['has_data'] == 0]
if states_no_data:
    print(f"States with NO WQP data at any test location: {', '.join(sorted(states_no_data))}")
    print()

# 7. LOCATIONS WITHOUT DATA (for investigation)
print("=" * 100)
print("7. LOCATIONS WITHOUT WQP DATA (NEED INVESTIGATION)")
print("=" * 100)
no_data_locations = [r for r in results if r['wqp_data'] and not r['wqp_data']['has_data']]
print(f"Total: {len(no_data_locations)} locations")
print()
if no_data_locations:
    print("Examples (first 10):")
    for r in no_data_locations[:10]:
        state = r['geolocation']['state_code'] if r['geolocation'] else 'Unknown'
        print(f"  {r['zip_code']} ({r['description']}) - {state}")
    print()

# 8. FULL PIPELINE SUCCESS
print("=" * 100)
print("8. END-TO-END PIPELINE SUCCESS")
print("=" * 100)
full_success = sum(1 for r in results
                   if r['geolocation'] and r['wqp_data'] and r['wqp_data']['has_data']
                   and r['wqi'] and r['ml_predictions'])
print(f"✓ Complete pipeline (geo → WQP → WQI → ML): {full_success}/{total} ({full_success/total*100:.1f}%)")
print()

print("=" * 100)
print("SUMMARY CONCLUSION")
print("=" * 100)
print(f"Out of {total} diverse US locations tested:")
print(f"  • {geo_success} ({geo_success/total*100:.0f}%) successfully geolocated")
print(f"  • {wqp_has_data} ({wqp_has_data/total*100:.0f}%) have WQP monitoring data available")
print(f"  • {wqi_calculated} ({wqi_calculated/total*100:.0f}%) have WQI calculated")
print(f"  • {ml_success} ({ml_success/total*100:.0f}%) have ML predictions")
print(f"  • {full_success} ({full_success/total*100:.0f}%) complete full pipeline successfully")
print()
print("=" * 100)
