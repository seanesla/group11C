# ML Model Implementation Plan - Evidence-Based, Zero Assumptions

## Executive Summary
Building production-quality sklearn ML models using European Kaggle water quality dataset (20,000 records, 1991-2017) to predict water quality for US locations via USGS/WQP APIs.

## Evidence from Data Analysis

### Kaggle Dataset Facts (data/raw/waterPollution.csv)
- **Shape:** 20,000 rows × 29 columns
- **Geographic:** European countries only (France 48%, UK 20%, Spain 16%, Germany, Czech Republic, etc.)
- **Temporal:** 1991-2017 (23 years, avg 870 records/year)
- **NO US DATA:** Confirmed - exclusively European monitoring stations

### Available WQI Parameters (Verified)
| Parameter | Kaggle Code | Records | Unit | Range | Mean |
|-----------|------------|---------|------|-------|------|
| pH | EEA_3152-01-0 | 1,157 | [pH] | 5.65-8.60 | 7.63 |
| Temperature | EEA_3121-01-5 | 898 | Cel | 0.50-23.80 | 12.04 |
| Dissolved Oxygen | EEA_3133-01-5 | 1,214 | mg{O2}/L | 0.25-58.50 | 4.36 |
| Nitrate | CAS_14797-55-8 | 1,289 | mg{NO3}/L | 0.07-188.24 | 12.65 |
| Conductance | EEA_3142-01-6 | 738 | uS/cm | 14.25-3496.67 | 407.48 |
| **Turbidity** | **NOT AVAILABLE** | 0 | N/A | N/A | N/A |

### Additional Features in Dataset
- Environmental: PopulationDensity, TerraMarineProtected, droughts_floods_temperature
- Economic: GDP, TouristMean, VenueCount, netMigration, literacyRate
- Waste: 9 composition percentages, recycling rate
- Temporal: phenomenonTimeReferenceYear, parameterSamplingPeriod
- Location: waterBodyIdentifier, Country, parameterWaterBodyCategory (RW/GW/LW)

### WQI Calculator Requirements (src/utils/wqi_calculator.py)
- **Method:** `calculate_wqi(ph, dissolved_oxygen, temperature, turbidity, nitrate, conductance)`
- **Returns:** `(wqi_score: float, parameter_scores: Dict, classification: str)`
- **Classifications:** 90-100 (Excellent), 70-89 (Good), 50-69 (Fair), 25-49 (Poor), 0-24 (Very Poor)
- **Safe threshold:** WQI ≥ 70

## Implementation Strategy

### Phase 1: Feature Engineering Pipeline
**File:** `src/preprocessing/feature_engineering.py`

#### 1.1 Data Loading and Filtering
```python
def load_kaggle_data() -> pd.DataFrame:
    """Load and validate Kaggle dataset"""
    # Load data/raw/waterPollution.csv
    # Verify shape (20000, 29)
    # Return raw dataframe
```

#### 1.2 Parameter Extraction
```python
def extract_wqi_parameters(df: pd.DataFrame) -> pd.DataFrame:
    """Extract 5 available WQI parameters using verified codes"""
    PARAMETER_MAPPING = {
        'EEA_3152-01-0': 'ph',
        'EEA_3121-01-5': 'temperature',
        'EEA_3133-01-5': 'dissolved_oxygen',
        'CAS_14797-55-8': 'nitrate',
        'EEA_3142-01-6': 'conductance'
    }
    # Filter by codes, pivot to wide format
    # Group by waterBodyIdentifier + year
    # Handle missing turbidity (set to None)
```

#### 1.3 WQI Calculation
```python
def calculate_wqi_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate WQI and labels using existing calculator"""
    # Use WQICalculator.calculate_wqi() for each row
    # Add columns: wqi_score, wqi_class, is_safe (WQI >= 70)
    # Handle rows with insufficient parameters
```

#### 1.4 Feature Creation
```python
def create_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features for ML models"""
    # Water quality features (5 parameters + derived)
    # Temporal features (year, season, trend)
    # Geographic features (country one-hot, water body type)
    # Environmental context (GDP, population, tourism, waste)
    # Missing value indicators
```

### Phase 2: Classification Model (Safe/Unsafe)
**File:** `src/models/classifier.py`

#### 2.1 Model Architecture
- **Algorithm:** RandomForestClassifier (baseline), then GradientBoostingClassifier
- **Target:** Binary classification (is_safe: WQI >= 70)
- **Class imbalance handling:** Check ratio, use class_weight='balanced' if needed

#### 2.2 Training Pipeline
```python
class WaterQualityClassifier:
    def prepare_data(self, df: pd.DataFrame) -> Tuple[X, y]:
        """Prepare features and labels"""
        # X = all features except target columns
        # y = is_safe (binary)
        # Handle missing values (median imputation for numerics)
        # Scale features (StandardScaler)

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train with cross-validation"""
        # Split: 60% train, 20% val, 20% test (stratified)
        # GridSearchCV with 5-fold CV on train set
        # Parameters to tune:
        #   - n_estimators: [100, 200, 300]
        #   - max_depth: [10, 20, None]
        #   - min_samples_split: [2, 5, 10]
        # Return best model + metrics

    def evaluate(self, model, X_test, y_test) -> Dict[str, float]:
        """Comprehensive evaluation"""
        # Metrics: Accuracy, Precision, Recall, F1-Score
        # Confusion Matrix
        # ROC-AUC if applicable
        # Feature importance analysis
```

### Phase 3: Regression Model (WQI Prediction)
**File:** `src/models/regressor.py`

#### 3.1 Model Architecture
- **Algorithm:** RandomForestRegressor (baseline), then GradientBoostingRegressor
- **Target:** Continuous WQI score (0-100)
- **Trend component:** Include year as feature for temporal trends

#### 3.2 Training Pipeline
```python
class WQIPredictionRegressor:
    def prepare_data(self, df: pd.DataFrame) -> Tuple[X, y]:
        """Prepare features and labels"""
        # X = all features including temporal
        # y = wqi_score (continuous)
        # Same preprocessing as classifier

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train with cross-validation"""
        # Split: 60% train, 20% val, 20% test
        # GridSearchCV with 5-fold CV
        # Parameters to tune (similar to classifier)
        # Return best model + metrics

    def evaluate(self, model, X_test, y_test) -> Dict[str, float]:
        """Comprehensive evaluation"""
        # Metrics: R², MAE, MSE, RMSE
        # Residual plots
        # Feature importance
        # Prediction vs actual scatter
```

### Phase 4: Model Persistence
```python
def save_models(classifier, regressor, metadata: Dict):
    """Save trained models with versioning"""
    # Timestamp: YYYYMMDD_HHMMSS
    # Files:
    #   - data/models/classifier_[timestamp].joblib
    #   - data/models/regressor_[timestamp].joblib
    #   - data/models/metadata_[timestamp].json
    # Metadata includes: training metrics, feature names, version
```

### Phase 5: Integration with Streamlit
**Updates to:** `streamlit_app/app.py`

```python
def load_ml_models():
    """Load latest trained models"""
    # Find most recent model files
    # Load with joblib
    # Cache in session state

def predict_water_quality(df: pd.DataFrame) -> Dict:
    """Apply ML models to US data"""
    # Extract same features as training
    # Handle missing turbidity appropriately
    # Return: {
    #   'classification': 'Safe/Unsafe',
    #   'confidence': probability,
    #   'predicted_wqi': score,
    #   'trend': 'Improving/Stable/Declining'
    # }
```

### Phase 6: Testing Strategy

#### 6.1 Unit Tests (src/models/)
- Test feature extraction with known inputs
- Test model training doesn't crash
- Test prediction shapes match expectations
- Test saved models can be loaded
- Test missing value handling

#### 6.2 Integration Tests
- Test full pipeline: raw data → features → predictions
- Test model integration with Streamlit
- Test predictions on real US data samples

#### 6.3 Edge Cases
- All parameters missing
- Single parameter available
- Extreme values (pH 14, temp 0°C)
- Future years beyond training range

## Critical Constraints & Mitigations

### Constraint 1: No Turbidity Data
**Impact:** One of 6 WQI parameters missing (13% weight)
**Mitigation:**
- Set turbidity=None in all training samples
- Train models to work with 5 parameters
- Document limitation clearly
- WQI calculator already handles missing parameters

### Constraint 2: European vs US Data
**Impact:** Different regulations, pollution sources, methodologies
**Mitigation:**
- Acknowledge in documentation
- Focus on universal relationships (pH is pH everywhere)
- Validate predictions against known US water quality
- Add confidence intervals to predictions

### Constraint 3: Temporal Gap
**Impact:** Training data ends 2017, predicting for 2024+
**Mitigation:**
- Include year as feature to learn trends
- Test extrapolation capability
- Document temporal limitation

## Success Criteria
1. ✅ All 312 existing tests still pass
2. ✅ 80+ new tests for ML components
3. ✅ Models achieve >75% accuracy (classification)
4. ✅ Models achieve R² >0.6 (regression)
5. ✅ Feature importance makes scientific sense
6. ✅ Integration with Streamlit works end-to-end
7. ✅ Real US data predictions are reasonable

## File Checklist
- [ ] src/preprocessing/feature_engineering.py
- [ ] src/models/classifier.py
- [ ] src/models/regressor.py
- [ ] src/models/__init__.py
- [ ] tests/test_feature_engineering.py
- [ ] tests/test_classifier.py
- [ ] tests/test_regressor.py
- [ ] Update streamlit_app/app.py
- [ ] data/models/ directory for saved models

## No Assumptions - Everything Verified
- Dataset structure: Verified via pandas analysis
- Parameter codes: Verified via value_counts
- Unit mappings: Verified via resultUom column
- Value ranges: Verified via min/max analysis
- WQI calculator signature: Verified via source code
- Missing turbidity: Confirmed not available

## Timeline
1. Feature engineering: 1 hour
2. Classification model: 1 hour
3. Regression model: 1 hour
4. Integration: 30 minutes
5. Testing: 1.5 hours
6. Documentation: 30 minutes

Total: ~5.5 hours of implementation