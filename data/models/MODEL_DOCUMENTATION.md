# Water Quality ML Models Documentation

**Generated:** November 3, 2025
**Version:** 1.0
**Models:** Classification (Safe/Unsafe) + Regression (WQI Score Prediction)

## Overview

This document describes the machine learning models trained to predict water quality safety and WQI scores based on water quality parameters and contextual features.

## Training Data

### Source
- **Dataset:** Kaggle Water Quality Dataset (public, multi-country; primarily European monitoring sites)
- **Original Size:** 20,000 records × 29 columns
- **Processed Size:** 2,939 samples × 69 features
- **Time Range:** 1991-2017 (27 years)
- **Geographic Coverage:** European countries (France 48%, UK 20%, Spain 16%, Germany, Czech Republic, and 23 others)

### Important Limitation
**⚠ CRITICAL:** Models are trained on non‑US monitoring data (primarily European sites) but applied to US water quality monitoring. This cross‑regional application has inherent limitations:
- Different regulatory standards and monitoring programs
- Different dominant pollution sources
- Potential differences in measurement methodologies
- Climate and geographic differences between regions

### Available WQI Parameters
From the Kaggle dataset, we successfully mapped 5 of 6 WQI parameters:

| Parameter | Code | Records | Unit | Range | Mean |
|-----------|------|---------|------|-------|------|
| pH | EEA_3152-01-0 | 1,157 | [pH] | 5.65-8.60 | 7.63 |
| Temperature | EEA_3121-01-5 | 898 | Cel | 0.50-23.80 | 12.04 |
| Dissolved Oxygen | EEA_3133-01-5 | 1,214 | mg{O2}/L | 0.25-58.50 | 4.36 |
| Nitrate | CAS_14797-55-8 | 1,289 | mg{NO3}/L | 0.07-188.24 | 12.65 |
| Conductance | EEA_3142-01-6 | 738 | uS/cm | 14.25-3496.67 | 407.48 |
| **Turbidity** | **NOT AVAILABLE** | 0 | N/A | N/A | N/A |

**Missing Data:** Turbidity (normally 13% weight in WQI calculation) is not available in the Kaggle dataset. Models are trained with explicit handling of this missing parameter.

## Feature Engineering

### Feature Categories

#### 1. Water Quality Features (5 raw + derived)
- **Raw Parameters:** pH, temperature, dissolved oxygen, nitrate, conductance
- **Derived Features:**
  - pH deviation from neutral (|pH - 7|)
  - DO-temperature ratio (oxygen saturation proxy)
  - Conductance categories (low/medium/high)
  - Nitrate pollution level (low/moderate/high/very high)

#### 2. Temporal Features
- Years since 1991 (baseline)
- Decade indicators (1990s, 2000s, 2010s)
- Year (for trend analysis)

#### 3. Geographic Features
- Country (one-hot encoded, top 10 + "Other")
- Water body type (River/Ground/Lake water)

#### 4. Environmental Context (19 features)
- Population density
- Marine protected areas
- Tourism statistics
- Climate indicators (droughts, floods, temperature)
- Literacy rate
- Renewable energy usage
- GDP

#### 5. Waste Management (10 features)
- Composition percentages (organic, glass, metal, plastic, etc.)
- Recycling rates

#### 6. Interaction Features
- Pollution stress index (high nitrate + low DO)
- Temperature stress (deviation from optimal)
- GDP per capita proxy

#### 7. Missing Value Indicators
- Binary flags for each missing water quality parameter
- Count of available parameters

**Total Features:** 69 (after one-hot encoding and feature engineering)

## Models

### 1. Water Quality Classifier

**Purpose:** Binary classification of water safety
**Target:** `is_safe` (True if WQI ≥ 70, False otherwise)
**Algorithm:** Random Forest Classifier

#### Hyperparameters (via GridSearchCV)
Tuned parameters:
- `n_estimators`: [100, 200, 300]
- `max_depth`: [10, 20, None]
- `min_samples_split`: [2, 5, 10]
- `min_samples_leaf`: [1, 2, 4]
- `class_weight`: ['balanced', None]

Cross-validation: 5-fold stratified

#### Performance Metrics
*To be filled after training completes*

**Expected Performance:**
- Accuracy: > 75%
- Precision: > 0.70
- Recall: > 0.70
- F1 Score: > 0.72
- ROC-AUC: > 0.80

#### Use Cases
- Binary safety classification for water sources
- Risk assessment for drinking water
- Prioritization of monitoring locations

### 2. WQI Prediction Regressor

**Purpose:** Continuous WQI score prediction and trend analysis
**Target:** `wqi_score` (0-100 continuous scale)
**Algorithm:** Random Forest Regressor

#### Hyperparameters (via GridSearchCV)
Tuned parameters:
- `n_estimators`: [100, 200, 300]
- `max_depth`: [10, 20, None]
- `min_samples_split`: [2, 5, 10]
- `min_samples_leaf`: [1, 2, 4]

Cross-validation: 5-fold

#### Performance Metrics
*To be filled after training completes*

**Expected Performance:**
- R² Score: > 0.60
- MAE: < 15 WQI points
- RMSE: < 20 WQI points

#### Use Cases
- Detailed WQI score prediction
- Temporal trend analysis
- Future water quality projection
- Comparative analysis across locations

## Data Preprocessing Pipeline

### 1. Missing Value Imputation
- Strategy: Median imputation for all numeric features
- Applied separately for training and inference
- Preserves feature distributions

### 2. Feature Scaling
- Method: StandardScaler (zero mean, unit variance)
- Applied after imputation
- Improves model convergence and performance

### 3. Categorical Encoding
- One-hot encoding for country and water body type
- Ordinal encoding for pollution levels
- Top N categories preserved, rest grouped as "Other"

## Model Files

Models are saved with timestamp versioning:

```
data/models/
├── classifier_YYYYMMDD_HHMMSS.joblib
├── regressor_YYYYMMDD_HHMMSS.joblib
├── metadata_YYYYMMDD_HHMMSS.json
└── MODEL_DOCUMENTATION.md (this file)
```

Each `.joblib` file contains:
- Trained model
- Fitted scaler
- Fitted imputer
- Feature names (in order)
- Training metrics
- Best hyperparameters
- Timestamp

## Usage

### Loading Models

```python
from src.models.model_utils import load_latest_models

# Load both models
classifier, regressor = load_latest_models()

# Or load specific model
from src.models import WaterQualityClassifier
classifier = WaterQualityClassifier.load('data/models/classifier_20251103_123456.joblib')
```

### Making Predictions

```python
import pandas as pd
import numpy as np

# Prepare features (must match training feature order)
# Features should be extracted using src.preprocessing.feature_engineering

# Classification prediction
y_pred = classifier.predict(X)  # Returns 0 (Unsafe) or 1 (Safe)
y_proba = classifier.predict_proba(X)  # Returns [P(Unsafe), P(Safe)]

# Regression prediction
wqi_pred = regressor.predict(X)  # Returns WQI scores (0-100)

# Trend prediction
trend_analysis = regressor.predict_trend(X, current_year=2024)
# Returns: {'trend': 'improving/stable/declining', 'current_wqi': ..., 'future_wqi': ...}
```

### Feature Requirements

**Critical:** Features must be provided in the exact order used during training. Use the `feature_names` attribute to verify order:

```python
print(classifier.feature_names)  # List of feature names in order
```

When applying to new data (e.g., US water quality data from USGS/WQP), ensure:
1. All 5 water quality parameters are extracted
2. Turbidity is set to None (or NaN)
3. Geographic features are mapped appropriately
4. Temporal features reflect the prediction date
5. Environmental context features are provided or imputed

## Known Limitations

### 1. Geographic Mismatch
- **Issue:** Trained on European data, applied to US data
- **Impact:** May not capture US-specific pollution patterns, regulations, or environmental conditions
- **Mitigation:** Models focus on universal chemical relationships (pH is pH everywhere)
- **Recommendation:** Use predictions as guidance, not absolute truth

### 2. Missing Turbidity Data
- **Issue:** One of six WQI parameters unavailable
- **Impact:** 13% of WQI calculation missing
- **Mitigation:** Models trained explicitly with turbidity=None
- **Recommendation:** WQI calculations automatically adjust weights

### 3. Temporal Extrapolation
- **Issue:** Training data ends 2017, predictions for 2024+
- **Impact:** 7+ years beyond training distribution
- **Mitigation:** Year included as feature to learn trends
- **Recommendation:** Add uncertainty estimates for future predictions

### 4. Missing Value Handling
- **Issue:** Many features have missing values (40-75% for some parameters)
- **Impact:** Imputation may introduce bias
- **Mitigation:** Median imputation + missing value indicators
- **Recommendation:** More data available = better predictions

### 5. Class Imbalance (Classifier Only)
- **Issue:** Training data has ~53% unsafe, ~47% safe (relatively balanced)
- **Impact:** May slightly favor unsafe predictions
- **Mitigation:** Class weighting during training
- **Recommendation:** Consider adjusting decision threshold if needed

## Feature Importance

*To be filled after training completes*

Expected top features:
1. Water quality parameters (pH, DO, nitrate, conductance)
2. Temporal trends (year, decade)
3. Geographic context (country, water body type)
4. Environmental stress indicators

## Model Validation

### Training/Validation/Test Split
- Training: 60% (stratified for classifier)
- Validation: 20%
- Test: 20% (held out, never seen during training)

All reported metrics are on the **held-out test set**.

### Cross-Validation
- Method: 5-fold (stratified for classifier)
- Used for hyperparameter tuning only
- Test set remains completely separate

## Future Improvements

### High Priority
1. **Add US training data** - Would dramatically improve performance
2. **Turbidity estimation** - Proxy from other parameters
3. **Regional calibration** - Adjust predictions by US region
4. **Uncertainty quantification** - Confidence intervals on predictions

### Medium Priority
5. **Feature engineering** - More domain-specific interactions
6. **Ensemble methods** - Combine multiple model types
7. **Time series modeling** - Better trend predictions
8. **Transfer learning** - Fine-tune on US data when available

### Low Priority
9. **Deep learning** - If much more data becomes available
10. **Real-time updates** - Online learning from new data

## Responsible Use

### Do's
✓ Use as a screening tool for water quality assessment
✓ Combine with local domain knowledge
✓ Consider predictions as estimates with uncertainty
✓ Validate against known water quality measurements
✓ Update models when more data becomes available

### Don'ts
✗ Don't treat predictions as definitive truth
✗ Don't ignore local water quality expertise
✗ Don't apply to contexts vastly different from training data
✗ Don't make critical decisions based solely on model output
✗ Don't assume perfect accuracy or generalization

## Technical Details

### Dependencies
- Python 3.11+
- scikit-learn 1.3+
- pandas 2.0+
- numpy 1.24+
- joblib 1.3+

### Computational Requirements
- Training: 4-8 CPU cores recommended, ~15-30 minutes
- Inference: Minimal (< 1 second per batch)
- Memory: ~500MB for loaded models

### Reproducibility
- Random seed: 42 (used throughout)
- GridSearchCV: deterministic with seed
- Train/test split: stratified + seeded

## References

1. Kaggle European Water Quality Dataset
2. NSF Water Quality Index (WQI) methodology
3. EPA Water Quality Standards
4. USGS National Water Information System (NWIS)
5. Water Quality Portal (WQP) - EPA/USGS/USDA collaboration

## Contact & Support

For questions about model usage, interpretation, or issues:
- See README.md in project root
- Check `.claude/plan.md` for implementation details
- Review source code in `src/models/`

## Version History

### v1.0 (2025-11-03)
- Initial model training on Kaggle European dataset
- RandomForest classifier and regressor
- 69 features including water quality, temporal, and contextual
- GridSearchCV hyperparameter tuning
- Comprehensive evaluation metrics

---

*This documentation will be updated after model training completes with actual performance metrics and feature importances.*
