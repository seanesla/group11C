# Production Deployment Checklist: US-Only WQI Model

**Model**: US-Only RandomForest (128 samples, MAE 1.98, R² 0.81)
**Status**: 🔴 **BLOCKED - Critical Issues Identified**
**Risk Level**: HIGH
**Date**: 2025-11-17

---

## Deployment Decision: ❌ **DO NOT DEPLOY**

The US-only model is **NOT READY** for production due to critical data quality and robustness issues. See [US_MODEL_FAILURE_MODE_ANALYSIS.md](US_MODEL_FAILURE_MODE_ANALYSIS.md) for complete analysis.

---

## Critical Blockers (Must Fix Before Deployment)

### 🔴 Blocker #1: Data Quality - Extreme Outliers

**Issue**: Training data contains physically impossible values that corrupt model learning.

| Parameter | Outlier Value | Physical Reality | Impact |
|-----------|---------------|------------------|--------|
| DO | 81.20 mg/L | Max possible ~15 mg/L | Model learns wrong DO-WQI relationship |
| Nitrate | 1,400 mg/L | 140× EPA MCL | Only 2 samples >20 mg/L, poor calibration |
| Conductance | 49,142 µS/cm | 20× seawater | Model trained on sensor errors |
| Temperature | 59.53°C | Near boiling | 4 samples >30°C, wrong thermal effects |

**Fix Required**:
- [ ] Remove samples with DO >15 mg/L (1 sample)
- [ ] Remove samples with nitrate >50 mg/L (2 samples)
- [ ] Remove samples with conductance >3,000 µS/cm (13 samples)
- [ ] Remove samples with temperature >40°C (4 samples)
- [ ] Retrain with ~110 clean samples

**Priority**: P0 (immediate, deployment blocker)

---

### 🟡 Blocker #2: Feature Engineering - Data Artifacts (CODE FIX IMPLEMENTED)

**Issue**: Top feature (30% importance) is `turbidity_missing`, a data collection pattern, not water quality signal.

**Why This is Dangerous**:
- Model learned: "turbidity missing → high WQI" (spurious correlation)
- If contaminated site ALSO doesn't measure turbidity → model predicts SAFE (wrong!)
- 30% of predictive power comes from artifact, not chemistry

**Example Failure**:
```
Input: pH=6.5, DO=4.0, Nitrate=15, Conductance=2000, Turbidity=MISSING
Ground Truth: WQI=55 (UNSAFE - multiple violations)
Model Prediction: WQI=75 (SAFE - because turbidity_missing=True)
Confidence: 85% (certain it's right, but wrong)
```

**Fix Implemented (2024-12-02)**:
- [x] Artifact features excluded when >95% missing (e.g., `turbidity_missing` now skipped)
- [x] KNN imputation with pre-scaling (5 neighbors, distance-weighted) replaces SimpleImputer
- [x] IQR-based outlier detection (3x multiplier) added to pipeline
- [x] `class_weight='balanced'` enforced on classifier

**Remaining**:
- [ ] Retrain models with new pipeline (`poetry run python train_models.py --core-params-only`)
- [ ] Verify feature importance is chemistry-driven (no `*_missing` features in top 10)

**Priority**: P0 (code implemented, pending retraining)

---

### 🔴 Blocker #3: No Input Validation

**Issue**: API accepts any input values, including impossible ones.

**Current Behavior**:
- pH = -5 → predicts WQI (nonsense)
- DO = 100 mg/L → predicts WQI (impossible)
- Nitrate = 10,000 mg/L → predicts WQI (toxic waste)

**Fix Required**:
- [ ] Implement strict input validation:
  ```python
  VALID_RANGES = {
      'ph': (0, 14),
      'dissolved_oxygen': (0, 15),
      'temperature': (-5, 40),
      'turbidity': (0, 100),
      'nitrate': (0, 50),
      'conductance': (0, 3000)
  }
  ```
- [ ] Reject predictions for out-of-range inputs
- [ ] Return error messages explaining valid ranges
- [ ] Log all validation failures

**Priority**: P0 (immediate, deployment blocker)

---

### 🔴 Blocker #4: Class Imbalance - Only 1.2% UNSAFE Samples

**Issue**: Training data is 98.8% SAFE, 1.2% UNSAFE (vs 15-20% real-world violation rate).

**Impact**:
- Model rarely sees contaminated water during training
- Overconfident in predicting SAFE for borderline cases
- No samples with WQI <50 (POOR category never learned)

**Fix Required**:
- [ ] Collect 30-50 samples from known contaminated sites (WQI <70)
- [ ] Target industrial zones, agricultural runoff, mining areas
- [ ] Retrain with balanced dataset (minimum 15% UNSAFE)

**Priority**: P0 (immediate, deployment blocker)

---

## High-Priority Improvements (Before Deployment)

### 🟠 Geographic Coverage - 77% of States Underrepresented

**Issue**: 37/48 states have <5 samples each.

| Region | Risk | Underrepresented States |
|--------|------|------------------------|
| Appalachia | Acid mine drainage (pH 3-5) | WV, KY, TN |
| Mountain West | High altitude, low temp | MT, WY, ID |
| Gulf Coast | Saltwater intrusion | MS, AL, LA |
| Southwest | High conductance (arid) | AZ, NM, UT |

**Fix Required**:
- [ ] Minimum 10 samples per state before deployment in that state
- [ ] Flag predictions for underrepresented states as "low confidence"
- [ ] Document geographic limitations clearly

**Priority**: P1 (high priority, partial blocker)

---

### 🟠 Temporal Coverage - Unknown Seasonal Distribution

**Issue**: Training data timestamps unknown - could be all summer samples.

**Seasonal Risks**:
- **Winter**: Ice cover → low DO, low temp (0 samples <0°C in training)
- **Spring**: Snowmelt runoff → high turbidity, nitrate spikes
- **Summer**: Algae blooms → DO swings, pH fluctuation
- **Fall**: Leaf litter → pH drop

**Fix Required**:
- [ ] Audit training data for seasonal distribution
- [ ] Collect validation samples from all 4 seasons
- [ ] Add seasonal disclaimer to predictions

**Priority**: P1 (high priority, partial blocker)

---

### 🟠 Overfitting Risk - 128 Samples, Complex Model

**Issue**: RandomForest (100 trees, max_depth=10) may memorize, not generalize.

**Evidence**:
- In-sample MAE: 1.98
- Cross-validation MAE: 3.07
- Ratio: 1.55× (55% degradation on unseen data)

**Fix Required**:
- [ ] Reduce complexity: max_depth=5, n_estimators=50
- [ ] Increase min_samples_leaf=10 (7.8% of data)
- [ ] Target n>200 samples before deploying complex model
- [ ] Return prediction uncertainty (not just point estimates)

**Priority**: P1 (high priority)

---

## Medium-Priority Improvements (Post-Deployment Monitoring)

### 🟡 No Prediction Uncertainty Quantification

**Issue**: All predictions appear equally reliable (no confidence intervals).

**Fix**:
- [ ] Implement out-of-bag variance for RandomForest
- [ ] Return [lower, prediction, upper] confidence intervals
- [ ] Flag predictions with high variance as "low confidence"

**Priority**: P2 (important for user trust)

---

### 🟡 No Model Version Control

**Issue**: Streamlit loads "latest model" - no rollback if performance degrades.

**Fix**:
- [ ] Implement semantic versioning (v1.0.0, v1.1.0)
- [ ] Add model registry with validation metrics
- [ ] Require manual promotion to production
- [ ] Add A/B testing capability

**Priority**: P2 (operational stability)

---

### 🟡 No Production Monitoring

**Issue**: No tracking of prediction quality or input distribution drift.

**Fix**:
- [ ] Log all predictions with input features
- [ ] Monitor input distribution vs training (KL divergence alert if >0.5)
- [ ] Track prediction variance (alert if avg >5)
- [ ] Weekly/monthly performance reports

**Priority**: P2 (operational visibility)

---

## Edge Case Testing Requirements

Before deployment, verify model behavior on these scenarios:

### Test Suite 1: Extreme Contamination

- [ ] **Acid mine drainage**: pH=4.0, DO=2.0, Conductance=3000 → should predict UNSAFE
- [ ] **Agricultural runoff**: Nitrate=25, DO=5.0 → should predict POOR/UNSAFE
- [ ] **Industrial pollution**: pH=5.0, Conductance=5000 → should predict UNSAFE
- [ ] **Hypoxic water**: DO=1.0 → should predict UNSAFE

### Test Suite 2: Seasonal Events

- [ ] **Winter ice cover**: DO=1.5, Temp=1°C → should predict UNSAFE
- [ ] **Spring runoff**: Turbidity=50, Nitrate=15 → should predict POOR
- [ ] **Summer algae bloom**: DO=2.0 (morning), DO=15 (afternoon) → both handled correctly
- [ ] **Fall turnover**: DO=6.0, Temp=10°C → should predict FAIR/UNSAFE

### Test Suite 3: Geographic Extremes

- [ ] **Coastal saltwater intrusion**: Conductance=15,000 → should warn
- [ ] **Desert arid**: Conductance=2000, Temp=35°C → should handle correctly
- [ ] **Mountain stream**: DO=12, Temp=5°C → should predict SAFE (if pH normal)
- [ ] **Groundwater well**: DO=3.0, Temp=15°C, Conductance=800 → mixed quality

### Test Suite 4: Invalid Inputs

- [ ] **Impossible values**: DO=100, pH=20, Temp=-50 → should REJECT with error
- [ ] **Sensor errors**: DO=81 (like training outlier) → should flag as error
- [ ] **Missing all parameters** → should return error, not prediction
- [ ] **Partial data**: Only 3/6 parameters → should return low-confidence prediction

---

## Recommended Deployment Path

### ❌ Option 1: Deploy Current Model (NOT RECOMMENDED)

**Pros**: Fast (can deploy today)
**Cons**:
- High failure risk (outliers, data artifacts, overfitting)
- Will fail silently on edge cases
- User trust damage when failures discovered

**Verdict**: ❌ **Do not proceed**

---

### ✅ Option 2: Use WQI Calculator Until Model is Ready (RECOMMENDED)

**Pros**:
- Physics-based, interpretable, reliable
- No training data bias
- Works on all inputs (no edge cases)

**Cons**:
- No ML benefits (learning from data)
- Limited to 6 parameters

**Timeline**: Immediate (already implemented)
**Verdict**: ✅ **Use this as fallback**

---

### ⚠️ Option 3: Deploy EU Calibrated Model (ACCEPTABLE COMPROMISE)

**Pros**:
- Larger training set (2,939 samples)
- Lower overfitting risk
- More robust to edge cases

**Cons**:
- Higher MAE (3.54 vs 1.98)
- Low-DO failures (MAE 9.35 on DO <6 mg/L)
- Domain shift issues

**Timeline**: Immediate (already trained)
**Verdict**: ⚠️ **Acceptable short-term, with disclaimers**

---

### ✅ Option 4: Retrain US Model with Clean Data (BEST LONG-TERM)

**Steps**:
1. **Week 1**: Data cleaning (remove outliers, fix features)
2. **Week 2**: Retrain and validate (expect MAE ~3-4 with clean data)
3. **Week 3**: Implement input validation and monitoring
4. **Week 4**: Deploy with clear limitations documented

**Pros**:
- Addresses all critical blockers
- Clean foundation for future improvement
- Realistic performance expectations

**Timeline**: 4 weeks
**Verdict**: ✅ **Recommended path**

---

## Final Recommendation

### Short-Term (Next 1-4 Weeks):

1. **Use WQI Calculator** as primary prediction method
2. **Deploy EU calibrated model** as "experimental ML prediction" (with disclaimer)
3. **Clean US training data** and retrain
4. **Implement input validation** layer
5. **Add monitoring dashboard**

### Medium-Term (1-3 Months):

1. **Collect 100 additional samples**:
   - 30 UNSAFE samples (industrial, agricultural)
   - 20 low-DO samples (eutrophic lakes, winter)
   - 20 high-nitrate samples (agricultural runoff)
   - 30 underrepresented states

2. **Retrain with 200+ samples**
3. **Validate on held-out test sets** (geographic, seasonal)

### Long-Term (3-6 Months):

1. **Ensemble model**: US + EU + WQI Calculator (weighted by confidence)
2. **Continuous learning**: Quarterly retraining with production data
3. **Regional models**: NE, SE, MW, SW, W (trained on local data)

---

## Sign-Off Checklist

Before deploying **ANY** model to production, verify:

- [ ] ✅ All critical blockers resolved (Blockers #1-4 above)
- [ ] ✅ Input validation implemented and tested
- [ ] ✅ Prediction uncertainty quantified (confidence intervals)
- [ ] ✅ Edge case testing passed (all 4 test suites)
- [ ] ✅ Geographic limitations documented
- [ ] ✅ Seasonal limitations documented
- [ ] ✅ Model version control implemented
- [ ] ✅ Production monitoring active
- [ ] ✅ Fallback mechanism tested (WQI Calculator)
- [ ] ✅ User documentation includes limitations and disclaimers

**Required Approvals**:
- [ ] Data Scientist: Model performance acceptable
- [ ] ML Engineer: Production infrastructure ready
- [ ] Domain Expert: Predictions align with water chemistry
- [ ] Product Manager: User experience and limitations clear
- [ ] Reliability Engineer: Failure modes understood and mitigated

---

## Current Status: 🔴 **DEPLOYMENT BLOCKED**

**Blocking Issues**: 4 critical blockers unresolved
**Earliest Deployment Date**: 4 weeks (after data cleaning and retraining)
**Recommended Action**: Use WQI Calculator or EU calibrated model until US model is retrained

---

**Last Updated**: 2025-11-17
**Review By**: Reliability Engineering Team
**Next Review**: After data cleaning (Week 1)

---

## Related Documents

- [Full Failure Mode Analysis](US_MODEL_FAILURE_MODE_ANALYSIS.md) - Complete technical analysis
- [Agent 12 Validation](AGENT_12_VALIDATION.md) - US-only vs Calibration comparison
- [WQI Standards](WQI_STANDARDS.md) - EPA/WHO reference standards
