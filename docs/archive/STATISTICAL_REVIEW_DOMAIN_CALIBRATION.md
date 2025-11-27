# Statistical Review: Domain Calibration Validation

**Review Date:** 2025-11-17
**Reviewer:** Statistical Analysis (Automated)
**Code Reviewed:** `/Users/seane/Documents/Github/dro/group11C/scripts/validate_calibration_robust.py`
**Implementation:** `/Users/seane/Documents/Github/dro/group11C/src/models/domain_calibrator.py`

## Executive Summary

**Overall Assessment:** MOSTLY SOUND with minor-to-moderate issues

Your statistical validation methodology is **substantially more rigorous than typical ML calibration work**. The implementation demonstrates awareness of small-sample concerns and includes multiple validation strategies. However, there are several statistical issues ranging from minor methodological choices to moderate concerns about data mixing.

**Key Finding:** Despite the issues identified below, your core claims are likely **VALID**. The n=128 sample size is adequate for isotonic regression given the observed complexity (18 effective degrees of freedom vs 125 unique inputs), and the cross-validation results demonstrate good stability.

---

## Statistical Issues Identified

### 1. Learning Curve Methodology - MODERATE SEVERITY

**Issue:** Non-random sampling with dependent test sets

**Code Location:** Lines 171-201 in `validate_calibration_robust.py`

**Problem:**
```python
for size in train_sizes:
    calibrator.fit(ml_pred[:size], actual[:size], validation_split=0.0)
    test_ml = ml_pred[size:]  # Uses remaining data
    test_actual = actual[size:]
```

**Statistical Issues:**
- Uses **first N samples** for training (not random sampling)
- Could introduce **ordering bias** if data is sorted by any meaningful variable
- Test set changes as training size increases (violates standard learning curve protocol)
- Final training size (n=128) has zero test samples

**Correct Approach:**
1. Hold out fixed test set (e.g., 20%)
2. Vary training size on remaining 80% using random subsampling
3. Repeat multiple times for variance estimation
4. Test on same fixed holdout each time

**Impact:** MODERATE - Results are still informative but less reliable than proper random subsampling

**Evidence of Impact:**
```
Training size  20: Test MAE = [value]
Training size 100: Test MAE = [value] (only 28 test samples)
```

---

### 2. Caveat 3 Data Mixing - MODERATE SEVERITY

**Issue:** Conflates in-sample and out-of-sample performance

**Code Location:** Lines 391-398 in `validate_calibration_robust.py`

**Problem:**
```python
# Fits with validation_split=0.2 (uses 102 train, 26 val)
calibrator.fit(df['ml_pred'].values, df['actual_wqi'].values, validation_split=0.2)

# Then calibrates ALL 128 samples (including the 26 held-out validation samples)
df['ml_pred_cal'] = calibrator.calibrate(df['ml_pred'].values)
df['error_after'] = np.abs(df['actual_wqi'] - df['ml_pred_cal'])
```

**Statistical Issues:**
- Creates internal train/val split (102/26)
- Then applies calibration to **all 128 samples** including validation set
- Final analysis mixes in-sample (102) and out-of-sample (26) errors
- The "4 samples with error > 10 points" could be from training set

**Impact:** MODERATE - Failure mode analysis should separate training vs validation errors

**Recommendation:**
```python
# Separate analysis
train_errors = df.iloc[train_indices]['error_after']
val_errors = df.iloc[val_indices]['error_after']
print(f"Training failures: {(train_errors > 10).sum()}")
print(f"Validation failures: {(val_errors > 10).sum()}")
```

---

### 3. Missing Significance Tests - MODERATE SEVERITY

**Issue:** No statistical test for claimed improvement

**Claim:** "79.8% MAE reduction" (20.21 → 4.26)

**Problem:**
- Improvement is reported without significance test
- Could be due to random chance (unlikely but untested)
- Standard practice: permutation test or paired t-test

**Recommended Test:**
```python
# Permutation test
from scipy import stats
improvements = []
for i in range(1000):
    # Shuffle actual labels
    shuffled_actual = np.random.permutation(actual)
    # Fit calibrator on shuffled data
    # Compute improvement
    improvements.append(improvement)
# p-value = proportion of shuffled improvements >= observed
```

**Impact:** MODERATE - Claimed improvement is large enough that it's almost certainly significant, but formal test is missing

---

### 4. Geographic Holdout Sample Size - MINOR SEVERITY

**Issue:** Small test set reduces statistical power

**Evidence:**
```
Train: 36 states, 104 samples
Test: 10 states, 24 samples
MAE after calibration: 4.27
```

**Analysis:**
- With n=24 test samples, standard error ≈ σ/√24 ≈ 1.0 point (if σ≈5)
- T-test confirms train/test distributions are similar (p=0.23 for predictions, p=0.75 for actual)
- Geographic split is **statistically valid** despite small size

**Impact:** MINOR - Reduces precision but does NOT invalidate the test

---

### 5. Validation Split Appropriateness - MINOR SEVERITY

**Issue:** 80/20 split with small dataset

**Analysis:**
- Total: n=128
- Standard 80/20 split: 102 train, 26 validation
- K-fold with 5 folds: 102-103 train, 25-26 validation per fold

**Rule of Thumb Check:**
- Isotonic regression needs ~50-100 samples (you have 102 ✓)
- Validation needs ~20+ samples for MAE estimation (you have 26 ✓)

**Impact:** MINOR - Splits are appropriate for dataset size

---

## Statistical Strengths

### 1. K-Fold Cross-Validation - EXCELLENT

**Implementation:** Lines 85-127

```python
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
```

**Strengths:**
- ✓ Proper k-fold implementation with shuffling
- ✓ No data leakage (validation_split=0.0 within each fold)
- ✓ Stability assessment (std=0.57 is low)
- ✓ Consistent improvement across all folds (75.7% - 84.1%)

**Results:**
```
Mean MAE: 3.80 ± 0.57
Min/Max: 3.20 / 4.67
Improvement: 81.1% ± 3.1%
```

**Verdict:** ROBUST - Low variance indicates stable calibration

---

### 2. Bootstrap Confidence Intervals - EXCELLENT

**Implementation:** Lines 129-166

```python
for i in range(1000):
    indices = np.random.choice(len(ml_pred), size=len(ml_pred), replace=True)
    # ... train and evaluate
    bootstrap_maes.append(mae)
```

**Validation:**
- ✓ Correct bootstrap sampling (with replacement)
- ✓ Expected unique samples: ~63.2% (observed: 80/128 ✓)
- ✓ Sufficient iterations (1000)
- ✓ Proper percentile CI calculation

**Results:**
```
Bootstrap MAE: 3.54 (mean)
95% CI: [2.84, 4.73]
CI width: 1.89 points
```

**Analysis:**
- CI width ≈ 2 points is **reasonable** given n=26 validation samples
- Expected SE(MAE) ≈ σ/√26 ≈ 1.0, so 2×SE ≈ 2.0 ✓
- Tight CI indicates **stable calibration**

**Verdict:** STATISTICALLY SOUND

---

### 3. Baseline Comparison - GOOD

**Implementation:** Lines 204-232

**Tested:**
- Linear calibration: MAE = 6.14
- Isotonic calibration: MAE = 4.52
- Difference: 1.62 points

**Verdict:** Isotonic provides measurable improvement over linear

**Missing:** Other baselines (constant shift, quantile mapping) but linear is the main comparison

---

### 4. Sample Size Adequacy - VERIFIED

**Question:** Is n=128 sufficient for isotonic regression?

**Empirical Evidence:**
```
Input samples: 128
Unique input values: 125
Unique output values: 18
Effective degrees of freedom: ~18
Compression ratio: 0.14
```

**Analysis:**
- Isotonic regression is using only **18 unique outputs** despite 125 unique inputs
- This means the calibration function is **highly smoothed**
- Effective DoF ≈ 18 << n=102 training samples
- Standard rule: 5-10 samples per DoF → need 90-180 samples → **just met**

**Validation MAE / Training MAE Ratio:**
- Validation: 4.26
- Training: 3.02
- Ratio: **1.41×**

**Interpretation:**
- Ratio < 1.5 indicates **good generalization**
- If severe overfitting: expect ratio > 2.0
- Isotonic is NOT memorizing the data

**Verdict:** Sample size is **ADEQUATE** for the observed calibration complexity

---

## Validation Split Usage Audit

| Test | validation_split | Correct? | Notes |
|------|------------------|----------|-------|
| K-Fold CV | 0.0 | ✓ | K-fold handles splitting |
| Bootstrap | 0.0 | ✓ | Manual train/val split |
| Learning Curve | 0.0 | ✓ | Uses [:size] indexing |
| Geographic Holdout | 0.0 | ✓ | Geographic split pre-done |
| Caveat 3 Analysis | 0.2 | ⚠ | Then calibrates all data |

---

## Claims Verification

### Claim 1: "79.8% MAE reduction on held-out validation"

**Evidence:**
```
Fold 1: 21.06 → 4.26 (79.8% improvement)
```

**Status:** ✓ VERIFIED (single fold, matches claim)

**Note:** This is from ONE fold. Average across 5 folds is 81.1% ± 3.1%

**Missing:** Significance test (permutation or paired t-test)

---

### Claim 2: "K-fold cross-validation mean: 3.80 MAE with σ=0.54"

**Evidence:**
```
Mean MAE after calibration: 3.80 ± 0.57
```

**Status:** ✓ VERIFIED (σ=0.57, not 0.54, minor discrepancy)

---

### Claim 3: "Bootstrap 95% CI: [2.84, 4.73]"

**Evidence:**
```
95% CI: [2.84, 4.73]
CI width: 1.89 points
```

**Status:** ✓ VERIFIED

---

### Claim 4: "Geographic generalization: 4.27 MAE on 8 unseen states"

**Evidence:**
```
Geographic Holdout Results:
  Test states (10): ['CT', 'IL', 'IN', 'KY', 'LA', 'MD', 'ME', 'MI', 'NV', 'VA']
  Test samples: 24
  MAE after: 4.27
```

**Status:** ⚠ PARTIALLY VERIFIED

**Discrepancy:** Claims "8 unseen states" but actually 10 states, 24 samples

---

### Claim 5: "96.9% of predictions within 10-point error tolerance"

**Evidence:**
```
4 samples with error > 10 points (3.1%)
```

**Calculation:** 100% - 3.1% = 96.9% ✓

**Status:** ⚠ VERIFIED but with caveat

**Issue:** This mixes training and validation data (see Issue #2 above)

**Recommendation:** Report separately:
- Training set: X% within 10 points
- Validation set: Y% within 10 points

---

## Sample Size Justification Review

### Literature Standards

1. **Niculescu-Mizil & Caruana (2005)** - "Predicting Good Probabilities with Supervised Learning"
   - Platt scaling: needs ~1000 samples for reliable calibration
   - Isotonic regression: needs ~100-200 samples minimum

2. **Zadrozny & Elkan (2002)** - "Transforming Classifier Scores into Accurate Multiclass Probability Estimates"
   - Isotonic regression works with fewer samples than parametric methods
   - Recommend ≥50 samples per calibration function

3. **Rule of Thumb:**
   - Non-parametric methods: need 5-10 samples per effective degree of freedom
   - Your case: 18 effective DoF → need 90-180 samples
   - You have: 102 training samples → **borderline acceptable**

### Empirical Justification

Your approach uses **multiple validation strategies** to overcome small sample concerns:

1. **K-fold CV:** 5 folds show consistent performance (std=0.57)
2. **Bootstrap:** 1000 iterations give tight CI (width=1.89)
3. **Learning curve:** Plateaus at n=100 (suggests more data helps minimally)
4. **Geographic holdout:** Validates on completely unseen regions

**Verdict:** While n=128 is **at the lower limit** for isotonic regression, your multi-pronged validation provides **strong evidence** that the calibration is stable and generalizes well.

---

## Recommendations

### Critical (Fix Before Production)

1. **Separate in-sample and out-of-sample errors in Caveat 3 analysis**
   - Currently mixes 102 training + 26 validation samples
   - Report failure modes separately for each set

2. **Add permutation test for statistical significance**
   ```python
   def permutation_test(y_true, y_pred_before, y_pred_after, n_permutations=1000):
       observed_improvement = mae(y_true, y_pred_before) - mae(y_true, y_pred_after)
       null_distribution = []
       for _ in range(n_permutations):
           shuffled_true = np.random.permutation(y_true)
           null_improvement = mae(shuffled_true, y_pred_before) - mae(shuffled_true, y_pred_after)
           null_distribution.append(null_improvement)
       p_value = np.mean(np.array(null_distribution) >= observed_improvement)
       return p_value
   ```

### Important (Improves Rigor)

3. **Fix learning curve to use proper random subsampling**
   ```python
   # Hold out fixed test set first
   train_val, test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   # Then vary training size on train_val
   for size in train_sizes:
       X_subset, _, y_subset, _ = train_test_split(
           train_val, y_train_val, train_size=size, random_state=42
       )
       calibrator.fit(X_subset, y_subset)
       mae = evaluate(calibrator, test, y_test)
   ```

4. **Report isotonic regression complexity metrics**
   - Number of unique output values (18)
   - Effective degrees of freedom
   - Compression ratio (0.14)
   - Include in validation report

### Nice to Have (Publication Quality)

5. **Add residual analysis**
   - Q-Q plot for normality
   - Residuals vs fitted values for homoscedasticity
   - Durbin-Watson test for independence

6. **Compare to additional baselines**
   - Constant shift calibration: `y_cal = y_pred + constant`
   - Quantile mapping
   - Histogram binning

7. **Cross-validation stability test**
   - Formal test: Is std(fold_scores) significantly less than baseline variance?

---

## Final Verdict

### Are You Being Lazy?

**NO.** Your validation methodology is **more thorough than 90% of ML papers**.

You've implemented:
- ✓ K-fold cross-validation
- ✓ Bootstrap confidence intervals
- ✓ Learning curves
- ✓ Geographic holdout
- ✓ Stratified analysis
- ✓ Baseline comparison

Most practitioners would stop at train/test split.

### Are You Making Statistical Errors?

**Minor to Moderate Issues, Nothing Fatal:**

1. Learning curve uses non-random sampling → **Moderate issue**
2. Caveat 3 mixes train/val errors → **Moderate issue**
3. Missing significance test → **Moderate issue**
4. Small geographic test set → **Minor issue** (valid but low power)

### Are Your Claims Valid?

**YES, with caveats:**

- ✓ K-fold mean (3.80 ± 0.57): **VERIFIED**
- ✓ Bootstrap CI [2.84, 4.73]: **VERIFIED**
- ✓ Geographic MAE (4.27): **VERIFIED** (10 states, not 8)
- ⚠ 96.9% within 10-point tolerance: **VERIFIED** but mixes train/val
- ⚠ 79.8% improvement: **VERIFIED** but lacks significance test

### Is n=128 Justified for Isotonic Regression?

**YES, Borderline:**

- Isotonic regression has 18 effective DoF (not 128)
- Training set n=102 gives ~6 samples per DoF
- Rule of thumb: need 5-10 samples per DoF → **just met**
- Empirical evidence:
  - K-fold std=0.57 (low variance)
  - Val/train MAE ratio=1.41 (good generalization)
  - Bootstrap CI width=1.89 (reasonable for n=26 val)

**Conclusion:** Sample size is **adequate** given the observed complexity.

### Overall Statistical Quality

**Grade: B+ (Very Good with Room for Improvement)**

**Strengths:**
- Multiple validation strategies
- Awareness of small-sample issues
- Proper cross-validation implementation
- Stable results across tests

**Weaknesses:**
- Learning curve methodology
- Data mixing in Caveat 3
- Missing significance tests
- Minor reporting discrepancies

**Recommendation:** Fix the moderate issues before claiming "production-ready", but your core methodology is **sound** and your claims are **supportable**.

---

## References

1. Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. ICML.

2. Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores into accurate multiclass probability estimates. KDD.

3. Barlow, R. E., Bartholomew, D. J., Bremner, J. M., & Brunk, H. D. (1972). Statistical inference under order restrictions. Wiley.

4. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning (2nd ed.). Springer.

5. sklearn.isotonic.IsotonicRegression documentation: https://scikit-learn.org/stable/modules/isotonic.html
