# Executive Summary: Peer Review of Statistical Methods

**Paper**: Fair Model Comparison - US-Only vs Calibrated EU Water Quality Models
**Reviewer Role**: Statistical Methods Validation for ML Journal
**Review Date**: November 17, 2025

---

## TL;DR

✅ **ACCEPT WITH MINOR REVISIONS**

The authors properly corrected a serious methodological flaw (mixing in-sample vs. out-of-sample evaluation) using rigorous 10-fold cross-validation. All 4 statistical tests confirm significance (p<0.05). The revised 15.6% improvement claim is valid, though much smaller than the original flawed 44% claim. Minor revisions needed to address normality assumption and practical significance discussion.

---

## Original vs. Corrected Comparison

| Aspect | Original (FLAWED) | Corrected (VALID) |
|--------|-------------------|-------------------|
| **US Model Evaluation** | In-sample (MAE=1.98) | 10-fold CV (MAE=3.07) |
| **Calibrated Model Evaluation** | Mixed train/val (MAE=3.54) | 10-fold CV (MAE=3.64) |
| **Claimed Improvement** | 44% | 15.6% |
| **Inflation Factor** | - | 2.8× |
| **Methodological Validity** | ❌ Invalid | ✅ Valid |

---

## Peer Review Checklist Results

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | Cross-validation for both models | ✅ PASS | Both use 10-fold CV, identical protocol |
| 2 | Same random seed/splits | ✅ PASS | Both use random_state=42, KFold splits |
| 3 | Paired t-test assumptions | ⚠️ QUESTIONABLE | Normality violated (2/3 tests), but n=128 adequate |
| 4 | Wilcoxon test correct | ✅ PASS | p=0.002884, non-parametric alternative |
| 5 | Permutation test (10K iter) | ✅ PASS | p=0.0138, distribution-free |
| 6 | Bootstrap CI excludes zero | ✅ PASS | 95% CI: [0.11, 0.99], all positive |
| 7 | All p-values < 0.05 | ✅ PASS | All 4 tests achieve significance |
| 8 | Effect size matches claim | ✅ PASS | 15.6% improvement verified |

**Score**: 7.5/8 ✅

---

## Statistical Test Summary

| Test | p-value | Significant? | Notes |
|------|---------|--------------|-------|
| Paired t-test | 0.012424 | ✅ YES | Normality questionable but acceptable |
| Wilcoxon signed-rank | 0.002884 | ✅ YES | Strongest result, non-parametric |
| Permutation (10,000) | 0.0138 | ✅ YES | Distribution-free, robust |
| Bootstrap 95% CI | [0.11, 0.99] | ✅ YES | Excludes zero entirely |

**Convergent Evidence**: All 4 independent tests reach the same conclusion.

---

## Key Strengths

1. ✅ **Methodological Rigor**: Proper 10-fold CV eliminates in-sample bias
2. ✅ **Fair Comparison**: Identical train/test splits for both models
3. ✅ **Multiple Tests**: Parametric + non-parametric alternatives
4. ✅ **Transparency**: Honest acknowledgment of original flaw (2.8× inflation)
5. ✅ **Reproducibility**: Code, data, and results fully documented
6. ✅ **Scientific Integrity**: Reduced claim from 44% to 15.6%

---

## Concerns Identified

### Major Concerns: NONE ✅

### Minor Concerns:

1. ⚠️ **Normality Assumption**:
   - Shapiro-Wilk test: p=0.000905 (rejects normality)
   - D'Agostino-Pearson: p=0.005 (rejects normality)
   - **BUT**: Wilcoxon and permutation tests confirm result (robust to violation)

2. ⚠️ **Multiple Comparisons**:
   - 4 tests performed without Bonferroni correction
   - **BUT**: All test same hypothesis (convergent evidence, not independent tests)
   - **POST-CORRECTION**: Wilcoxon (p=0.003) still significant at α=0.0125

3. ⚠️ **Practical Significance**:
   - Absolute improvement: 0.57 WQI points (on 0-100 scale)
   - **QUESTION**: Is this meaningful for water quality management?
   - Statistical significance ≠ practical importance

4. ⚠️ **Sample Size**:
   - n=128 samples (modest for CV)
   - **BUT**: Adequate for all tests used, low variance across folds

---

## Critical Verifications Performed

### 1. No In-Sample Contamination ✅
- **Verified**: Every sample predicted by model that NEVER saw it in training
- **US model**: Lines 141-153 train on fold_train, predict on fold_val
- **Calibrated model**: Lines 208-223 train on fold_train, predict on fold_val

### 2. Identical Train/Test Splits ✅
- **Verified**: Same KFold(random_state=42) for both models
- **Reproduced**: Independent re-implementation matched results (MAE=3.64 vs 3.64)

### 3. Paired Tests Correctly Applied ✅
- **Verified**: Tests compare 128 paired errors (per-sample), not 10 fold MAEs
- **Code**: `stats.ttest_rel(errors_cal, errors_us)` on 128-element arrays

### 4. Effect Size Accurately Reported ✅
- **Claimed**: 15.6% improvement
- **Computed**: (3.64 - 3.07) / 3.64 × 100 = 15.65% ✅

---

## Normality Assessment

**Test Results** (on error differences):
```
Shapiro-Wilk:        p = 0.000905  ❌ Reject normality
Kolmogorov-Smirnov:  p = 0.097     ✅ Accept normality
D'Agostino-Pearson:  p = 0.005     ❌ Reject normality

Skewness:  -0.38  (✅ approximately symmetric)
Kurtosis:   1.63  (⚠️ moderately heavy tails)
Outliers:   7/128 (5.5%)
```

**Verdict**: Normality is questionable (2/3 tests reject), BUT:
- n=128 is large enough for Central Limit Theorem
- Non-parametric tests (Wilcoxon, permutation) confirm significance
- Result is **robust to normality violations**

---

## Recommendation Details

### Recommendation: ✅ **ACCEPT WITH MINOR REVISIONS**

**Required Revisions** (estimated: 2-3 hours):

1. **Add Normality Discussion** (30 min):
   - Include Shapiro-Wilk test results
   - Add Q-Q plot of error differences
   - Justify t-test use OR emphasize non-parametric tests

2. **Discuss Practical Significance** (45 min):
   - Contextualize 0.57 WQI point improvement
   - Explain real-world implications for water quality decisions
   - Consider minimum meaningful difference for WQI

3. **Address Multiple Comparisons** (30 min):
   - Acknowledge 4 tests performed
   - Justify lack of Bonferroni correction (convergent evidence)
   - OR show results remain significant with correction

**Optional Enhancements**:
- Report Cohen's d or standardized effect size
- Perform sensitivity analysis (5-fold vs. 10-fold CV)
- Analyze improvement across different WQI ranges

---

## Detailed Findings

### Cross-Validation Implementation

**Fold Structure**:
```
Folds 1-8:  Train=115, Validation=13
Folds 9-10: Train=116, Validation=12
Total: 128 samples, 10 folds
```

**Consistency**:
- US model CV MAE range: [2.29, 4.23], std=0.64 ✅ Low variance
- Calibrated CV MAE range: [2.80, 5.34], std=0.75 ✅ Low variance

**Verdict**: Stable, generalizable performance estimates

---

### Statistical Power Analysis

**Sample Size**: n=128 paired differences

**Power for Paired t-test**:
- Effect size: d = 0.5637 / 2.52 ≈ 0.22 (small-to-medium)
- α = 0.05 (two-tailed)
- Observed p = 0.012 (significant)

**Power for Wilcoxon**:
- More robust to outliers and non-normality
- Achieved p = 0.003 (highly significant)

**Verdict**: Adequate sample size for detecting observed effect

---

## Comparison to Literature Standards

**Model Comparison Best Practices** (Demšar, 2006; Dietterich, 1998):

| Best Practice | Implementation | Status |
|---------------|----------------|--------|
| Use cross-validation | 10-fold CV | ✅ |
| Same train/test splits | random_state=42 | ✅ |
| Multiple statistical tests | 4 tests | ✅ |
| Non-parametric alternatives | Wilcoxon, permutation | ✅ |
| Report effect size | MAE, RMSE, R² | ✅ |
| Correct for multiple comparisons | Not applied | ⚠️ |

**Overall Alignment**: ✅ Excellent adherence to standards

---

## Reproducibility Assessment

**Code Quality**: ✅ Excellent
- Clear variable names
- Well-documented functions
- Modular design

**Data Availability**: ✅ Excellent
- Results saved to JSON
- Intermediate data accessible
- Random seeds documented

**Documentation**: ✅ Excellent
- Methodology clearly explained
- Original flaw acknowledged
- Correction process documented

**Independent Verification**: ✅ Successful
- Re-ran comparison script: Results matched
- Re-implemented calibrated CV: MAE=3.64 (expected: 3.64)

**Reproducibility Score**: 5/5 ✅

---

## Ethical Considerations

**Scientific Integrity**: ✅ Exemplary

1. ✅ **Honest Error Acknowledgment**: Original 44% claim acknowledged as flawed
2. ✅ **Transparent Correction**: Methodology explained in detail
3. ✅ **Reduced Claim**: Accepted 15.6% despite being less impressive
4. ✅ **Inflation Factor Reported**: Disclosed 2.8× overstatement

**Verdict**: Authors demonstrate exceptional scientific integrity by publicly correcting their error rather than defending the flawed methodology.

---

## Risk Assessment

**Risk of Type I Error** (false positive):
- Without correction: p < 0.05 for all 4 tests
- With Bonferroni correction (α=0.0125):
  - Wilcoxon: p=0.003 ✅ Still significant
  - t-test: p=0.012 ✅ Still significant
  - Permutation: p=0.014 ⚠️ Borderline

**Risk of Type II Error** (false negative):
- Effect size (0.57 points) is real and consistent
- Multiple tests converge on same conclusion
- Low variance across CV folds

**Overall Risk**: ⚠️ LOW - Results are robust even under conservative corrections

---

## Conclusion

The authors have properly identified and corrected a critical methodological flaw in their model comparison. The revised analysis employs rigorous 10-fold cross-validation for both models, eliminating the in-sample bias that inflated the original improvement claim by 2.8×.

**Key Findings**:
1. ✅ Methodology is sound and properly implemented
2. ✅ All statistical tests achieve significance (p<0.05)
3. ✅ Improvement (15.6%) is real but modest
4. ⚠️ Normality assumption is questionable but doesn't invalidate conclusions
5. ⚠️ Practical significance should be discussed

**Final Recommendation**: **ACCEPT WITH MINOR REVISIONS**

The correction is methodologically sound, statistically rigorous, and demonstrates exemplary scientific integrity. The minor concerns (normality, practical significance, multiple comparisons) should be addressed through additional discussion, but they do not undermine the core finding.

---

**Reviewer Confidence**: HIGH

**Estimated Revision Time**: 2-3 hours

**Re-review Required**: NO (minor revisions can be verified by editor)

---

## References

- Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research*, 7, 1-30.
- Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation*, 10(7), 1895-1923.
- Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error. *Machine Learning*, 52(3), 239-281.
