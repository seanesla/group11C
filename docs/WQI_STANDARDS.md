# Water Quality Index Standards Documentation

## Purpose
This document provides authoritative references for Water Quality Index (WQI) calculation, including official parameter weights, scoring curves, and regulatory standards.

## NSF Water Quality Index (NSF-WQI)

### Official Formula and Methodology

**Source:** National Sanitation Foundation (Boulder County Watershed Information Network)
**URL:** https://bcn.boulder.co.us/basin/watershed/wqi_nsf.html

**Formula:**
```
WQI = Σ(Q_i × W_i)
```
Where:
- Q_i = subindex value for parameter i (0-100)
- W_i = weight for parameter i
- Sum of all weights = 1.0

### Official NSF-WQI Parameters and Weights

The NSF-WQI uses **9 parameters**:

| Parameter | Weight | Percentage |
|-----------|--------|------------|
| Dissolved Oxygen (DO) | 0.17 | 17% |
| Fecal Coliform | 0.16 | 16% |
| pH | 0.11 | 11% |
| Biochemical Oxygen Demand (BOD) | 0.11 | 11% |
| Temperature Change (ΔT) | 0.10 | 10% |
| Total Phosphate | 0.10 | 10% |
| Nitrate | 0.10 | 10% |
| Turbidity | 0.08 | 8% |
| Total Solids | 0.07 | 7% |
| **TOTAL** | **1.00** | **100%** |

### NSF-WQI Classification Scale

| WQI Range | Classification | Description |
|-----------|----------------|-------------|
| 90-100 | Excellent | Pristine water quality |
| 70-90 | Good | Safe for most uses |
| 50-70 | Medium | Acceptable but monitor |
| 25-50 | Bad | Treatment recommended |
| 0-25 | Very Bad | Significant contamination |

### Partial Parameter Calculation

When fewer than 9 parameters are available:
1. Calculate Q_i for available parameters
2. Sum the products (Q_i × W_i) for available parameters
3. Divide by the sum of corresponding weights

Formula: `WQI = Σ(Q_i × W_i) / Σ(W_i)`

---

## EPA Water Quality Standards

### Primary Drinking Water Standards (MCLs)

**Source:** EPA National Primary Drinking Water Regulations
**URL:** https://www.epa.gov/ground-water-and-drinking-water/national-primary-drinking-water-regulations

| Contaminant | MCL | Units | Health Concern |
|-------------|-----|-------|----------------|
| Nitrate (as N) | 10 | mg/L | Blue-baby syndrome in infants <6 months |
| Nitrite (as N) | 1 | mg/L | Similar to nitrate, infant health risk |
| Turbidity | 1 | NTU | Treatment technique; indicates filtration effectiveness |

### Secondary Drinking Water Standards (SMCLs)

**Non-mandatory aesthetic guidelines:**

| Parameter | SMCL | Units | Concern |
|-----------|------|-------|---------|
| pH | 6.5 - 8.5 | units | Corrosion, taste |
| Conductivity | No standard | µS/cm | Aesthetic indicator |
| Temperature | No standard | °C | Affects aquatic life, not human health |

**Note:** Dissolved Oxygen (DO) is **not regulated** for drinking water. DO is a surface water quality parameter for aquatic ecosystem health.

---

## WHO Water Quality Guidelines

**Source:** WHO Guidelines for Drinking-water Quality (Fourth Edition)

### Key Parameters

| Parameter | WHO Guideline | Notes |
|-----------|---------------|-------|
| pH | No health-based guideline | Recommended operational range: 6.5-9.5 |
| Temperature | <25°C preferred | Affects taste and chemical reactions |
| Nitrate (as NO3) | 50 mg/L | Equivalent to 11.3 mg/L as N |
| Turbidity | <5 NTU ideal | Higher values indicate treatment problems |

---

## Current Implementation Analysis

### Our WQI Calculator Parameters (6 parameters)

The project uses **6 of the 9 NSF-WQI parameters**:

| Parameter | Used? | Data Source |
|-----------|-------|-------------|
| pH | ✅ Yes | Water Quality Portal |
| Dissolved Oxygen | ✅ Yes | Water Quality Portal |
| Temperature | ✅ Yes | Water Quality Portal |
| Turbidity | ✅ Yes | Water Quality Portal |
| Nitrate | ✅ Yes | Water Quality Portal |
| Specific Conductance | ✅ Yes* | Water Quality Portal |
| Fecal Coliform | ❌ No | Not readily available |
| BOD | ❌ No | Requires lab analysis |
| Total Phosphate | ❌ No | Not prioritized |
| Total Solids | ❌ No | Not prioritized |

*Note: Using Specific Conductance instead of Total Solids as a water quality indicator.

### Weight Discrepancy Identified

**CRITICAL BUG FOUND:**

**Documented Weights (PARAMETER_WEIGHTS dict, lines 39-49):**
- Matches NSF-WQI official weights ✅

**Actual Weights Used (calculate_wqi method, lines 306-326):**
| Parameter | Actual Weight | NSF Weight | Discrepancy |
|-----------|---------------|------------|-------------|
| pH | 0.20 | 0.11 | **+81% increase** |
| Dissolved Oxygen | 0.25 | 0.17 | **+47% increase** |
| Temperature | 0.15 | 0.10 | **+50% increase** |
| Turbidity | 0.15 | 0.08 | **+88% increase** |
| Nitrate | 0.15 | 0.10 | **+50% increase** |
| Conductance | 0.10 | N/A | New parameter |
| **TOTAL** | **1.00** | **0.66** | - |

### Analysis

1. **The PARAMETER_WEIGHTS dictionary is defined but never used** - Classic dead code bug
2. **Hardcoded weights don't follow NSF-WQI methodology**
3. **Weights are arbitrarily chosen, not scientifically derived**
4. **Using 6 parameters but not redistributing NSF weights proportionally**

### Correct Weight Redistribution

If we're using 6 parameters, we should proportionally redistribute the NSF-WQI weights:

**NSF weights for our 6 parameters (sum = 0.66):**
- DO: 0.17
- pH: 0.11
- Temperature: 0.10
- Turbidity: 0.08
- Nitrate: 0.10
- (Conductance substitutes Total Solids: 0.07)

**Option 1: Proportional Redistribution (Normalize to 1.0)**
```python
CORRECTED_WEIGHTS = {
    'dissolved_oxygen': 0.17 / 0.63 = 0.270,  # ~27%
    'ph': 0.11 / 0.63 = 0.175,                 # ~17%
    'temperature': 0.10 / 0.63 = 0.159,        # ~16%
    'turbidity': 0.08 / 0.63 = 0.127,          # ~13%
    'nitrate': 0.10 / 0.63 = 0.159,            # ~16%
    'conductance': 0.07 / 0.63 = 0.111         # ~11%
}
# Sum = 1.001 (rounding)
```

Note: 0.63 = sum of NSF weights for our 6 parameters (excluding fecal coliform, BOD, phosphate)

**Option 2: Use NSF Weights Directly (Normalize Dynamically)**
```python
CORRECTED_WEIGHTS = {
    'dissolved_oxygen': 0.17,
    'ph': 0.11,
    'temperature': 0.10,
    'turbidity': 0.08,
    'nitrate': 0.10,
    'conductance': 0.07  # Same as Total Solids
}
# Calculate: WQI = Σ(Q_i × W_i) / Σ(W_i)
# This maintains NSF-WQI relative importance
```

---

## Recommendation

**Use Option 2: NSF weights with dynamic normalization**

**Rationale:**
1. Maintains relative importance of parameters as determined by NSF experts
2. Scientifically defensible - references authoritative standard
3. Handles missing parameters gracefully
4. Allows future addition of more parameters without recalculating weights

**Implementation:**
```python
# At class level
PARAMETER_WEIGHTS = {
    'dissolved_oxygen': 0.17,
    'ph': 0.11,
    'temperature': 0.10,
    'turbidity': 0.08,
    'nitrate': 0.10,
    'conductance': 0.07  # Substituting for Total Solids
}

# In calculate_wqi method
def calculate_wqi(self, ...):
    scores = {}
    weights_used = {}

    if ph is not None and not pd.isna(ph):
        scores['ph'] = self.calculate_ph_score(ph)
        weights_used['ph'] = self.PARAMETER_WEIGHTS['ph']  # 0.11

    # ... repeat for other parameters ...

    # Normalize weights
    total_weight = sum(weights_used.values())
    normalized_weights = {k: v / total_weight for k, v in weights_used.values()}

    # Calculate WQI
    wqi = sum(scores[param] * normalized_weights[param] for param in scores)
```

---

## References

### Primary Sources

1. **National Sanitation Foundation WQI**
   - Boulder County Watershed Information: https://bcn.boulder.co.us/basin/watershed/wqi_nsf.html
   - Brown, R.M., et al. (1970). "A water quality index—do we dare?" Water Sewage Works, 117(10), 339-343.
   - Comprehensive review: https://pmc.ncbi.nlm.nih.gov/articles/PMC10006569/

2. **EPA Standards**
   - National Primary Drinking Water Regulations: https://www.epa.gov/ground-water-and-drinking-water/national-primary-drinking-water-regulations
   - Secondary Standards: https://www.epa.gov/sdwa/secondary-drinking-water-standards-guidance-nuisance-chemicals
   - Nitrate factsheet: https://archive.cdc.gov/www_atsdr_cdc_gov/csem/nitrate-nitrite/standards.html

3. **WHO Guidelines**
   - Guidelines for Drinking-water Quality (Fourth Edition): https://iris.who.int/bitstream/handle/10665/44584/9789241548151_eng.pdf

### Scientific Literature

- ResearchGate NSF-WQI Table: https://www.researchgate.net/figure/NSF-WQI-Analytes-and-Weights_tbl1_237259303
- Comprehensive WQI review (2023): https://pmc.ncbi.nlm.nih.gov/articles/PMC10006569/
- Critical review of NSF-WQI application: https://www.sciencedirect.com/science/article/abs/pii/S0269749118304627

---

## Decision Log

**Date:** 2025-11-03
**Decision:** Fix weight discrepancy by using NSF-WQI weights with dynamic normalization
**Rationale:** Maintains scientific validity while handling partial parameter availability
**Impact:** WQI calculations will change; need to retrain ML models if they depend on current WQI values

**Action Items:**
1. ✅ Document NSF-WQI standards (this file)
2. ⏳ Fix PARAMETER_WEIGHTS usage in calculate_wqi method
3. ⏳ Update all existing tests to use corrected weights
4. ⏳ Create validation tests against NSF-WQI methodology
5. ⏳ Verify ML model alignment with corrected WQI values
6. ⏳ Update README and documentation

---

**Last Updated:** 2025-11-03
**Status:** Weight discrepancy identified and documented; fix pending user approval
