# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Critical Rules - NO EXCEPTIONS

### Never Guess or Assume
- If you don't know something, **ask** or **search** for the answer
- Don't assume file locations, API formats, or implementation details
- Don't proceed with uncertain information - verify first

### No Mocks or Fake Data
- Use REAL DATA only
- No placeholder data, mock responses, or synthetic test data
- If an API call fails, **fix it** - don't fake the response

### No Shortcuts
- No incomplete implementations marked as "TODO"
- No skipping error handling or edge cases
- Test everything before claiming completion

### No Fallbacks or Graceful Degradation
- If something fails, **stop and fix it**
- Don't return partial results with a disclaimer
- Don't paper over errors with default values

## Commands

```bash
# Install dependencies (requires Python 3.11+)
poetry install

# Run Streamlit app
poetry run streamlit run streamlit_app/app.py

# Train models
poetry run python train_models.py                  # Full feature set
poetry run python train_models.py --core-params-only  # Core params only (recommended for deployment)

# Code quality
poetry run black .
poetry run flake8 src
poetry run mypy src

# Run tests
poetry run pytest                        # All tests
poetry run pytest path/to/test_file.py   # Single file
poetry run pytest -k "test_name"         # Single test by name
```

## Commit Messages
Use prefixes: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`. Keep imperative and short.

## Architecture

### Core Data Flow
```
ZIP Code → Geocode (pgeocode) → API Query (USGS/WQP) →
Daily Aggregation → WQI Calculation → ML Prediction →
SHAP Explanation → Streamlit Visualization
```

### Data Collection Strategy
The system uses a progressive fallback strategy to maximize data coverage:
- **Primary source**: Water Quality Portal (WQP) - aggregates USGS, EPA, state, and tribal agencies
- **Fallback source**: USGS NWIS (converted to WQP format for compatibility)
- **Parallel fetching**: ThreadPoolExecutor queries both APIs simultaneously; WQP results are preferred, USGS acts as backup
- **Radius expansion** (if no data found): Expands search from user-specified radius to 10 km → 25 km → 50 km
- **Date range extension** (if still no data): Falls back from 1 year to 4 years of history
- **Site filtering**: Surface water only by default (Stream, River, Lake, Reservoir); groundwater optional; marine/estuarine excluded by conductance threshold (>3000 µS/cm)

### Key Modules
- `src/data_collection/` - USGS NWIS and Water Quality Portal API clients with fallback logic
- `src/utils/wqi_calculator.py` - NSF-WQI implementation using 6 parameters (pH, DO, temp, turbidity, nitrate, conductance)
- `src/utils/validation_metrics.py` - Bootstrap CIs, ECE, Brier score, reliability diagrams for model validation
- `src/models/` - Random Forest classifier (SAFE/UNSAFE) and regressor (WQI 0-100)
- `src/services/search_strategies.py` - Progressive fallback: expands search radius (5→10→25→50 km) and date range when no data found
- `src/geolocation/` - ZIP-to-lat/long via pgeocode
- `streamlit_app/app.py` - UI only; business logic belongs in `src/services/` or `src/utils/`

### Key Dependencies
- **pgeocode** - ZIP code geocoding (offline postal code database)
- **shap** - Model explanations (SHAP values for per-sample feature contributions)
- **scipy** - Statistical tests (Shapiro-Wilk, Spearman) for model validation
- **requests** - API clients (USGS NWIS, Water Quality Portal)

### Critical: Nitrate Unit Conversion
The codebase handles a 4.43× unit mismatch between data sources:
- **Kaggle dataset**: mg{NO3}/L (full nitrate molecule)
- **EPA/USGS standard**: mg/L as N (nitrogen content only)
- **Conversion factor**: `NITRATE_NO3_TO_N = 0.2258` in WQP client

### Model Limitations
The ML models **cannot detect**: lead, heavy metals, bacteria, pesticides, or PFAS. The NSF-WQI excludes these parameters, resulting in 100% false negatives on lead-contaminated water (documented in `docs/ENVIRONMENTAL_JUSTICE_ANALYSIS.md`).

**Feature Mismatch - US Predictions**: Production predictions use only 18 CORE features from `src/preprocessing/us_data_features.py`. These exclude:
- **Turbidity**: Not available from USGS/WQP APIs (0% coverage in US data)
- **Missing-value indicators**: Excluded to prevent spurious correlations
- **European geographic features**: GDP, waste management, country (imputed as defaults for US, degrading performance)

This training/deployment feature mismatch explains why US predictions have lower R² (~0.4-0.5) compared to European test set (0.97).

**Model Training Details**:
- **Classifier**: `class_weight='balanced'` is hardcoded (not tuned in grid search) because training data is 98.8% SAFE
- **Validation**: 5-fold stratified cross-validation via GridSearchCV
- **Regressor**: Includes 12-month forecast capability

### Feature Engineering
Training data pipeline (`src/preprocessing/feature_engineering.py`) builds two feature sets:
- **Full features** (59 total): Includes parameter features, temporal context, derived metrics, European geographic features (GDP, waste management, country), and interaction terms
- **Core features** (18 total): Only core water quality parameters + temporal context + basic derived metrics (no geographic features, no turbidity, no missing-value indicators)
  - Raw parameters (5): pH, dissolved oxygen, temperature, nitrate, conductance
  - Temporal (6): year, years_since_1991, decade, is_1990s/2000s/2010s
  - Derived (5): ph_deviation_from_7, do_temp_ratio, conductance_low/medium/high
  - Interactions (2): pollution_stress, temp_stress

**Critical Note**: Production uses 18 CORE features only. Training uses full 59 features. US data collection cannot provide turbidity (0% availability from USGS/WQP), missing-value indicators, or European geographic features.

### Data Quality Pipeline
The training data pipeline includes several quality controls:
- **Physical bounds**: `VALID_RANGES` in `feature_engineering.py:93-99` filters impossible values
- **Statistical outliers**: IQR detection (3× multiplier) runs in dry-run mode by default (logs candidates without removing) because it flagged valid DO values 5-9 mg/L due to skewed distribution
- **Artifact features**: Missing indicators excluded if >95% missing (prevents spurious correlations like `turbidity_missing`)
- **Imputation**: Median imputation (robust to outliers with 60-75% missingness; KNN was tested but degraded performance)
- **Class weights**: Always `class_weight='balanced'` in classifier (hardcoded, not optional in grid search) - necessary because training data is 98.8% SAFE
- **Quality report**: Generated at `data/processed/data_quality_report.json` during training

## Data Files
- `data/raw/waterPollution.csv` - Kaggle training data (gitignored, must download)
- `data/models/` - Trained models (binaries gitignored, metadata JSON tracked)

## Environment Variables
- `WQP_TIMEOUT` - Water Quality Portal API request timeout (seconds; default: 30)
  - Increase if API is slow or you get timeout errors
  - Example: `export WQP_TIMEOUT=60`
- `USGS_TIMEOUT` - USGS NWIS API request timeout (seconds; default: 30)
  - Increase if API is slow or you get timeout errors
  - Example: `export USGS_TIMEOUT=60`
- `WQI_SKIP_PATH_VALIDATION` - Bypass model path security checks (testing/debugging only)
  - Set to any value to skip validation: `export WQI_SKIP_PATH_VALIDATION=1`
  - Never use in production

## Plan Tracking
- `.claude/plan.md` - Single source of truth for implementation plans
- Mark completed: `[x]`, in-progress: `← IN PROGRESS`
- Delete plans when complete or replacing

## MCP Servers

### Context7
Use `resolve-library-id` then `get-library-docs` for current documentation on any library/framework.

### Chrome DevTools
For browser performance traces, network inspection, console logs, screenshots.
