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

### Key Modules
- `src/data_collection/` - USGS NWIS and Water Quality Portal API clients with fallback logic
- `src/utils/wqi_calculator.py` - NSF-WQI implementation using 6 parameters (pH, DO, temp, turbidity, nitrate, conductance)
- `src/models/` - Random Forest classifier (SAFE/UNSAFE) and regressor (WQI 0-100)
- `src/services/search_strategies.py` - Progressive fallback: expands search radius (5→10→25→50 km) and date range when no data found
- `src/geolocation/` - ZIP-to-lat/long via pgeocode
- `streamlit_app/app.py` - UI only; business logic belongs in `src/services/` or `src/utils/`

### Key Dependencies
- **pgeocode** - ZIP code geocoding (offline postal code database)
- **shap** - Model explanations (SHAP values for per-sample feature contributions)
- **requests** - API clients (USGS NWIS, Water Quality Portal)

### Critical: Nitrate Unit Conversion
The codebase handles a 4.43× unit mismatch between data sources:
- **Kaggle dataset**: mg{NO3}/L (full nitrate molecule)
- **EPA/USGS standard**: mg/L as N (nitrogen content only)
- **Conversion factor**: `NITRATE_NO3_TO_N = 0.2258` in WQP client

### Model Limitations
The ML models **cannot detect**: lead, heavy metals, bacteria, pesticides, or PFAS. The NSF-WQI excludes these parameters, resulting in 100% false negatives on lead-contaminated water (documented in `docs/ENVIRONMENTAL_JUSTICE_ANALYSIS.md`).

## Data Files
- `data/raw/waterPollution.csv` - Kaggle training data (gitignored, must download)
- `data/models/` - Trained models (binaries gitignored, metadata JSON tracked)
- Environment variables:
  - `WQP_TIMEOUT`, `USGS_TIMEOUT` - API request timeouts
  - `WQI_SKIP_PATH_VALIDATION` - Bypass model path security checks (testing only)

## Plan Tracking
- `.claude/plan.md` - Single source of truth for implementation plans
- Mark completed: `[x]`, in-progress: `← IN PROGRESS`
- Delete plans when complete or replacing

## MCP Servers

### Context7
Use `resolve-library-id` then `get-library-docs` for current documentation on any library/framework.

### Chrome DevTools
For browser performance traces, network inspection, console logs, screenshots.
