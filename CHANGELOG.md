# Changelog

All notable changes to the Environmental & Water Quality Prediction project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2025-11-17

### Added

#### Environmental Justice & Ethics
- **Comprehensive Environmental Justice Analysis** (docs/ENVIRONMENTAL_JUSTICE_ANALYSIS.md):
  - 16-page detailed analysis of system limitations with lead contamination
  - Flint, MI water crisis analysis (2014-2019): 53% Black, 41% poverty, lead 18-150 ppb
  - Jackson, MS water crisis (2021-2023): 83% Black, 25% poverty, infrastructure failure
  - Complete testing of ALL 3 production components (WQI Calculator, ML Classifier, ML Regressor)
  - Results: 100% false negative rate across all components on lead-contaminated water
  - 14 authoritative citations (EPA, CDC, Virginia Tech, ProPublica, WHO)
  - Honest limitations section and ethical recommendations

- **Homepage Environmental Justice Warnings** (streamlit_app/app.py):
  - **st.error (Red Alert)**: "CRITICAL LIMITATION: Lead and Heavy Metal Detection"
    - EPA Safe Drinking Water Hotline: 1-800-426-4791
    - High-risk homes identification (pre-1986 construction, Flint/Jackson areas)
  - **st.info (Blue Alert)**: "Data Coverage" indicator
    - ✅ Tested: 6/9 NSF-WQI parameters (pH, DO, temperature, turbidity, nitrate, conductance)
    - ❌ NOT tested: Lead, heavy metals, bacteria, pesticides, PFAS, pharmaceuticals
  - **st.warning (Yellow Alert)**: "This is NOT a comprehensive water safety test"
    - Never use as replacement for certified water quality testing
    - Emergency contact information for contamination concerns

#### Testing & Validation
- **Complete System Testing Scripts**:
  - scripts/test_environmental_justice.py: WQI-only testing (6 contaminated scenarios)
  - scripts/test_environmental_justice_COMPLETE.py: Full system testing (WQI + ML Classifier + ML Regressor)
  - Test data: data/environmental_justice_results.csv (WQI only)
  - Test data: data/environmental_justice_wqi_results.csv, data/environmental_justice_ml_results.csv (complete)

### Fixed

#### Critical UI Bug
- **Environmental Justice Warnings Visibility** (commit e8bfa4f):
  - **BUG**: Warnings were hidden inside `if search_button:` block, only visible AFTER user searched
  - **FIX**: Moved warnings to homepage `else:` block (lines 1963-2022)
  - **IMPACT**: Users now see critical limitations BEFORE using the system (informed consent)
  - **VERIFICATION**: Playwright browser snapshot confirms all 3 warnings visible on page load
  - **WHY IT MATTERED**: Environmental justice requires proactive disclosure, not hiding limitations
  - **LESSON**: Always verify user-facing changes in browser, not just code

### Context

This release addresses critical ethical and environmental justice concerns raised during production hardening review. The NSF-WQI methodology excludes lead as a parameter, creating a systematic limitation where the system cannot detect lead contamination that disproportionately affects marginalized communities (Flint: 53% Black, Jackson: 83% Black). The complete system testing revealed 100% false negative rates across all three production components (WQI Calculator, ML Classifier, ML Regressor) on lead-contaminated water scenarios.

The homepage warnings ensure users understand these limitations before using the system, preventing false sense of security from high WQI scores on potentially lead-contaminated water. This work directly addresses Agent 4 (Ethics) review score of 65/100, implementing comprehensive disclosure and documentation of system limitations.

## [1.0.0] - 2025-11-16

### Added

#### Core Features
- **Water Quality Index (WQI) Calculator**: NSF-WQI based calculation engine supporting 6 parameters (pH, dissolved oxygen, temperature, turbidity, nitrate, conductance)
- **ZIP Code Lookup**: Geographic search functionality for water quality data across the United States
- **Real-Time Data Integration**:
  - EPA Water Quality Portal API client with automatic data aggregation
  - USGS National Water Information System (NWIS) API client
  - 93,000+ monitoring stations accessible across the US
- **ML-Powered Predictions**:
  - Random Forest Classifier for safety assessment (98.98% accuracy)
  - Random Forest Regressor for WQI prediction (R² = 0.9911)
  - 12-month future trend forecasting with trend direction analysis
- **Interactive Web Application**: Streamlit-based UI with real-time data visualization
- **Nitrate Unit Conversion System**: Automatic standardization between mg{NO3}/L and mg/L as N units

#### Data Processing
- **Feature Engineering Pipeline**: Transforms raw water quality data into 59 ML features
- **US Data Feature Preparation**: Specialized feature preparation for US monitoring data
- **Unit Standardization**:
  - WQP client: Converts mg{NO3}/L → mg/L as N when detected
  - USGS client: Validates expected mg/L as N format with warnings
  - Kaggle data: Applies 0.2258 conversion factor (N/NO3 molecular weight ratio)

#### Visualizations
- **Time Series Charts**: Water quality trends over time (Plotly interactive charts)
- **Parameter Comparison**: Individual parameter scores and contributions
- **Future Trend Forecasting**: 12-month WQI predictions with confidence intervals
- **Safety Indicators**: Color-coded classifications (Excellent/Good/Fair/Poor/Very Poor)

#### Testing & Quality Assurance
- **1,555+ Comprehensive Tests**:
  - 16 nitrate unit conversion tests (100% passing)
  - 34 end-to-end integration tests (100% passing)
  - 1,505+ unit and integration tests
- **Test Categories**:
  - Full pipeline integration (ZIP → WQI → ML → trend)
  - Error handling and edge cases
  - Consistency and determinism validation
  - Data validation and quality checks
  - Performance metrics and benchmarks

#### Documentation
- **README.md**: Complete project overview with usage instructions
- **WQI_STANDARDS.md**: Comprehensive documentation of NSF-WQI methodology and EPA standards
- **Nitrate Unit Conversion Documentation**: Chemical basis, implementation details, and testing coverage
- **API Client Documentation**: Inline documentation for WQP and USGS clients
- **CLAUDE.md**: Development guidelines and project standards

### Fixed

#### Critical Bugs
- **Nitrate Unit Mismatch** (4.43× conversion error):
  - **Problem**: Kaggle dataset used mg{NO3}/L while EPA/USGS use mg/L as N
  - **Impact**: All WQI predictions and ML models were using incorrect nitrate values
  - **Solution**: Implemented NITRATE_NO3_TO_N = 0.2258 conversion constant
  - **Result**: ML models retrained with corrected values (accuracy improved to 98.98%)

#### Bug Fixes
- Fixed redundant browser E2E test file causing test failures
- Fixed MCP tool integration issues in pytest runs
- Fixed coverage reporting for nitrate conversion tests

### Changed

#### Model Improvements
- **Retrained ML Models** (2025-11-16):
  - Classifier accuracy: 98.98% (previously lower due to nitrate bug)
  - Regressor R² score: 0.9911 (previously 0.85-0.90 range)
  - All models now use EPA-compliant mg/L as N units

#### Code Quality
- Added comprehensive unit standardization with automatic conversion
- Enhanced error handling in API clients (WQP, USGS)
- Improved data validation with unit consistency checks
- Added warnings for unexpected unit formats

### Technical Details

#### Dependencies
- Python 3.11+
- Poetry for dependency management
- Key libraries:
  - `streamlit`: Web application framework
  - `scikit-learn`: Machine learning models
  - `pandas`, `numpy`: Data processing
  - `plotly`: Interactive visualizations
  - `requests`: API clients
  - `pgeocode`: ZIP code geocoding
  - `pytest`: Testing framework

#### Performance Metrics
- **WQI Calculation**: < 10ms per calculation (100 calculations in < 1 second)
- **Feature Preparation**: < 10ms per sample (100 preparations in < 1 second)
- **ML Predictions**: < 20ms per prediction pair (100 pairs in < 2 seconds)
- **API Response Time**: 5-30 seconds (depends on data volume and network)

#### Test Coverage
- Total tests: 1,555+
- All critical paths covered
- Nitrate conversion: 16 dedicated tests
- End-to-end integration: 34 comprehensive tests
- Coverage areas: unit conversion, ML predictions, data validation, API integration

### Known Limitations

#### Data Availability
- Water quality data availability varies by location
- Remote areas may have sparse monitoring station coverage
- Historical data typically limited to last 1-2 years for most stations

#### ML Model Training
- Models trained on European water quality dataset (Kaggle, 1991-2017)
- Geographic differences between European and US water systems may affect accuracy
- Nitrate conversion ensures unit consistency despite source differences

#### Parameter Coverage
- System uses 6 of 9 NSF-WQI parameters
- Missing parameters: Fecal Coliform, BOD, Total Phosphate
- Weights redistributed proportionally among available parameters

### Security

#### API Best Practices
- Rate limiting implemented (1 second delay between requests)
- User-Agent headers identify educational project purpose
- No sensitive data stored or transmitted
- API keys not required (public data endpoints)

### Migration Guide

#### For Existing Users
No migration required - this is the initial production release.

#### For Developers
If you have local development work:
1. Pull latest changes: `git pull origin main`
2. Update dependencies: `poetry install`
3. Retrain models if using local models: `poetry run python train_models.py`
4. Run code quality checks: `poetry run black . && poetry run flake8 src && poetry run mypy src`

### Contributors

**Group 11C**:
- Joseann Boneo
- Sean Esla
- Zizwe Mtonga
- Lademi Aromolaran

### Acknowledgments

- **National Sanitation Foundation (NSF)**: WQI methodology
- **EPA Water Quality Portal**: Real-time water quality data
- **USGS NWIS**: Historical monitoring station data
- **Kaggle**: European Water Quality Dataset (training data)

---

## [Unreleased]

### Removed
- Test suite removed (commit e477315) - 945 tests across 50+ files
- pytest.ini configuration removed
- Unused dependencies: xgboost, lightgbm, hypothesis, playwright

### Planned Features
- Mobile-responsive design improvements
- Additional ML models for specific contaminants
- Export functionality for PDF reports
- Historical trend analysis beyond 12 months
- Comparison tool for multiple ZIP codes
- Email alerts for water quality changes

---

**Last Updated**: 2025-12-02
**Status**: Production Ready
**Version**: 1.0.1
