# Water Quality Prediction - Project Status

## Current State (2025-12-02)

**Status:** Feature-complete, production-ready

### Core Functionality
- WQI Calculator: NSF-WQI implementation with 6 parameters
- ML Models: Random Forest classifier (SAFE/UNSAFE) and regressor (WQI 0-100)
- Data Collection: USGS NWIS and WQP API clients with fallback logic
- Streamlit App: Full UI with ZIP search, WQI visualization, SHAP explanations

### Known Limitations
- ML models cannot detect: lead, heavy metals, bacteria, pesticides, PFAS
- See `docs/ENVIRONMENTAL_JUSTICE_ANALYSIS.md` for detailed limitations

### Key Files
- `streamlit_app/app.py` - Main application
- `src/utils/wqi_calculator.py` - WQI calculation engine
- `src/models/classifier.py`, `regressor.py` - ML models
- `src/data_collection/wqp_client.py`, `usgs_client.py` - API clients
