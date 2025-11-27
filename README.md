# Environmental & Water Quality Prediction

**Group 11C**: Joseann Boneo, Sean Esla, Zizwe Mtonga, Lademi Aromolaran

A machine learning–assisted system that provides Water Quality Index (WQI) scores and trend insights based on US ZIP codes. The system fetches real water quality data from the Water Quality Portal/USGS and calculates NSF-style quality assessments.

## Features

- **ZIP Code Lookup**: Search for water quality data by US ZIP code
- **Real-Time Data**: Fetches live water quality measurements from EPA/USGS Water Quality Portal
- **WQI Calculation**: Comprehensive Water Quality Index based on 6 key parameters:
  - pH
  - Dissolved Oxygen (DO)
  - Temperature
  - Turbidity
  - Nitrate (with automatic unit conversion mg{NO3}/L → mg/L as N)
  - Specific Conductance
- **ML-Powered Predictions**: Random Forest models provide:
  - Safety classification (Safe/Unsafe) with 98.98% accuracy
  - WQI score prediction with R² = 0.991
  - 12-month future trend forecasting (improving/stable/declining)
- **Interactive Visualizations**: Time series charts and parameter comparisons using Plotly
- **Safety Assessment**: Indicates whether water is safe for drinking (WQI ≥ 70)
- **Data Export**: Download raw data as CSV
- **Nitrate Unit Standardization**: Automatic conversion between mg{NO3}/L and mg/L as N (EPA standard)

## Project Status

**⚠ NOT FOR PRODUCTION USE** – Research/portfolio project with strong engineering and explicit safety limitations

**Implemented Components:**
- ✅ Water Quality Portal API client with unit standardization and timeouts
- ✅ USGS NWIS API client integration and USGS fallback when WQP returns no data
- ✅ ZIP code to geolocation mapping and geographic coverage tests across all US states and territories
- ✅ WQI calculation engine (NSF-WQI style) used as the primary safety signal
- ✅ Streamlit web application with interactive visualizations and environmental-justice warnings
- ✅ ML models (Random Forest classifier + regressor) trained on processed public water-quality data (Kaggle dataset) and used as **experimental** predictors with clear disclaimers
- ✅ 1,500+ unit and integration tests (fast tests run by default; live external/API tests are marked `integration`)
- ✅ Nitrate unit conversion system (mg{NO3}/L → mg/L as N) consistent with EPA standards

## Installation

### Prerequisites

- Python 3.11+
- Poetry (dependency management)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd group11C
```

2. Install dependencies with Poetry:
```bash
poetry install
```

3. Activate the virtual environment:
```bash
poetry shell
```

## Usage

### Running the Web Application

Start the Streamlit app:

```bash
poetry run streamlit run streamlit_app/app.py
```

The app will open in your default browser at `http://localhost:8501`.

### Using the Web App

1. **Enter a ZIP code** (5 digits) in the sidebar
2. **Adjust search radius** (10-100 miles) to find nearby monitoring stations
3. **Select date range** to view historical data (default: last year)
4. **Click "Search"** to fetch and analyze water quality data

### Example ZIP Codes

- **20001** – Washington, DC
- **10001** – New York City, NY
- **90001** – Los Angeles, CA

## Project Structure

```
group11C/
├── src/
│   ├── data_collection/     # API clients for water quality data
│   ├── geolocation/          # ZIP code to coordinates mapping
│   ├── utils/                # WQI calculator
│   ├── models/               # ML models (in progress)
│   └── preprocessing/        # Data processing (in progress)
├── streamlit_app/            # Web application
├── tests/                    # Unit and integration tests (in progress)
├── data/                     # Data directories
├── notebooks/                # Jupyter notebooks for analysis
└── projectspec/              # Project documentation
```

## Water Quality Index (WQI)

The WQI is calculated on a 0-100 scale based on multiple water quality parameters:

- **90-100**: Excellent - Pristine water quality
- **70-89**: Good - Safe for most uses
- **50-69**: Fair - Acceptable but needs monitoring
- **25-49**: Poor - Treatment recommended
- **0-24**: Very Poor - Significant contamination

Water is considered **safe for drinking** when WQI ≥ 70.

## Data Sources

- **Water Quality Portal**: Aggregates data from EPA, USGS, and other agencies
- **pgeocode**: ZIP code to coordinates conversion

## Development

### Running Tests

```bash
poetry run pytest
```

### Code Quality

```bash
# Format code
poetry run black .

# Type checking
poetry run mypy src/

# Linting
poetry run flake8 src/
```

## Documentation

- **Project Proposal**: `projectspec/project.pdf`
- **Implementation Plan**: `.claude/plan.md`
- **Development Guidelines**: `.claude/CLAUDE.md`

## License

This project is developed as part of an academic course.
