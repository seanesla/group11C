# Environmental & Water Quality Prediction

**Group 11C**: Joseann Boneo, Sean Esla, Zizwe Mtonga, Lademi Aromolaran

A machine learning system that provides Water Quality Index (WQI) scores and trend predictions based on US ZIP codes. The system fetches real water quality data from the Water Quality Portal and calculates comprehensive quality assessments.

## Features

- **ZIP Code Lookup**: Search for water quality data by US ZIP code
- **Real-Time Data**: Fetches live water quality measurements from EPA/USGS Water Quality Portal
- **WQI Calculation**: Comprehensive Water Quality Index based on 6 key parameters:
  - pH
  - Dissolved Oxygen (DO)
  - Temperature
  - Turbidity
  - Nitrate
  - Specific Conductance
- **Interactive Visualizations**: Time series charts and parameter comparisons using Plotly
- **Safety Assessment**: Indicates whether water is safe for drinking (WQI ≥ 70)
- **Data Export**: Download raw data as CSV

## Project Status

**50% Complete** - MVP functional with end-to-end system working with real data

**Completed:**
- ✅ Water Quality Portal API client
- ✅ ZIP code to geolocation mapping
- ✅ WQI calculation engine
- ✅ Streamlit web application with visualizations

**In Progress:**
- 🔨 ML models for trend prediction
- 🔨 Unit and integration tests

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

- **20001** - Washington, DC (4,287 measurements, WQI: 91.2 Excellent)
- **10001** - New York City, NY (3,504 measurements)
- **99501** - Anchorage, AK (21 measurements)

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