# Water Quality Prediction (Group 11C)

Environmental & Water Quality Prediction using machine learning and open water-quality data.

> AI4ALL Ignite capstone project (educational use only)

## Team

- Sean Esla
- Lademi Aromolaran
- Joseann Boneo

## Project Overview

This project builds an end-to-end pipeline for exploring surface-water quality in the United States and summarizing it with an interpretable **Water Quality Index (WQI)**.

Given a U.S. ZIP code, the Streamlit app:

- Geocodes the ZIP to latitude/longitude.
- Queries public APIs (USGS **National Water Information System** and the **Water Quality Portal**) for recent measurements.
- Aggregates those measurements into daily values for core parameters (pH, dissolved oxygen, temperature, turbidity, nitrate, and specific conductance).
- Computes a single WQI score (0–100) and classifies it as **Excellent / Good / Fair / Poor / Very Poor**.
- Optionally uses trained ML models to predict WQI and a SAFE/UNSAFE label, with rich feature-importance and SHAP-based explanations.

The goal is to make regulatory-grade water-quality data more accessible while highlighting environmental justice concerns around unequal data coverage and model performance.

## Demo

Screenshots of the application are available in `docs/screenshots/`.

## Key Features

- **ZIP-based search** – Enter a U.S. ZIP code to locate nearby monitoring sites and retrieve recent surface-water measurements.
- **Robust data collection** – Uses both the **USGS NWIS** API and the **Water Quality Portal (WQP)** with a fallback layer to maximize coverage and handle API failures.
- **Scientific WQI calculation** – Implements an NSF/EPA-inspired Water Quality Index using six core parameters with documented thresholds and weights.
- **ML-based predictions** – Random forest classifier and regressor trained on a Kaggle water-pollution dataset, with support for a core-parameter-only feature set for better geographic generalization.
- **Model interpretability** – Global feature importance (top features, availability in US data) and per-sample SHAP explanations, with mathematical checks that SHAP contributions sum to the prediction delta.
- **Fairness & environmental justice** – Scripts and tests to examine performance across regions and demographics, focusing on where data or models may under-serve certain communities.
- **Modular Python package** – Clean `src/` layout with organized modules for data collection, preprocessing, models, and utilities.

## Documentation

Most deep-dive docs live under `docs/`:

- **Standards & risk**
  - `docs/WQI_STANDARDS.md` – Scientific background for the WQI scale and parameter thresholds.
  - `docs/EXECUTIVE_SUMMARY_MODEL_RISK.md` – High-level summary of model risk and deployment cautions.
  - `docs/PRODUCTION_DEPLOYMENT_CHECKLIST.md` – Step-by-step checklist for any serious deployment.
- **Statistical validation**
  - `docs/AGENT_12_VALIDATION.md` – US-only vs calibration-model statistical comparison.
  - `docs/AGENT_12_STATISTICAL_VERIFICATION.md` – Independent verification of the Agent 12 validation math.
  - `docs/STATISTICAL_REVIEW_DOMAIN_CALIBRATION.md` – Review of the domain calibration implementation and tests.
- **Fairness & environmental justice**
  - `docs/ENVIRONMENTAL_JUSTICE_ANALYSIS.md` – Narrative and results for EJ-focused experiments.
  - `data/environmental_justice_*.csv` – Saved tabular results consumed by tests and analysis.
- **Testing & UX**
  - `docs/CHROME_DEVTOOLS_TEST_REPORT.md` – Browser-based performance and UX notes.
  - `docs/screenshots/*.png` – UI snapshots used in reports and for README images.
- **Course materials**
  - `docs/projectspec/project.pdf` – Main project brief.
  - `docs/projectspec/AI4ALL Ignite Rubrics.pdf`, `AI4ALL Ignite Syllabus.pdf`, `GitHub Guidelines.pdf` – Supporting course documents.

## Data Sources

### Training data (Kaggle)

ML models are trained on a Kaggle **water pollution / water quality** dataset (`data/raw/waterPollution.csv`) with global measurements of surface-water chemistry.

- Place the downloaded CSV at: `data/raw/waterPollution.csv`.
- The preprocessing pipeline (`src/preprocessing/feature_engineering.py`) builds a clean ML dataset and saves processed features under `data/processed/`.

> Note: The exact dataset name and license are defined in your course materials. Make sure you follow Kaggle/owner licensing when reusing the data.

### Live U.S. water-quality data

The app uses live (or near-real-time) U.S. monitoring data:

- **USGS National Water Information System (NWIS)** via `src/data_collection/usgs_client.py`.
- **Water Quality Portal (WQP)** via `src/data_collection/wqp_client.py` (aggregates USGS, EPA, and state agencies).

These APIs provide time series of pH, dissolved oxygen, temperature, turbidity, nitrate, conductivity, and related parameters around the requested ZIP code.

### Data directory layout

- `data/raw/` – Raw Kaggle CSV and any other unprocessed input files (not tracked in git except for `.gitkeep`).
- `data/processed/` – Cleaned and feature-engineered datasets used for modeling.
- `data/models/` – Saved classifier, regressor, and metadata files (model binaries are gitignored; metadata JSONs are tracked).

## Repository Structure

High-level layout:

```text
streamlit_app/
  app.py                # Streamlit UI (ZIP search, visualizations, explanations)

src/
  data_collection/      # USGS & WQP API clients, robust fallback logic
  geolocation/          # ZIP-to-lat/long mapping
  preprocessing/        # Kaggle loading, feature engineering, US feature pipelines
  models/               # Classifier, regressor, domain calibration, model utilities
  services/             # Search strategies and other orchestration logic
  utils/                # WQI calculator, feature definitions, importance utilities

data/
  raw/                  # Raw datasets (Kaggle, etc.)
  processed/            # Processed ML-ready features
  models/               # Trained models + metadata

scripts/
  *.py                  # Training, calibration, fairness analysis

docs/projectspec/       # Original project specification & GitHub guidelines (PDFs)
pyproject.toml          # Poetry project configuration
```

## Getting Started

### Prerequisites

- **Python**: 3.11+
- **Poetry** for dependency management
- A modern browser for the Streamlit UI

### Installation

From the project root:

```bash
poetry install
```

This installs both runtime and dev dependencies defined in `pyproject.toml`.

## Running the Streamlit App

From the project root:

```bash
poetry run streamlit run streamlit_app/app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

Typical workflow inside the app:

1. Enter a **U.S. ZIP code** (e.g., `20001`) and optional date range.
2. The app looks up nearby monitoring stations and fetches recent measurements from USGS NWIS and WQP.
3. Daily WQI scores are computed and visualized over time with color-coded quality categories.
4. If compatible ML models are available in `data/models/`, you will also see:
   - A **SAFE/UNSAFE** prediction with probability.
   - A predicted **WQI score** with confidence.
   - Global and per-sample **feature importance / SHAP explanations**.

If no compatible models are found, the app gracefully falls back to rule-based WQI calculations only.

> Environment timeouts: you can override default API timeouts via `USGS_TIMEOUT` and `WQP_TIMEOUT` environment variables if needed.

## Training the ML Models

The main training entrypoint is `train_models.py` at the project root.

Make sure `data/raw/waterPollution.csv` is available, then run:

```bash
# Full feature set (includes dataset-specific context features)
poetry run python train_models.py

# Core-parameter-only models (recommended for deployment)
poetry run python train_models.py --core-params-only
```

Training will:

- Load and validate the Kaggle dataset.
- Build a feature matrix and labels under `data/processed/`.
- Train a random forest **classifier** (SAFE vs UNSAFE) and **regressor** (continuous WQI).
- Save models and metadata into `data/models/` with timestamped filenames.
- Log performance metrics and check simple success criteria (e.g., classifier accuracy ≥ 75%, regressor R² ≥ 0.6).

The Streamlit app then loads the **latest** compatible models via `src/models/model_utils.py`.

## Code Quality

```bash
poetry run black .
poetry run flake8 src
poetry run mypy src
```

## Water Quality Index (WQI)

The WQI implementation lives in `src/utils/wqi_calculator.py` and is inspired by the **National Sanitation Foundation Water Quality Index (NSF-WQI)** and **EPA** standards.

- Parameters used: **pH**, **dissolved oxygen**, **temperature**, **turbidity**, **nitrate**, and **specific conductance**.
- Each parameter is mapped to a 0–100 quality score using scientifically motivated thresholds.
- Parameter scores are combined with weights to produce a single WQI (0–100).
- WQI categories used throughout the app:
  - **90–100**: Excellent
  - **70–89**: Good
  - **50–69**: Fair
  - **25–49**: Poor
  - **0–24**: Very Poor

The WQI values shown in the app are meant to be intuitive summaries, not a replacement for regulatory assessments or local advisories.

## Ethics, Fairness & Limitations

This is an educational project, not a production safety tool. Please keep in mind:

- **Do not use this app as the sole basis for drinking or recreation decisions.** Always consult local water authorities and official advisories.
- **Data coverage is uneven.** Many communities have sparse monitoring; model performance and WQI estimates are more uncertain in data-poor regions.
- **Training data bias.** The Kaggle dataset and U.S. monitoring networks over-represent certain geographies and conditions, which can bias both the WQI calibration and ML models.
- **Fairness & environmental justice.** The repository includes scripts (under `scripts/`) to explore whether errors or uncertainty cluster in particular demographic or geographic groups.
- **Interpretability is an aid, not a guarantee.** SHAP and feature-importance plots explain model behavior but do not make the underlying data unbiased.

Treat all numbers as **approximate indicators** to support learning and discussion about environmental justice, not as definitive regulatory values.

## Acknowledgements

- **AI4ALL Ignite** instructors and mentors for guidance and project structure.
- **Kaggle** and the creators of the water pollution dataset used for training.
- **USGS**, **EPA**, and partners behind the **National Water Information System** and **Water Quality Portal** for providing open water-quality data.
