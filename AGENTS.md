# Repository Guidelines

## Project Structure & Module Organization
- `src/`: core packages for data collection, geolocation, preprocessing, modeling, and shared services. Add new utilities as small, testable modules.
- `streamlit_app/app.py`: user-facing UI only. Keep business logic in `src/services/` or `src/utils/` and import it.
- `data/`: raw CSVs in `data/raw/`, processed features in `data/processed/`, model binaries and metadata in `data/models/` (binaries gitignored, metadata JSON tracked).
- `scripts/`: CLI helpers for training, sampling, and analysis. Prefer parameterized scripts over notebooks.

## Build, Test, and Development Commands
- Install deps: `poetry install` (Python 3.11+).
- Run UI: `poetry run streamlit run streamlit_app/app.py`.
- Train models (core features only): `poetry run python train_models.py --core-params-only`.
- Run tests: `poetry run pytest` (target regression tests for any bug fix).

## Coding Style & Naming Conventions
- Python only: Black-formatted, 4-space indents, PEP8.
- Imports ordered: stdlib → third-party → local.
- Naming: `snake_case` for functions/vars, `CapWords` for classes, `UPPER_SNAKE` for constants.
- Use type hints everywhere; prefer pure functions and dependency injection for sessions/clients.
- Lint/format/type-check: `poetry run black .`, `poetry run flake8 src`, `poetry run mypy src`.

## Testing Guidelines
- Use `pytest` for unit and regression tests under `tests/` (or nearby to the code under test).
- Name tests descriptively (`test_feature_x_handles_missing_values`).
- For bugs, first add a failing regression test, then fix the code.

## Commit & Pull Request Guidelines
- Commit prefixes: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`. Keep messages short and imperative.
- PRs should clearly state scope, link issues when relevant, and describe testing performed (`pytest`, `streamlit` smoke checks). Include screenshots for UI changes when possible.

## Security & Agent-Specific Notes
- Do not commit secrets; use env vars (e.g., `WQP_API_KEY`) and respect timeouts/retries for external calls.
- When expanding Streamlit behavior, keep search/data logic in `src/services/` and reuse it in tests.

