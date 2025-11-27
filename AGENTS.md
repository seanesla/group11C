# Repository Guidelines

## Project Structure & Module Organization
- `src/`: core packages for data collection, geolocation, preprocessing, models, and shared services; keep new utilities small and testable.
- `streamlit_app/app.py`: user-facing UI; avoid putting business logic here—add to `src/services/` or `src/utils/` and import.
- `data/`: raw Kaggle CSV in `data/raw/`, processed features in `data/processed/`, model binaries + metadata in `data/models/` (binaries stay gitignored; metadata JSON is tracked).
- `tests/`: unit tests (`@unit`) and integration tests (`@integration` for real APIs). Fixtures live in `tests/fixtures/`.
- `scripts/`: CLI helpers (training, sampling). Prefer parameterized scripts over notebooks for repeatability.

## Build, Test, and Development Commands
- Install deps: `poetry install` (Python 3.11+).
- Run UI: `poetry run streamlit run streamlit_app/app.py`.
- Fast tests: `poetry run pytest` (excludes `integration`).
- Specific module test: `poetry run pytest tests/test_streamlit_app.py::TestFetchWaterQualityData -q`.
- Lint/format/type-check: `poetry run black .`, `poetry run flake8 src`, `poetry run mypy src`.
- Train models (core features only): `poetry run python train_models.py --core-params-only`.

## Coding Style & Naming Conventions
- Black-formatted, 4-space indents, PEP8. Imports ordered stdlib → third-party → local.
- Use type hints everywhere; prefer pure functions; inject sessions/clients for testability.
- Naming: `snake_case` for vars/functions, `CapWords` for classes, constants in `UPPER_SNAKE`.

## Testing Guidelines
- Framework: pytest. File names `test_*.py`, functions `test_*`.
- Mark live API hits with `@pytest.mark.integration`; keep defaults fast/deterministic.
- Add unit tests with fixtures; avoid network in unit scope. Target regression tests for any bug fix.

## Commit & Pull Request Guidelines
- Commit prefixes: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`. Keep messages imperative and short.
- PRs should state scope, testing done (`pytest`, `streamlit` smoke), and include screenshots for UI changes when possible.

## Security & Configuration Tips
- No secrets in code; use env vars (e.g., `WQP_API_KEY` if added later). Respect timeouts/retries for external calls.
- Large binaries (models) stay out of git; keep metadata JSONs for traceability.

## Agent-Specific Notes
- When expanding Streamlit behavior, keep search logic in `src/services/` and reuse in tests.
- Default to explicit timeouts for long-running tools and API requests; never leave commands hanging.
