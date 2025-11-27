# Repository Guidelines

## Project Structure & Modules
- Application code lives in `src/` (data collection, geolocation, WQI utils, models, preprocessing) and the Streamlit UI in `streamlit_app/app.py`.
- Tests are under `tests/` with realistic API fixtures in `tests/fixtures/`.
- Data artifacts and trained models are stored in `data/` (do not commit new large binaries without review).
- Scripts for training/analysis are in `scripts/`; documentation and reports live in `docs/`.

## Environment, Build, and Run Commands
- Install and activate: `poetry install` then `poetry shell`.
- Run the web app: `poetry run streamlit run streamlit_app/app.py`.
- Execute fast test suite: `poetry run pytest` (default excludes `@integration` tests that hit live APIs/browsers).
- Lint/format/type-check: `poetry run black .`, `poetry run flake8 src`, `poetry run mypy src`.

## Coding Style & Naming Conventions
- Python 3.11+, Black-formatted (4-space indent, no manual line wrapping). Keep imports sorted logically; prefer standard-library → third-party → local.
- Use type hints; mypy must pass. Follow PEP8: `snake_case` for modules/functions/variables, `CapWords` for classes, constants in `UPPER_SNAKE`.
- Favor pure functions in `src/utils/` and keep API clients in `src/data_collection/` narrow and testable (inject session when possible).

## Testing Guidelines
- Framework: pytest (`test_*.py`, functions `test_*`). Markers available: `unit`, `integration`, `slow`.
- Aim for good coverage on core logic; coverage is reported but not enforced as a hard gate. Add fixtures under `tests/fixtures/` and prefer deterministic inputs over live API calls; integration tests that hit REAL endpoints should be explicitly marked.
- For new features, add focused unit tests plus an end-to-end check if UI or data pipeline behavior changes.

## Commit & Pull Request Guidelines
- Commit style follows conventional prefixes (e.g., `feat:`, `fix:`, `docs:`, `refactor(ui):`, `test:`) with concise intent.
- PRs should include: summary of changes, testing performed (`pytest`, `streamlit` smoke), and any screenshots for UI updates. Link to issues/tasks when applicable.

## Security & Data Handling
- Do not embed secrets; rely on environment variables for API keys. Avoid committing large model binaries—use `data/models/` for staged artifacts and coordinate if size exceeds repo norms.
- When adding new external calls, ensure timeout/retry logic and document expected response shapes in the relevant client docstring or README snippet.


Long-running tooling (tests, docker compose, migrations, etc.) must always be invoked with sensible timeouts or in non-interactive batch mode. Never leave a shell command waiting indefinitely—prefer explicit timeouts, scripted runs, or log polling after the command exits.
