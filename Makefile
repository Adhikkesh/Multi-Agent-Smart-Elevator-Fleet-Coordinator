.PHONY: help install serve test test-py test-ui e2e lint format clean ui-dev ui-build

help:
	@echo "LiftZero Multi-Agent Smart Elevator Fleet Coordinator"
	@echo ""
	@echo "Targets:"
	@echo "  install    Sync python packages and install pnpm dependencies"
	@echo "  serve      Start FastAPI server at http://localhost:8000"
	@echo "  test       Run both Python backend and frontend unit tests"
	@echo "  test-py    Run Python test suite with pytest"
	@echo "  test-ui    Run Vitest frontend unit tests"
	@echo "  e2e        Run Playwright end-to-end browser tests"
	@echo "  lint       Check code style and linting (Python & TypeScript)"
	@echo "  format     Format Python code with ruff"
	@echo "  ui-dev     Start Vite development server"
	@echo "  ui-build   Compile React frontend to static bundle in src/elevator_mas/web/dist"
	@echo "  clean      Remove temporary build artifacts and caches"

install:
	uv sync
	cd frontend && pnpm install

serve:
	uv run elevator serve

test: test-py test-ui

test-py:
	uv run pytest

test-ui:
	cd frontend && pnpm test

e2e:
	cd frontend && pnpm e2e

lint:
	uv run ruff check .
	uv run ruff format --check .
	cd frontend && pnpm lint

format:
	uv run ruff format .
	uv run ruff check --fix .

ui-dev:
	cd frontend && pnpm dev

ui-build:
	cd frontend && pnpm build

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov coverage.xml
	rm -rf frontend/node_modules/.vite frontend/dist frontend/coverage
