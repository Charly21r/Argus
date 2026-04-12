.PHONY: test lint typecheck format serve clean help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

test:  ## Run unit tests
	python -m pytest tests/ -v --tb=short

test-unit:  ## Run unit tests only
	python -m pytest tests/unit/ -v --tb=short

test-bias:  ## Run bias constraint tests (requires model artifacts)
	python -m pytest tests/test_bias_constraints.py -v --tb=short

lint:  ## Run linter (ruff)
	ruff check src/ scripts/ tests/

typecheck:  ## Run type checker (mypy)
	mypy src/

format:  ## Auto-format code
	ruff format src/ scripts/ tests/
	ruff check --fix src/ scripts/ tests/

serve:  ## Start the API server
	uvicorn serving.app:app --host 0.0.0.0 --port 8000 --reload --app-dir src

clean:  ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info

install:  ## Install project in dev mode
	pip install -e ".[dev]"

install-training:  ## Install project with training deps
	pip install -e ".[training]"

install-serving:  ## Install project with serving deps
	pip install -e ".[serving]"
