.DEFAULT_GOAL := help
FIXTURE ?= fixtures/runs/default.json

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-10s %s\n", $$1, $$2}'

up: ## Build and start redis, api and worker
	docker compose up --build

down: ## Stop everything and drop volumes
	docker compose down -v

test: ## Run the test suite with coverage (live tests excluded)
	uv run pytest --cov=ra --cov-report=term-missing --cov-fail-under=80 -m "not live"

lint: ## Lint and check formatting
	uv run ruff check . && uv run ruff format --check .

fmt: ## Apply formatting and safe lint fixes
	uv run ruff check --fix . && uv run ruff format .

demo: ## Replay a recorded run offline, no keys needed
	uv run python -m ra.replay $(FIXTURE)

record: ## Record a real run to fixtures. Usage: make record Q="your question"
	uv run python scripts/record_run.py "$(Q)"

.PHONY: help up down test lint fmt demo record
