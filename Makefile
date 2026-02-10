.DEFAULT_GOAL := help
# Whichever fixture comes first alphabetically, so this keeps working once you record
# real runs of your own. Override with: make demo FIXTURE=fixtures/runs/yours.json
FIXTURE ?= $(firstword $(wildcard fixtures/runs/*.json))

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

trace: ## Serve the static trace viewer at http://localhost:8111/demo/trace.html
	@echo "open http://localhost:8111/demo/trace.html"
	python3 -m http.server 8111

.PHONY: help up down test lint fmt demo record trace
