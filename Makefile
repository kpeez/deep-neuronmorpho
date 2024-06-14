.PHONY: check_uv install install-dev install-cpu install-dev-cpu check test docs docs-test update help

DGL_URL := https://data.dgl.ai/wheels/cu121/repo.html

check_uv: # install `uv` if not installed
	@if ! command -v uv > /dev/null 2>&1; then \
		echo "uv is not installed, installing now..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi

install: check_uv ## Install the virtual environment and pre-commit hooks
	@echo "📦 Creating and seeding virtual environment"
	@uv venv --seed
	@echo "📦 Installing dependencies"
	@. .venv/bin/activate && \
		uv pip compile pyproject.toml -o requirements.txt -f $(DGL_URL) && \
		uv pip sync requirements.txt -f $(DGL_URL) && \
		uv pip install -e .

install-dev: check_uv ## Install the virtual environment and pre-commit hooks
	@echo "📦 Creating and seeding virtual environment"
	@uv venv --seed
	@echo "📦 Installing dependencies"
	@. .venv/bin/activate && \
		uv pip compile pyproject.toml -o requirements.txt -f $(DGL_URL) && \
		uv pip compile pyproject.toml -o requirements-dev.txt -f $(DGL_URL) --extra=dev && \
		uv pip sync requirements-dev.txt -f $(DGL_URL) && \
		uv pip install -e . && \
		pre-commit install

install-cpu: check_uv ## Install the virtual environment and pre-commit hooks
	@echo "📦 Creating and seeding virtual environment"
	@uv venv --seed
	@echo "📦 Installing dependencies"
	@. .venv/bin/activate && \
		uv pip compile pyproject.toml -o requirements-cpu.txt && \
		uv pip sync requirements-cpu.txt && \
		uv pip install -e .

install-dev-cpu: check_uv ## Install the virtual environment and pre-commit hooks
	@echo "📦 Creating and seeding virtual environment"
	@uv venv --seed
	@echo "📦 Installing dependencies"
	@. .venv/bin/activate && \
		uv pip compile pyproject.toml -o requirements-cpu.txt && \
		uv pip compile pyproject.toml -o requirements-dev-cpu.txt --extra=dev && \
		uv pip sync requirements-dev-cpu.txt && \
		uv pip install -e . && \
		pre-commit install; mypy --install-types --non-interactive

check: ## Run code quality tools
	@. .venv/bin/activate && \
		echo "⚡️ Linting code: Running ruff" && \
		ruff check . && \
		echo "🧹 Checking code: Running pre-commit" && \
		pre-commit run --all-files && \
		echo "🔬 Static type checking: Running mypy" && \
		mypy .

test: ## Test the code with pytest
	@echo "🧪 Testing code: Running pytest" && \
		source .venv/bin/activate; pytest --doctest-modules

docs: ## Build and serve the documentation
	@. .venv/bin/activate && \
	mkdocs serve

docs_test: ## Test if documentation can be built without warnings or errors
	@echo "⚙️ Testing documentation build"
	@. .venv/bin/activate && \
		mkdocs build --strict

update: ## Update pre-commit hooks
	@echo "⚙️ Updating environment and pre-commit hooks"
	@. .venv/bin/activate && \
		pre-commit autoupdate

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help