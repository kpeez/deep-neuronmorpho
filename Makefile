.PHONY: check_uv install install-dev install-cpu install-dev-cpu check test docs docs-test update help

DGL_URL := https://data.dgl.ai/wheels/cu118/repo.html

check_uv: # install `uv` if not installed
	@if ! command -v uv > /dev/null 2>&1; then \
		echo "uv is not installed, installing now..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	@uv self update

install: check_uv ## Install the virtual environment and  pre-commit hooks
	@echo "📦 Creating virtual environment"
	@uv sync --all-extras -f $(DGL_URL)
	@echo "🛠️ Installing developer tools..."
	@uvx pre-commit install
	@. .venv/bin/activate && mypy --install-types --non-interactive

requirements: check_uv
	@echo "Making requirements.txt and requirements-cpu.txt"
	@uv pip compile pyproject.toml -o requirements.txt -f $(DGL_URL)
	@uv pip compile pyproject.toml -o requirements-cpu.txt --python-platform=macos

check: ## Run code quality tools
	@echo "⚡️ Linting code: Running ruff"
	@uvx ruff check .
	@echo "🧹 Checking code: Running pre-commit"
	@uvx pre-commit run --all-files
	@echo "🔬 Static type checking: Running mypy"
	@. .venv/bin/activate && mypy .

test: ## Test the code with pytest
	@echo "✅ Testing code: Running pytest"
	@uvx pytest

docs: ## Build and serve the documentation
	@uvx mkdocs serve

docs-test: ## Test if documentation can be built without warnings or errors
	@echo "⚙️ Testing documentation build"
	@uvx mkdocs build --strict

update: ## Update pre-commit hooks
	@echo "⚙️ Updating dependencies and pre-commit hooks"
	@uv lock --upgrade
	@uvx pre-commit autoupdate

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
