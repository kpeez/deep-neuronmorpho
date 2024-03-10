.PHONY: install install-cuda check test build clean-build docs docs-test update help

install: ## Install the poetry environment and install the pre-commit hooks
	@echo "📦 Creating virtual environment using poetry"
	@poetry install	
	@poetry run pre-commit install
	@poetry shell

install-cuda: ## install CUDA-dependent pacakges
	@echo "📦 Installing CUDA-dependent packages..."
	@poetry run pip install --upgrade pip
	@echo "Installing CUDA-enabled version of DGL..."
	@poetry run pip uninstall dgl -y
	@poetry run pip install dgl==1.1.3 -f https://data.dgl.ai/wheels/cu118/repo.html

check: ## Run code quality tools.
	@echo "🔒 Checking Poetry lock file consistency with 'pyproject.toml': Running poetry lock --check"
	@poetry lock --check
	@echo "🧹 Linting code: Running pre-commit"
	@poetry run pre-commit run -a
	@echo "🔬 Static type checking: Running mypy"
	@poetry run mypy

test: ## Test the code with pytest
	@echo "✅ Testing code: Running pytest"
	@poetry run pytest --doctest-modules

build: clean-build ## Build wheel file using poetry
	@echo "🛞 Creating wheel file"
	@poetry build

clean-build: ## clean build artifacts
	@rm -rf dist

docs: ## Build and serve the documentation
	@poetry run mkdocs serve

docs-test: ## Test if documentation can be built without warnings or errors
	@poetry run mkdocs build -s

update: ## Update the poetry environment and pre-commit hooks
	@echo "⚙️ Updating poetry environment and pre-commit hooks"
	@poetry update
	@poetry run pre-commit autoupdate

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
