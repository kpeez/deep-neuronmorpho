.PHONY: check-conda ensure-uv-installed install install-dev check install_cuda test docs docs-test update help

check-conda:
	@if [ -n "$$CONDA_DEFAULT_ENV" ]; then \
		echo "Deactivating conda environment $$CONDA_DEFAULT_ENV..."; \
		conda deactivate; \
	else \
		echo "No conda environment is currently active."; \
	fi

ensure-uv-installed:
	@if [ -z $$(which uv) ]; then \
		echo "🛠️ uv not found, installing..."; \
		pip install uv; \
	else \
		echo "✅ uv is already installed."; \
	fi

install: ensure-uv-installed ## Install the virtual environment and pre-commit hooks
	@echo "📦 Creating and seeding virtual environment"
	@uv venv --seed
	@$(MAKE) check-conda
	@echo "📦 Installing dependencies"
	@uv pip compile -o requirements.txt pyproject.toml
	@. .venv/bin/activate && \
		uv pip install -r requirements.txt

install-dev: ensure-uv-installed ## Install the virtual environment for development
	@echo "📦 Creating virtual environment"
	@uv venv --seed
	@$(MAKE) check-conda
	@echo "📦 Installing dependencies"
	@uv pip compile -o requirements.txt pyproject.toml
	@uv pip compile -o requirements-dev.txt --extra=dev pyproject.toml
	@. .venv/bin/activate && \
		uv pip install -r requirements-dev.txt

install-cuda: install-dev ## Install CUDA-dependent packages
	@echo "📦 Installing CUDA-dependent packages..."
	@echo "Installing CUDA-enabled version of DGL..."
	@. .venv/bin/activate && \
		pip uninstall dgl -y && \
		pip install dgl==1.1.3 -f https://data.dgl.ai/wheels/cu118/repo.html

check: ## Run code quality tools
	@echo "🧹 Linting code: Running pre-commit" && \
		source .venv/bin/activate && \
		ruff check --all && \
		pre-commit run -a
	@echo "🔬 Static type checking: Running mypy" && \
		source .venv/bin/activate && \
		mypy --install-types --non-interactive

test: ## Test the code with pytest
	@echo "🧪 Testing code: Running pytest" && \
	source .venv/bin/activate && \
	which pytest && \
	pytest --doctest-modules

docs: ## Build and serve the documentation
	@echo "📃 Building and serving documentation" && \
	source .venv/bin/activate && \
	mkdocs serve

docs-test: ## Test if documentation can be built without warnings or errors
	# TODO - Implement

update: ## Update pre-commit hooks
	@echo "⚙️ Updating environment and pre-commit hooks" && \
	source .venv/bin/activate && \
	pre-commit autoupdate

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
