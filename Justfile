set shell := ["bash", "-euo", "pipefail", "-c"]

default: help

# Check if `uv` is installed, and install it if not.
_check_uv:
    @if ! command -v uv > /dev/null 2>&1; then \
        echo "uv is not installed, installing now..."; \
        curl -LsSf https://astral.sh/uv/install.sh | sh; \
    fi

# Install the virtual environment and pre-commit hooks
install: _check_uv
    echo "📦 Creating virtual environment"
    uv sync --all-extras --python=3.12
    echo "🛠️ Installing developer tools..."
    uv run pre-commit install

# Export dependencies to requirements.txt
requirements: _check_uv
    echo "Exporting dependencies to requirements.txt..."
    uv export -o requirements.txt --no-hashes --no-dev

# Run code quality tools (ruff and pre-commit)
check:
    echo "⚡️ Linting code: Running ruff"
    uv run ruff check .
    echo "🧹 Checking code: Running pre-commit"
    uv run pre-commit run --all-files

# Test the code with pytest
test:
    echo "✅ Testing code: Running pytest"
    uv run pytest

# Build and serve the documentation locally
docs:
    uv run mkdocs serve

# Test if documentation can be built without errors
docs-test:
    echo "⚙️ Testing documentation build"
    uv run mkdocs build --strict

# Update dependencies and pre-commit hooks
update:
    echo "⚙️ Updating dependencies and pre-commit hooks"
    uv lock --upgrade
    uv run pre-commit autoupdate

help:
    @just --list