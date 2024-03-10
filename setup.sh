#!/bin/bash

check_conda() {
    if [ -n "$CONDA_DEFAULT_ENV" ]; then
        echo "Deactivating conda environment $CONDA_DEFAULT_ENV..."
        conda deactivate
    elif [ -n "$VIRTUAL_ENV" ] && [ -n "$CONDA_PREFIX" ]; then
        echo "Both VIRTUAL_ENV and CONDA_PREFIX are set. Choosing to unset CONDA_PREFIX..."
        unset CONDA_PREFIX
    else
        echo "No conda environment is currently active."
    fi
}

ensure_uv_installed() {
    if ! command -v uv &> /dev/null; then
        echo "🛠️ uv not found, installing..."
        pip install uv
    else
        echo "✅ uv is already installed."
    fi
}

install() {
    ensure_uv_installed
    echo "📦 Creating and seeding virtual environment"
    uv venv --seed
    check_conda
    echo "📦 Installing dependencies"
    uv pip compile -o requirements.txt pyproject.toml
    source .venv/bin/activate
    uv pip install -r requirements.txt
}

install_dev() {
    ensure_uv_installed
    echo "📦 Creating virtual environment"
    uv venv --seed
    check_conda
    echo "📦 Installing dependencies"
    uv pip compile -o requirements.txt pyproject.toml
    uv pip compile -o requirements-dev.txt --extra=dev pyproject.toml
    source .venv/bin/activate
    uv pip install -r requirements-dev.txt
}

install_cuda() {
    install_dev
    echo "📦 Installing CUDA-dependent packages..."
    echo "Installing CUDA-enabled version of DGL..."
    source .venv/bin/activate
    pip uninstall dgl -y
    pip install dgl==1.1.3 -f https://data.dgl.ai/wheels/cu118/repo.html
}

case "$1" in
    install)
        install
        ;;
    install-dev)
        install_dev
        ;;
    install-cuda)
        install_cuda
        ;;
    *)
        echo "Usage: $0 {install|install-dev|install-cuda}"
        exit 1
        ;;
esac

