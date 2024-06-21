# deep-neuronmorpho

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/kpeez/deep-neuronmorpho/blob/main/LICENSE)

Deep learning based approaches for neuron morphology embeddings. Contact [Kyle Puhger](https://github.com/kpeez) for questions.

- **Github repository**: <https://github.com/kpeez/deep-neuronmorpho/>

## Installation

> If you do not have a CUDA enabled machine, please use the `requirements-cpu.txt` file to install the necessary dependencies.
> Note: The GPU version only runs on CUDA 11, not CUDA 12.

This package requires the following dependencies:

- Python ≥ 3.10
- PyTorch >= 2.0.0
- DGL == 1.1.3 (GPU version ran using CUDA 11.8)

The easiest way to install the packages is to create a new virtual environment and install the appropriate requirements file.

- `requirements.txt` contains the base requirements for a CUDA enabled machine.
- `requirements-dev.txt` contains the requirements for development on a CUDA enabled machine.
- `requirements-cpu.txt` contains the base requirements for a CPU only machine.
- `requirements-dev-cpu.txt` contains the requirements for development on a CPU only machine.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt # replace w/ the relevant requirements file
```

This package can also be installed from the Makefile using one of the following commands:

- `make install` for a CUDA enabled machine
- `make install-dev` for development on a CUDA enabled machine
- `make install-cpu` for a CPU only machine
- `make install-dev-cpu` for development on a CPU only machine

## Development

- When making contributions to this project, ensure you have the development requirements installed.
- Run the `make check` and `make test` commands to ensure the code is formatted correctly and all tests pass.
