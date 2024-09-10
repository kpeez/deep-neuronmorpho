# deep-neuronmorpho

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/kpeez/deep-neuronmorpho/blob/main/LICENSE)

Deep learning based approaches for neuron morphology embeddings. Contact [Kyle Puhger](https://github.com/kpeez) for questions.

## Installation

> If you do not have a CUDA enabled machine, please use the `requirements-cpu.txt` file to install the necessary dependencies.
> Note: The GPU version only runs on CUDA 11, not CUDA 12.

This package requires the following dependencies:

- `python == 3.11`
- `torch == 2.1.0`
- `DGL == 2.0.0`

The easiest way to install the packages is to create a new virtual environment and install the appropriate requirements file.

- `requirements.txt` contains requirements for a CUDA enabled machine.
- `requirements-cpu.txt` contains the base requirements for a CPU only machine.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt # replace w/ the relevant requirements file
```

## Development

- When making contributions to this project, ensure you have the development requirements installed. You can install the package for development purposes from the Makefile using the `make install` command.
- Before submitting your contributions, run the `make check` and `make test` commands to ensure the code is formatted correctly and all tests pass. If you have made changes to the code, please add tests to cover the new functionality.
- If you have made changes to the documentation, please run the `make docs` command to ensure the documentation builds correctly.
