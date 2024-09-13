# deep-neuronmorpho

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/kpeez/deep-neuronmorpho/blob/main/LICENSE)

Deep learning based approaches for neuron morphology embeddings. Contact [Kyle Puhger](https://github.com/kpeez) for questions.

## Installation

This package requires the following dependencies:

- `python == 3.11`
- `torch == 2.3.0`
- `DGL == 2.2.1`
- `pytorch-lightning >= 2.4.0`

The easiest way to install the packages is to create a new virtual environment and install from the `requirements.txt` file.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Development

- When making contributions to this project, ensure you have the development requirements installed. You can install the package for development purposes from the Makefile using the `make install` command.
- Before submitting your contributions, run the `make check` and `make test` commands to ensure the code is formatted correctly and all tests pass. If you have made changes to the code, please add tests to cover the new functionality.
- If you have made changes to the documentation, please run the `make docs` command to ensure the documentation builds correctly.
