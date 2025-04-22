# deep-neuronmorpho

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/kpeez/deep-neuronmorpho/blob/main/LICENSE)

Deep learning based approaches for neuron morphology embeddings. Contact [Kyle Puhger](https://github.com/kpeez) for questions.

## Installation

This package uses the following dependencies:

- `python >= 3.11`
- `torch >= 2.6.0`
- `torch-geometric >= 2.6.1`
- `pytorch-lightning >= 2.5.1`

**Using `uv` (Recommended):**

The easiest way to install `deep-neuronmorpho` is with [`uv`](https://github.com/astral-sh/uv):

```bash
# clone the repository
git clone https://github.com/kpeez/deep-neuronmorpho.git
cd deep-neuronmorpho
# install the latest release
uv sync

# on linux and macos you can use make to install the package
make install
```

**Using `pip`:**
The easiest way to install the packages is to create a new virtual environment and install from the `requirements.txt` file.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Contributing

- When making contributions to this project, ensure you have the development requirements installed. You can install the package for development purposes from the Makefile using the `make install` command.
- Before submitting your contributions, run the `make check` and `make test` commands to ensure the code is formatted correctly and all tests pass. If you have made changes to the code, please add tests to cover the new functionality.
- If you have made changes to the documentation, please run the `make docs` command to ensure the documentation builds correctly.
