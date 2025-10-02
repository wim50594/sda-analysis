# Statistical Disclosure Attacks Analysis
This project provides a research and evaluation framework for Statistical Disclosure Attacks (SDA) and their variants. It includes simulation of sender–receiver communication datasets, multiple attack implementations, baseline comparisons using regression and Transformers, and evaluation/visualization of results.

## Installation

### Using `uv` (recommended if you have uv installed)
```bash
uv sync --group dev

# Optional: install CUDA-enabled Torch for GPU acceleration
uv sync --group cuda129 --group dev

uv run python -m ipykernel install --user --name sda --display-name "SDA"
```

### Using `pip`
If you prefer standard pip + venv workflow:
```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install . ipykernel>=6.30.1
python -m ipykernel install --user --name sda --display-name "SDA"
```
