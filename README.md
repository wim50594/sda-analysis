# Statistical Disclosure Attacks
A small research / evaluation project containing multiple implementations of Statistical Disclosure Attacks (SDA) and comparisons to baseline methods (linear regression and a Transformer model).

## Installation

### Using `uv` (recommended if you have uv installed)
```bash
uv sync --dev
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