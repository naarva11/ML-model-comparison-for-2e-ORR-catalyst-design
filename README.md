# KOH 2e-ORR Selectivity Modeling

This repository contains the code and processed dataset used to reproduce the machine-learning analyses for the associated manuscript on H$_2$O$_2$ selectivity modeling in alkaline 2e-ORR.

## Repository contents

- `main_model.py` — main modeling workflow, cross-validation benchmarking, feature importance, validation-curve plotting, and output table export.
- `learning_curve.py` — synthetic-data pretraining / learning-curve experiment.
- `modeling_data.xlsx` — processed dataset workbook used by the scripts.
- `DATA_DICTIONARY.md` — description of workbook sheets and columns.
- `requirements.txt` — installation requirements.
- `requirements-lock-reference.txt` — reference package versions recorded for this release package.
- `LICENSE` — Apache-2.0 license for the source code in this repository.
- `CITATION.cff` — citation metadata template for GitHub / Zenodo release citation.

## Python environment

Target local Python version recorded for this release: **Python 3.14.3**.

## Installation

Create and activate a virtual environment, then install the requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Input data

Both scripts look for the dataset file in the same directory as the Python scripts:

- `modeling_data.xlsx`

The main modeling sheet used by the code is:

- `modeling_data`

## Reproducing the main results

Run the main modeling pipeline:

```bash
python main_model.py
```

Outputs are written to:

- `main_results/`

Run the learning-curve experiment:

```bash
python learning_curve.py
```

Outputs are written to:

- `learning_results/`

## Expected generated outputs

`main_model.py` produces:
- model-comparison tables
- selectivity vs parameter plots
- Pearson correlation matrix
- feature-importance plot
- validation-curve plot
- Excel summary tables

`learning_curve.py` produces:
- learning-curve summary tables
- baseline and synthetic-pretraining comparison tables
- R2 / RMSE / MAE learning-curve figures

## License

The source code in this repository is released under the **Apache License 2.0**. See `LICENSE`.

The processed dataset is included for reproducibility of the study. When reusing the dataset, cite the associated manuscript and the original literature sources where relevant.
