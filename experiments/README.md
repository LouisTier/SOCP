# Experiments

The code that reproduces the paper's results. Every step reads
[`config.yaml`](config.yaml), and the [Makefile](../Makefile) runs the steps in
order.

## Pipeline steps

| File | Make target | What it does |
| --- | --- | --- |
| `download_data.py` | `make download` | Downloads the nine public datasets into `data/` |
| `prepare.py` | `make prepare` | Splits each dataset, fits the predictors and score models, and builds the representations |
| `run.py` | `make run` | Calibrates every score and calibrator and evaluates it on the test split, with the Regime 3 and SOM-sensitivity analyses |
| `aggregate.py` | `make aggregate` | Averages the seeds and runs the paired Wilcoxon tests with Holm's correction |

## Shared code

| File | Used by | What it does |
| --- | --- | --- |
| `cli.py` | `make prepare`, `run`, `aggregate` | The `socp-experiments` command |
| `campaign.py` | `prepare.py`, `run.py`, `aggregate.py` | Reads the config and selects the datasets, seeds, and device |
| `methods.py` | `run.py` | Builds each score and calibrator, including its hyperparameter selection |
| `analysis_contract.py` | `run.py`, `aggregate.py` | The paper's datasets, base scores, and dataset labels |

## Scripts

| File | What it does |
| --- | --- |
| `scripts/imagenet_cache.py` | Builds the ImageNet-1k cache from the validation images and ResNet-50 weights (see the main README) |
| `scripts/run_parallel.sh` | Runs prepare, run, and aggregate for every dataset and seed in parallel, one CPU each |
