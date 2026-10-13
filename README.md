# SOCP: Self-Organized Conformal Prediction

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2606.29403-b31b1b.svg)](https://arxiv.org/abs/2606.29403)
[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-4b44ce.svg)](https://arxiv.org/abs/2606.29403)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/socp-logo-dark.svg">
  <img src=".github/socp-logo-light.svg" alt="SOCP">
</picture>

This repository is the official code for [*Self-Organized Conformal Prediction: Reducing Regional Coverage Gaps with Unsupervised Group Discovery*](https://arxiv.org/abs/2606.29403) (Berthier et al., NeurIPS 2026).

[Paper](https://arxiv.org/abs/2606.29403)
| [Reproducing the experiments](#reproducing-the-experiments)
| [Citation](#citation)

</div>

---

## Overview

Split conformal prediction (SCP) guarantees $`1-\alpha`$ coverage on average. But one pooled threshold can undercover some regions of the feature space and overcover others.

**SOCP** keeps the predictor and the nonconformity score and changes only the calibration. A self-organizing map (SOM), trained on training inputs before calibration, splits the feature space into cells. Each test point is then calibrated with the calibration scores retrieved around its best-matching unit (BMU). The result is a trade-off: lower regional coverage gaps than pooled calibration of the same score, for a small increase in output size.

### Key results

Across ten regression and classification benchmarks, with ten paired seeds:

| Comparison | Result |
| --- | --- |
| SO-SCP vs pooled SCP | Lower mean weighted coverage gap (WCovGap) on **all ten datasets**: **-14.8%** on average, for a **+3.0%** mean change in output size |
| Five scores per task | SO calibration lowers the ten-seed mean WCovGap in **43 of 50** score and dataset comparisons |
| Other calibrators: LCP, RLCP, Mondrian k-means, Classwise, Clustered | On Bike Sharing, Bio, and California Housing, only SO-SCP and pooled SCP keep every interval finite. Mondrian k-means lowers WCovGap more than SO-SCP on the five classification datasets, but enlarges sets more on four of them |
| External k-means partitions with 10, 25, and 50 groups | SO-SCP lowers WCovGap for **every dataset** at all three granularities |
| Diagnostics that need no feature-space partition | Held-out WSC target error improves on **all nine** datasets with a value (**-29.6%** on average), $`L_1`$ ERT on **9 of 10** (**-11.1%**), and the SSC gap on **6 of 10** |

---

## How it works

1. Fit the predictor and train a SOM on the training inputs. The map never sees calibration data.
2. Send each calibration point to its BMU cell.
3. For a test point, retrieve the calibration scores in a buffer around its BMU and use their conformal quantile as the threshold.

The buffer defines three regimes. Writing $`k^\ast`$ for the test point's BMU, the paper proves these coverage guarantees:

| Regime | Calibration buffer | Conditioning event | Coverage lower bound | Status |
| --- | --- | --- | --- | --- |
| 1: the BMU cell | $`\mathcal{I}_k`$ | $`k^\ast = k`$ | $`1-\alpha`$ | exact |
| 2: a fixed grid neighborhood | $`\mathcal{I}_{N(k,r)}`$ | $`k^\ast \in N(k,r)`$ | $`1-\alpha`$ | exact |
| | | $`k^\ast = k`$ | $`1-\alpha-\varepsilon_k`$ | approximate |
| | | none | $`1-\alpha-\bar\varepsilon`$ | approximate |
| 3: the neighborhood plus cells in prototype-distance order | $`\mathcal{I}_{R_{L^\star}(k)}`$ | $`k^\ast \in R_{L^\star}(k)`$ | $`1-\alpha`$ | exact |
| | | $`k^\ast = k`$ | $`1-\alpha-\varepsilon_k(R_{L^\star}(k))`$ | approximate |

Here $`\mathcal{I}_A`$ holds the calibration points whose BMU lies in the cell set $`A`$, $`N(k,r)`$ the cells within grid distance $`r`$ of cell $`k`$, and $`R_{L^\star}(k)`$ that neighborhood plus the next $`L^\star`$ cells by prototype distance.  
$`\varepsilon_k(A)`$ is the Kolmogorov-Smirnov bias of cell $`k`$ over $`A`$, with $`\varepsilon_k = \varepsilon_k(N(k,r))`$, and $`\bar\varepsilon`$ averages $`\varepsilon_k`$ over cells, weighted by their mass.  
Rows conditioned on a cell set hold for the threshold of a fixed reference cell $`k`$, and the other rows hold for the deployed threshold.

Regime 3 is for sparse calibration sets. Its global cell budget $`L^\star`$ is chosen from training cell occupancies and the planned calibration size, before any calibration score is seen.

---

## Installation

You need Python 3.11 or newer. We use [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/LouisTier/SOCP.git
cd SOCP
uv sync
```

`pip install -e .` works too. Both install TorchSOM from commit `020ec68` (tag [`socp-neurips2026`](https://github.com/michelin/TorchSOM/tree/socp-neurips2026)), which adds the fixed-cell retrieval used by Regime 3.

---

## Repository layout

```text
socp/conformal/   scores, localizers, calibration mechanisms, and their composition
socp/models/      base predictors needed by the scores
socp/data/        dataset loaders
socp/evaluation/  coverage metrics and paired statistics
experiments/      config.yaml, every pipeline step, and two helper scripts (see its README)
```

---

## Reproducing the experiments

### 1. Check the installation

```bash
make smoke   # Synthetic data through prepare, run, and aggregate, in under a minute.
```

### 2. Get the data

```bash
make download DATASET=all
```

This downloads the nine public datasets. ImageNet-1k can't be downloaded automatically. With access to the ILSVRC2012 validation images and the torchvision ResNet-50 `IMAGENET1K_V2` weights file, build the cache it needs:

```bash
uv run python experiments/scripts/imagenet_cache.py --imagenet-root <root containing val/> \
    --weights resnet50-11ad3fa6.pth --out data/raw/imagenet_val_cache.npz
```

The cache holds the 50,000 labels, the classifier probabilities, and the 2,048-dimensional penultimate representations. The protocol splits it into 8,000 representation-training, 2,000 selection, 20,000 calibration, and 20,000 test rows.

### 3. Run the experiments

```bash
make prepare     # Splits, fitted predictors, and representations    -> data/prepared
make run         # Test predictions and metrics for every method      -> results/runs
make aggregate   # Seed summaries and paired tests                   -> results/aggregate
```

The full profile covers ten datasets, ten seeds, and all 25 regression and 35 classification score and calibrator pairs declared in [`experiments/config.yaml`](experiments/config.yaml).  
On the paper's Intel Xeon Gold 6134 CPUs, the run step took about 31 minutes per regression dataset and seed, and about 6.5 hours per classification dataset and seed.  
[`experiments/scripts/run_parallel.sh`](experiments/scripts/run_parallel.sh) runs the dataset and seed pairs in parallel, one CPU each. To run a single pair:

```bash
make prepare DATASETS=concrete SEEDS=42
make run DATASETS=concrete SEEDS=42
```

Each method writes `predictions.csv` (coverage, size, and cutoff for each test row), `metrics.csv`, and `metadata.json` under `results/runs/<dataset>/seed_<seed>/<score>__<calibrator>/`.  
`make aggregate` then combines them into `results/aggregate/`:

| File | Contents |
| --- | --- |
| `seed_metrics.csv` | The metrics of every method, dataset, and seed |
| `summary.csv` | Their means over seeds |
| `paired_tests.csv` | Wilcoxon tests of SO-SCP against SCP on WCovGap, with Holm's correction across datasets |
| `regime3.csv` | SCP, Regime 2, and Regime 3 on Concrete and Auto MPG, by calibration size |
| `sensitivity.csv` | SO-SCP on California Housing across SOM grid sizes and neighborhood radii |

---

## Methods

| Kind | Method | Reference |
| --- | --- | --- |
| Regression score | Absolute residual | [Lei et al., 2018](https://doi.org/10.1080/01621459.2017.1307116) |
| Regression score | CQR | [Romano et al., 2019](https://proceedings.neurips.cc/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html) |
| Regression score | CHR | [Sesia and Romano, 2021](https://proceedings.neurips.cc/paper/2021/hash/31b3b31a1c2f8a370206f111127c0dbd-Abstract.html) |
| Regression score | NFCP | [Colombo, 2024](https://proceedings.mlr.press/v244/colombo24a.html) |
| Regression score | RCP | [Plassier et al., 2025](https://proceedings.mlr.press/v267/plassier25a.html) |
| Classification score | LAC | [Sadinle et al., 2019](https://doi.org/10.1080/01621459.2017.1395341) |
| Classification score | APS | [Romano et al., 2020](https://proceedings.neurips.cc/paper/2020/hash/244edd7e85dc81602b7615cd705545f5-Abstract.html) |
| Classification score | RAPS | [Angelopoulos et al., 2021](https://arxiv.org/abs/2009.14193) |
| Classification score | SAPS, rank-only | [Huang et al., 2024](https://proceedings.mlr.press/v235/huang24aa.html) |
| Calibration | Pooled SCP | [Vovk et al., 2005](https://doi.org/10.1007/b106715) |
| Calibration | LCP | [Guan, 2023](https://doi.org/10.1093/biomet/asac040) |
| Calibration | RLCP | [Hore and Barber, 2025](https://doi.org/10.1093/jrsssb/qkae103) |
| Calibration | Mondrian on k-means groups | [Vovk et al., 2003](https://alrw.net/old/04.pdf) |
| Calibration | Classwise, Clustered | [Ding et al., 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/cb931eddd563f8d473c355518ce8601c-Abstract-Conference.html) |
| Calibration | **SOCP** | [This paper](https://arxiv.org/abs/2606.29403) |

---

## Citation

```bibtex
@inproceedings{berthier2026socp,
  title     = {Self-Organized Conformal Prediction: Reducing Regional Coverage Gaps with Unsupervised Group Discovery},
  author    = {Berthier, Louis and Shokry, Ahmed and Moreaud, Maxime and Ramelet, Guillaume and Dieuleveut, Aymeric},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  url       = {https://arxiv.org/abs/2606.29403}
}
```

See also [`CITATION.cff`](CITATION.cff).

---

## Acknowledgments

- [Centre de Mathématiques Appliquées (CMAP)](https://cmap.ip-paris.fr/), École Polytechnique
- [Manufacture Française des Pneumatiques Michelin](https://www.michelin.com/)
- Aymeric Dieuleveut is supported by French State aid managed by the Agence Nationale de la Recherche under the France 2030 program, reference ANR-23-IACL-0005 (Hi! PARIS).

---

## License

SOCP is released under the [Apache License 2.0](LICENSE).
