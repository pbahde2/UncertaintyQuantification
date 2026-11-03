# Uncertainty Quantification in Small-Data Regression

## Overview

This repository accompanies a study of predictive uncertainty in small-data regression. It compares Gaussian-process and neural-network-based uncertainty quantification methods under a reproducible experimental protocol.

The comparison consists of two experiments. The first investigates how increasing training set size affects predictive accuracy, uncertainty quality, and computational efficiency across two real-world regression datasets. The second examines how the methods behave under out-of-distribution (OOD) inputs, focusing on predictive performance and uncertainty calibration using both synthetic and real-world data.

The experiments compare an exact Gaussian process, three sparse Gaussian-process configurations,
a deep ensemble, Monte Carlo dropout, and deep kernel learning. All models return a predictive
mean and variance and are evaluated with RMSE, Gaussian negative log-likelihood (NLL), Gaussian
CRPS, and empirical coverage of a nominal 90% prediction interval. For the first experiment inference and training-wall-clock-time are measured.

## Quickstart

Python 3.10--3.12 is recommended. Run all commands from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run Experiment 1:

```bash
python -u experiments/experiment_1.py
```

Run Experiment 2:

```bash
python -u experiments/experiment_2.py
```

Both experiments are full repeated sweeps and can take a long time. Progress, elapsed time, and
an estimated time remaining are printed while they run. The UCI datasets are downloaded on first
use and then cached under `.cache/uci/`, so the first run requires internet access.

### Outputs and analysis

- Experiment 1 saves its per-seed table to `results/tables/experiment_1.csv`.
- Experiment 2 saves per-seed ID/OOD results to `results/tables/experiment_2.csv` and
  synthetic prediction plots to `results/figures/experiment_2/`.
- The main analyses are in `notebooks/analysis_experiment_1.ipynb` and
  `notebooks/analysis_experiment_2.ipynb`.


## Experiment design

### Shared protocol

Both experiments use the common `execute_experiment()` interface implemented in
`execute_experiment.py`. It provides the shared orchestration, training, and evaluation workflow.
Shared protocol parameters and model factories are defined in `experiments/common.py`.

When a validation size is configured, validation-based early stopping is used. After a 50-epoch
warm-up, the checkpoint with the best validation NLL is retained, and training stops after five
consecutive epochs without an improvement of at least `1e-4`. The retained model is evaluated directly without
refitting.

Each model run is timed. Training time covers construction and fitting through selection of the
retained checkpoint. After fitting, inference time is measured as the median of 20 prediction
calls following three untimed warm-up calls. 

The interface also accepts an optional `fitted_model_callback`, which is called with each fitted
model and its training data. Experiment 2 registers this callback for the one-dimensional
synthetic dataset to collect predictions across runs; after the sweep, those predictions are used
to create the model comparison figures. 

Sparse-GP configurations are skipped unless the fitting
set is strictly larger than the number of inducing points.

### Experiment 1: small-data learning curves

Experiment 1 measures how predictive accuracy and uncertainty change as the fitting set grows on
two UCI regression datasets:

| Dataset | UCI ID | Inputs | Split |
| --- | ---: | ---: | --- |
| Combined Cycle Power Plant | 294 | 4 | 7,654 train-pool / 1,914 test |
| Electrical Grid Stability | 471 | 11 | 8,000 train-pool / 2,000 test |

For Electrical Grid Stability, the continuous `stab` variable is the target and the documented
non-predictive `p1` column is removed. Each dataset is randomly divided 80/20 for each of 20
seeds. Models are fitted on nested subsets of 50, 100, 200, 400, 800, and 1,600 observations.
The sweep produces 1,520 eligible model fits
after applying the sparse-GP eligibility rule.

The result table contains one row per dataset, seed, fitting-set size, and eligible model
configuration. It records the selected epoch, inducing-point count where applicable, training and
inference time, RMSE, NLL, CRPS, and 90% interval coverage.

### Experiment 2: ID versus OOD uncertainty

Experiment 2 asks whether predictive uncertainty increases appropriately away from the training
distribution. It repeats every configuration over 20 seeds in two settings:

- **Synthetic OOD:** 300 noisy fitting observations are sampled uniformly from `[-3, 3]`, with
  60 additional validation observations. Evaluation uses 1,000 evenly spaced inputs on `[-5, 5]`;
  `[-3, 3]` is labeled ID and the two outer intervals are labeled OOD. The latent function is the
  smooth multi-bump function defined in `experiments/experiment_2.py`, with Gaussian noise of
  standard deviation `0.2`.
- **Power Plant OOD:** standardized input features are projected onto their first principal
  component. The middle 80% of observations along PC1 form the ID pool and the lower and upper
  10% tails form the OOD region. Each seed uses 400 ID fitting observations and 80 additional ID
  validation observations. The test set contains all remaining ID observations and all OOD
  observations.

Metrics are reported separately for ID and OOD observations. Experiment 2 also records mean
predictive standard deviation and creates one aggregate synthetic prediction plot per model. In
the one-dimensional synthetic setting, the deep ensemble, MC-dropout, and DKL feature networks
use two hidden layers of width 10. The complete experiment contains 280 model fits.

## Models

Every model exposes the same NumPy-facing prediction interface:

```python
mean, variance = model.predict(X_test)
```

| Model | Implementation | File |
| --- | --- | --- |
| Exact GP | BoTorch `SingleTaskGP` with a constant mean, Matérn-5/2 ARD kernel, learned output scale, and learned Gaussian observation noise. Parameters are optimized with the exact marginal log likelihood. | `models/exact_gp.py` |
| Sparse GP (`m=16,64,256`) | BoTorch `SingleTaskVariationalGP` with learned inducing locations, the same Matérn-5/2 ARD covariance, and variational ELBO training. | `models/sparse_gp.py` |
| Deep ensemble | Five independently initialized, heteroscedastic Gaussian MLPs. Their Gaussian components are combined by moment matching, capturing member disagreement and predicted observation noise. | `models/deep_ensemble.py` |
| MC dropout | One heteroscedastic Gaussian MLP with dropout probability `0.1`; five stochastic forward passes are combined by moment matching at prediction time. | `models/mc_dropout.py` |
| Deep kernel learning (DKL) | A jointly optimized two-hidden-layer neural feature extractor followed by an exact Matérn-5/2 GP. The learned feature space preserves the original input dimension. | `models/dkl.py` |





Shared model utilities are implemented in `models/_common.py`, including random seeding, target
standardization, the heteroscedastic Gaussian MLP, Gaussian NLL, and the neural-network training
and early-stopping loop. The Deep Ensemble, MC Dropout, and DKL constructors all accept an optional
`hidden_size`; when it is omitted, the width is `max(2, 2 * input_dim)`. The shared model factories
and hyperparameters are defined in `experiments/common.py`.
All implementations use 64-bit floating-point values.

## Repository structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── execute_experiment.py
├── metrics.py
├── experiments/
│   ├── common.py
│   ├── experiment_1.py
│   └── experiment_2.py
├── models/
│   ├── __init__.py
│   ├── _common.py
│   ├── exact_gp.py
│   ├── sparse_gp.py
│   ├── deep_ensemble.py
│   ├── mc_dropout.py
│   └── dkl.py
├── notebooks/
│   ├── analysis_experiment_1.ipynb
│   └── analysis_experiment_2_new.ipynb
└── results/
    ├── tables/
    └── figures/
```

- `experiments/common.py` defines shared dataset loading and caching, the validation ratio, number
  of splits, model hyperparameters, and model factories used by both experiments.
- `experiments/experiment_1.py` loads and validates the two UCI datasets, executes the
  learning-curve sweep, and writes the Experiment 1 table.
- `experiments/experiment_2.py` defines the synthetic function and both ID/OOD partitions, reuses
  the model configurations, saves per-seed regional results, and creates synthetic figures.
- `execute_experiment.py` implements the shared nested-subset creation, feature scaling, training,
  timing, callback, and evaluation loop.
- `metrics.py` implements RMSE, Gaussian NLL, Gaussian CRPS, and prediction-interval coverage.
- `models/_common.py` contains reproducibility, target scaling, the shared Gaussian MLP, and neural
  early stopping. The other files in `models/` implement the five model families.
- `notebooks/` contains result analysis and visualization; `results/` contains generated CSV tables
  and figures.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
