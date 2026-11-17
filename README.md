# Uncertainty Quantification in Small-Data Regression

## Overview

This repository accompanies a study of predictive uncertainty in small-data regression. It compares Gaussian-process and neural-network-based uncertainty quantification methods under a reproducible experimental protocol.

The comparison consists of two experiments. The first investigates how increasing training set
size affects predictive accuracy, uncertainty quality, and computational efficiency across three
real-world regression datasets. The second examines how the methods behave under
out-of-distribution (OOD) inputs, focusing on predictive performance and uncertainty calibration
using both synthetic and real-world data.

The experiments compare an exact Gaussian process, three sparse Gaussian-process configurations,
a deep ensemble, two Monte Carlo Dropout variants, and deep kernel learning. All models return a
predictive mean and variance and are evaluated with RMSE, Gaussian negative log-likelihood (NLL),
Gaussian CRPS, and empirical coverage of a nominal 90% prediction interval. For the first
experiment inference and training-wall-clock-time are measured.

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
an estimated time remaining are printed while they run. The UCI and OpenML datasets are downloaded
on first use and cached under `.cache/`, so the first run requires internet access.

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

Each model run is timed. Training time covers construction and fitting until the model-specific
stopping criterion is met. After
fitting, inference time is measured as the median of 20 prediction calls following three untimed
warm-up calls.

The interface also accepts an optional `fitted_model_callback`, which is called with each fitted
model and its training data. Experiment 2 registers this callback for the one-dimensional
synthetic dataset to collect predictions across runs; after the sweep, those predictions are used
to create the model comparison figures. 

Sparse-GP configurations are skipped unless the complete training subset is strictly larger than
the number of inducing points. Their mini-batch size is configured centrally as
`SPARSE_GP_BATCH_SIZE = 32` in `experiments/common.py`.

### Experiment 1: small-data learning curves

Experiment 1 measures how predictive accuracy and uncertainty change as the fitting set grows on
three regression datasets:

| Dataset | Repository | Data ID | Inputs | Split |
| --- | --- | ---: | ---: | --- |
| Combined Cycle Power Plant | UCI | 294 | 4 | 7,654 train-pool / 1,914 test |
| Electrical Grid Stability | UCI | 471 | 11 | 8,000 train-pool / 2,000 test |
| kin8nm | OpenML | 189 | 8 | 6,553 train-pool / 1,639 test |

For Electrical Grid Stability, the continuous `stab` variable is the target and the documented
non-predictive `p1` column is removed. Each dataset is randomly divided 80/20 for each of 20
seeds. Models are fitted on nested subsets of 50, 100, 200, 400, 800, 1600, 3200 observations.
The sweep produces 3120 eligible model fits
after applying the sparse-GP eligibility rule.

The result table contains one row per dataset, seed, fitting-set size, and eligible model
configuration. It records the total and objective-training sizes, a zero validation size, the
number of completed optimization epochs, selected best-checkpoint epoch, GP fitting status,
inducing-point count where applicable,
training and inference time, RMSE, NLL, CRPS, and 90% interval coverage.

### Experiment 2: ID versus OOD uncertainty

Experiment 2 asks whether predictive uncertainty increases appropriately away from the training
distribution. It repeats every configuration over 20 seeds in two settings:

- **Synthetic OOD:** The total label budget contains 100 noisy training observations sampled
  uniformly from `[-3, 3]`, all of which are used by every method. Evaluation uses 500 evenly
  spaced inputs on `[-5, 5]`;
  `[-3, 3]` is labeled ID and the two outer intervals are labeled OOD. The latent function is the
  smooth multi-bump function defined in `experiments/experiment_2.py`, with Gaussian noise of
  standard deviation `0.2`.
- **Power Plant OOD:** standardized input features are projected onto their first principal
  component. The middle 60% of observations along PC1 form the ID pool and the lower and upper
  20% tails form the OOD region. Each seed has a total budget of 400 ID training observations,
  all of which are used by every method. The test set contains all remaining ID observations and
  all OOD observations.

Metrics are reported separately for ID and OOD observations. Experiment 2 also records mean
predictive standard deviation and creates one aggregate synthetic prediction plot per model. In
the one-dimensional synthetic setting, the deep ensemble, MC-dropout, and DKL feature networks
use two hidden layers of width 10. The complete experiment contains 320 model fits.

## Models

Every model exposes the same NumPy-facing prediction interface:

```python
mean, variance = model.predict(X_test)
```

| Model | Implementation | File |
| --- | --- | --- |
| Exact GP | BoTorch `SingleTaskGP` with a constant mean, Matérn-5/2 ARD kernel, learned output scale, and learned Gaussian observation noise. The exact marginal log likelihood is fitted through `fit_gpytorch_mll`. | `models/exact_gp.py` |
| Sparse GP (`m=16,64,256`) | BoTorch `SingleTaskVariationalGP` with learned inducing locations, the same Matérn-5/2 ARD covariance, and `fit_gpytorch_mll_torch` training with Adam, batch size 32, and epoch-level ELBO convergence checks. | `models/sparse_gp.py` |
| Deep ensemble | Five independently initialized, heteroscedastic Gaussian MLPs. Their Gaussian components are combined by moment matching, capturing member disagreement and predicted observation noise. | `models/deep_ensemble.py` |
| MC dropout (heteroscedastic) | A Gaussian MLP with dropout probability `0.1` and an input-dependent variance head; the stochastic Gaussian components are combined by moment matching. | `models/mc_dropout.py` |
| MC dropout (standard) | A mean-only MLP trained with MSE and dropout probability `0.1`; predictive variance is the variance across ten stochastic forward passes. | `models/mc_dropout.py` |
| Deep kernel learning (DKL) | A jointly optimized two-hidden-layer neural feature extractor followed by an exact Matérn-5/2 GP. The marginal log likelihood is optimized through `fit_gpytorch_mll_torch`. | `models/dkl.py` |

### Optimization, stopping & further information

All methods use the complete training subset; no validation observations are held out. Inputs are
standardized by the experiment runner, targets are standardized inside each model, and all model
computations use 64-bit floating-point values. `MAX_TRAINING_EPOCHS` in `experiments/common.py`
provides the common, centrally configurable safety limit (currently 800). For iterative GP
optimizers, this limit refers to optimizer iterations or steps rather than data epochs.

**Exact GP.** The exact marginal log likelihood is fitted with BoTorch's `fit_gpytorch_mll_torch`. The default
torch-based optimizer is configured as Adam with learning rate `0.05`. After a 50-step warm-up, the full-training
negative exact marginal log likelihood is monitored after every optimization step. Training stops
after five consecutive monitored steps without an improvement of at least `1e-4`, subject to the
central safety limit, and the best training-objective checkpoint is restored. The kernel
hyperparameters, constant mean, output scale, and homoscedastic likelihood noise are optimized
jointly.

**Sparse variational GPs.** All three inducing-point configurations optimize the variational ELBO
with `fit_gpytorch_mll_torch`, Adam with learning rate `0.05`, and shuffled mini-batches of size 32.
The inducing locations, variational distribution, kernel parameters, mean, output scale, and
likelihood noise are learned jointly. A custom `EpochELBOStoppingCriterion` collects one loss per
mini-batch and evaluates convergence only after a complete pass through the DataLoader. After a
50-epoch warm-up, it uses the mean mini-batch negative ELBO as the epoch loss and stops after five
consecutive epochs without an improvement of at least `1e-4`. The best training-objective
checkpoint is restored, and training is capped by the central safety limit.

**Deep Ensemble.** Each of the five heteroscedastic members is optimized independently with Adam,
learning rate `0.001`, batch size 32, and Gaussian NLL. After every epoch, that member's Gaussian
NLL is evaluated on the complete training subset once its 50-epoch warm-up is complete. Every
member has its own best checkpoint and patience counter, so members may stop at different epochs.
A member stops after five consecutive monitored epochs without an improvement of at least `1e-4`,
subject to the central safety limit, and its best checkpoint is restored. The result column
`selected_epoch` contains the mean of the five independently selected best epochs. At inference
time, the five Gaussian predictions are combined by moment
matching; predicted member variances represent heteroscedastic aleatoric uncertainty, while
variation between member means represents epistemic uncertainty.

**Heteroscedastic MC Dropout.** The network is optimized with Adam, learning rate `0.001`, batch
size 32, dropout probability `0.1`, and Gaussian NLL. Checkpoint selection begins after a 50-epoch 
warm-up and uses one deterministic `eval()` pass with
dropout disabled over the complete training subset. The monitored objective is Gaussian NLL.
Training uses patience five, `min_delta=1e-4`, and the central safety limit, after which the best
checkpoint is restored. For final prediction, dropout is active again and the ten stochastic
Gaussian components are moment-matched, combining dropout-based epistemic uncertainty with their
input-dependent aleatoric variances.

**Standard MC Dropout.** This variant uses the same Adam settings, mini-batch size, dropout
probability, 50-epoch warm-up, deterministic `eval()` checkpoint selection, patience, and central
safety limit, but has a mean-only output. It is optimized and stopped using MSE rather than NLL,
and the best-MSE checkpoint is restored. At prediction time, dropout is active and its predictive
variance is solely the variance between the ten stochastic mean predictions. It therefore
represents epistemic uncertainty only and does not include an aleatoric observation-noise term.

**Deep kernel learning.** DKL jointly optimizes the neural feature extractor, exact-GP kernel,
constant mean, output scale, and homoscedastic likelihood noise using the exact marginal log
likelihood and `fit_gpytorch_mll_torch`. As for the Exact GP, Adam uses learning rate `0.05`; after
a 50-step warm-up, the full-training negative exact marginal log likelihood is monitored with
`min_delta=1e-4` and patience five, and the best checkpoint is restored. The step limit comes from
the central safety limit. Since the GP following the feature extractor is exact, an individual optimization step still has the
scaling cost of an exact GP; the step limit controls runtime but does not change that asymptotic
cost.

Shared model utilities are implemented in `models/_common.py`, including random seeding, target
standardization, the optional heteroscedastic MLP output, Gaussian NLL, and Gaussian-mixture moment
matching. The Deep Ensemble, MC Dropout, and DKL constructors all accept an optional
`hidden_size`; when it is omitted, the width is `max(2, 2 * input_dim)`. The shared model factories
and hyperparameters are defined in `experiments/common.py`.

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

- `experiments/common.py` defines shared dataset loading and caching, the number of splits, model
  hyperparameters, and model factories used by both experiments.
- `experiments/experiment_1.py` loads and validates the three benchmark datasets, executes the
  learning-curve sweep, and writes the Experiment 1 table.
- `experiments/experiment_2.py` defines the synthetic function and both ID/OOD partitions, reuses
  the model configurations, saves per-seed regional results, and creates synthetic figures.
- `execute_experiment.py` implements the shared nested-subset creation, feature scaling, training,
  timing, callback, and evaluation loop.
- `metrics.py` implements RMSE, Gaussian NLL, Gaussian CRPS, and prediction-interval coverage.
- `models/_common.py` contains reproducibility, target scaling, the shared Gaussian MLP, and
  Gaussian-mixture moment matching. The other files in `models/` implement the five model families.
- `notebooks/` contains result analysis and visualization; `results/` contains generated CSV tables
  and figures.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
