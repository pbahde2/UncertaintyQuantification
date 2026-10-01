# Uncertainty Quantification in Small-Data Regression

This project is an experimental framework for comparing uncertainty-aware regression models when
only a small number of labeled observations are available. It focuses on the regime from a few
dozen to a few hundred training examples, where uncertainty estimates can be as important as point
predictions and where classical Gaussian processes may remain competitive with neural methods.

## Motivation and research questions

Regression models often need to communicate not only what they predict, but also how uncertain
that prediction is. This matters in settings such as scientific modeling, engineering, and
decision-making, where labeled data can be expensive and an overconfident error may be more harmful
than an imprecise prediction.

Gaussian processes (GPs) provide a probabilistic model with principled uncertainty estimates and
are especially attractive for small datasets. Their exact form, however, scales cubically with the
number of training observations. Neural approaches such as deep ensembles and Monte Carlo dropout
scale more naturally to larger datasets, but their uncertainty estimates may be poorly calibrated
when data are scarce. Deep kernel learning (DKL) attempts to combine learned neural features with a
GP, raising the question of whether it preserves GP-like uncertainty behavior.

The intended study therefore asks:

1. At what training-set size does an exact GP stop being the strongest practical baseline?
2. Which method provides the best balance of predictive accuracy and calibrated uncertainty?
3. How much quality is lost when an exact GP is replaced by a sparse approximation?
4. Does DKL behave more like a GP or an overconfident neural model in the small-data regime?
5. How do these conclusions change in interpolation and extrapolation settings?

## Current state of development

The repository contains an executable end-to-end benchmark on a synthetic one-dimensional
regression problem. The data pipeline, shared model interface, repeated experiment loop, metrics,
CSV export, and basic plotting helpers are implemented.

The exact GP is a working NumPy implementation. The sparse GP, deep ensemble, MC dropout, and DKL
classes currently preserve the intended interface but use lightweight proxy behavior instead of
trained implementations. The existing results should therefore be treated as a validation of the
benchmark infrastructure, **not as scientific evidence about which model is best**.

There is one additional calibration limitation: the exact GP currently returns latent-function
uncertainty without adding observation noise to its predictive variance. Its NLL and interval
coverage are therefore not yet valid measures of full predictive calibration.

## Quickstart

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/pbahde2/UncertaintyQuantification.git
cd UncertaintyQuantification

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python experiments/run_small_data_sweep.py
```

The command runs the default synthetic experiment, prints results averaged across splits, and
writes the complete result table to:

```text
results/tables/synthetic_small_data_benchmark.csv
```

The synthetic experiment works without downloading data. Internet access is required only when
the OpenML loader in `data_preparation.py` is used.

## Experimental approach

### Shared evaluation protocol

Every model is evaluated using the same data split, training subset, test observations, and
metrics. This isolates differences between models as far as the current scaffold permits.

The benchmark follows this sequence:

1. Load or generate a regression dataset.
2. Divide it into a training pool and a held-out test set.
3. Fit feature standardization on the training pool and transform both partitions.
4. Select a reproducible subset of the requested training size.
5. Construct each model from the same subset.
6. Predict a mean and standard deviation for every test observation.
7. Calculate accuracy and uncertainty metrics.
8. Repeat the procedure across seeds and training sizes.
9. Store one result row per model, split, and training size.

The default sweep uses:

| Setting | Current value |
| --- | --- |
| Dataset | Synthetic nonlinear one-dimensional regression |
| Total observations per split | 300 |
| Training/test pool | 240/60 observations |
| Training subset sizes | 25, 50, 100, 150 |
| Repeated seeds | 3 |
| Models | 5 |
| Result rows | 60 |

For each seed, the synthetic inputs are sampled uniformly from `[-3, 3]` and targets follow:

```text
y = sin(x) + 0.3x + ε,    ε ~ Normal(0, 0.2²)
```

Different seeds generate different synthetic datasets rather than alternative partitions of one
fixed dataset.

### Models being compared

All models implement the same prediction contract:

```python
mean, std = model.predict(X_test)
```

| Model | Motivation | Current implementation |
| --- | --- | --- |
| Exact GP | Principled small-data probabilistic baseline | Exact posterior equations in NumPy, Matérn-5/2 kernel, heuristic lengthscale, fixed noise |
| Sparse GP | Test whether inducing-point approximations retain GP quality at lower cost | Distance-based proxy; variational inference is not implemented |
| Deep ensemble | Strong practical neural uncertainty baseline based on disagreement between independently trained networks | Stochastic proxy; neural members and heteroscedastic heads are not implemented |
| MC dropout | Inexpensive approximate Bayesian neural baseline using stochastic inference passes | Monte Carlo proxy; a trained dropout network is not implemented |
| DKL | Combine learned neural representations with a probabilistic GP head | Distance-based proxy; the feature extractor and GP head are not implemented |

The exact GP uses the full training covariance matrix, which is appropriate for the small-data
focus but will become expensive as the dataset grows. The common NumPy-array interface keeps the
benchmark independent of a specific training framework. Future neural implementations can create
PyTorch `DataLoader` instances internally, while full-batch GP models can continue to consume the
same prepared arrays directly.

### Metrics

The benchmark records four metrics:

| Metric | What it measures | Preferred value |
| --- | --- | --- |
| RMSE | Accuracy of the predictive mean | Lower |
| Gaussian NLL | Probability assigned to the observed targets using the predicted mean and standard deviation | Lower |
| Gaussian CRPS | Combined probabilistic calibration and sharpness | Lower |
| 90% interval coverage | Fraction of targets inside `mean ± 1.64485 × std` | Close to 0.90 |

RMSE alone cannot determine whether uncertainty estimates are useful. NLL penalizes both inaccurate
means and inappropriate uncertainty, but can become very large for confident errors. CRPS provides
a probabilistic score in the target's units and is generally less dominated by isolated,
overconfident predictions. Coverage checks whether prediction intervals achieve their stated
frequency, although coverage alone does not reward narrow intervals.

Wall-clock runtime is part of the intended comparison but is not yet recorded.

### Result format

The generated CSV contains these columns:

| Column | Meaning |
| --- | --- |
| `dataset` | Dataset identifier |
| `split` | Repetition/seed index |
| `train_size` | Requested number of training observations |
| `model` | Model identifier |
| `rmse` | Root mean squared error |
| `nll` | Mean Gaussian negative log-likelihood |
| `crps` | Mean Gaussian CRPS |
| `coverage_90` | Empirical coverage of the nominal 90% interval |

The command-line script saves all individual rows but prints metric means grouped by model and
training size.

### Plotting

The current plotting helpers display learning curves and mean 90% coverage:

```python
import pandas as pd

from plots import plot_coverage_vs_nominal, plot_learning_curve

results = pd.read_csv("results/tables/synthetic_small_data_benchmark.csv")

plot_learning_curve(results, metric="rmse", dataset="synthetic")
plot_coverage_vs_nominal(results, dataset="synthetic")
```

Plots are shown interactively and are not automatically saved. Calibration curves, confidence
bands across repetitions, and one-dimensional extrapolation visualizations remain future work.

## Using the benchmark with another model or dataset

`benchmark_models` accepts dictionaries of model factories and a custom dataset loader:

```python
from benchmark import benchmark_models
from models import ExactGPModel

results = benchmark_models(
    model_factories={
        "exact_gp": lambda X_train, y_train: ExactGPModel(X_train, y_train),
    },
    dataset_names=["my_dataset"],
    train_sizes=[25, 50, 100],
    n_splits=3,
    dataset_loader=my_dataset_loader,
)
```

A dataset loader must accept a dataset name and `random_seed`, then return:

```python
{
    "X_train": ...,
    "X_test": ...,
    "y_train": ...,
    "y_test": ...,
}
```

Without a custom loader, the benchmark calls `prepare_dataset_for_benchmark`. That function loads
a named OpenML dataset, creates an 80/20 split, and standardizes features using only training-pool
statistics. Targets are currently left on their original scale.

## Project structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── data_preparation.py
├── metrics.py
├── benchmark.py
├── plots.py
├── models/
│   ├── __init__.py
│   ├── exact_gp.py
│   ├── sparse_gp.py
│   ├── deep_ensemble.py
│   ├── mc_dropout.py
│   └── dkl.py
├── experiments/
│   └── run_small_data_sweep.py
└── results/
    └── tables/
        └── synthetic_small_data_benchmark.csv
```

### Root files

- **`README.md`** — explains the research motivation, experimental design, current state, and
  usage of the repository.
- **`LICENSE`** — contains the MIT license under which the project is distributed.
- **`requirements.txt`** — lists runtime and development dependencies. NumPy, pandas,
  scikit-learn, Matplotlib, and seaborn support the current pipeline (SciPy is installed as a
  scikit-learn dependency and is used by the metrics); PyTorch and GPyTorch are included for the
  planned trained neural and GP implementations; JupyterLab supports interactive analysis.
- **`.gitignore`** — excludes virtual environments, Python caches, editor state, build output,
  and other generated local files from version control.
- **`data_preparation.py`** — loads named regression datasets from OpenML, creates reproducible
  train/test splits, standardizes input features, and selects smaller training subsets.
- **`metrics.py`** — implements RMSE, Gaussian NLL, Gaussian CRPS, 90% interval coverage, and a
  convenience function that calculates the complete metric set.
- **`benchmark.py`** — contains the shared experiment orchestration. It loads each dataset,
  iterates over seeds and training sizes, constructs every model, evaluates predictions, and
  returns a pandas DataFrame.
- **`plots.py`** — provides Matplotlib helpers for metric learning curves and comparisons between
  empirical and nominal interval coverage.

### `models/`

The model package defines the common set of uncertainty-aware regressors:

- **`models/__init__.py`** — exposes all five model classes through the `models` package.
- **`models/exact_gp.py`** — implements exact GP prediction with a Matérn-5/2 kernel using NumPy
  linear algebra.
- **`models/sparse_gp.py`** — defines the sparse-GP interface and its current distance-based
  placeholder posterior.
- **`models/deep_ensemble.py`** — defines the ensemble interface and its current stochastic
  prediction proxy.
- **`models/mc_dropout.py`** — defines the MC-dropout interface and generates repeated stochastic
  proxy predictions.
- **`models/dkl.py`** — defines the deep-kernel-learning interface and its current
  distance-dependent proxy uncertainty.

### `experiments/`

- **`experiments/run_small_data_sweep.py`** — executable experiment entry point. It generates
  synthetic data, defines the five model factories, runs the 3-seed × 4-size sweep, saves the
  detailed CSV, and prints averages by model and training size.

### `results/`

This folder holds generated experiment artifacts:

- **`results/tables/`** — stores machine-readable result tables.
- **`results/tables/synthetic_small_data_benchmark.csv`** — current 60-row output of the default
  synthetic sweep.

A dedicated plot-output folder is not currently used because the plotting functions display
figures rather than saving them.

## Known limitations and next steps

The most important next step is replacing the four proxies with trained implementations and
returning observation-inclusive predictive uncertainty from the exact GP. Further planned work
includes:

1. standardizing targets and transforming predictions back to their original scale;
2. fitting preprocessing on each actual training subset rather than the full training pool;
3. creating nested subsets for more stable learning curves;
4. adding curated UCI/OpenML dataset configurations;
5. defining a consistent validation and hyperparameter-tuning protocol;
6. recording training and inference time;
7. reporting aggregate uncertainty across more repeated splits;
8. adding calibration and extrapolation diagnostics; and
9. adding automated tests and explicit reproducibility controls.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
