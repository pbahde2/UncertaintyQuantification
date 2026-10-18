# Uncertainty Quantification in Small-Data Regression

This project is an experimental framework for comparing uncertainty-aware regression models when
only a small number of labeled observations are available. It focuses on the regime from a few
dozen to a few hundred training examples, where uncertainty estimates can be as important as point
predictions and where classical Gaussian processes may remain competitive with neural methods.

## Motivation and research questions



## Quickstart

Python 3.10-3.12 is recommended (PyTorch wheel availability may lag behind new Python releases).

```bash
git clone https://github.com/pbahde2/UncertaintyQuantification.git
cd UncertaintyQuantification

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python experiments/poc_experiment.py
```

The command runs the default synthetic experiment, prints results averaged across splits, and
writes the complete result table to:

```text
results/tables/poc_experiment.csv
```

The synthetic experiment works without downloading data. Internet access is required only when
the OpenML loader in `data_preparation.py` is used.

### Concrete Compressive Strength experiment

The real-data experiment uses the verified OpenML copy (data ID 44959) of the UCI Concrete
Compressive Strength dataset:

```bash
python -u experiments/concrete_experiment.py
```

It evaluates training sizes 50, 100, 200, 400, 800, and 1,000 over 10 reproducible splits,
for a total of 380 eligible model runs. Because the dataset contains 1,030 rows, each split uses exactly
1,000 rows as the nested training pool and 30 held-out rows for testing. Results are written to
`results/tables/concrete_experiment.csv`. The dataset is downloaded once and cached under
`.cache/scikit_learn/`.

### Main experiment

The main experiment compares UCI Combined Cycle Power Plant (ID 294) and Protein Tertiary
Structure (ID 265) using separate reproducible 80/20 train/test splits:

```bash
python -u experiments/power_plant_experiment.py
```

It evaluates training sizes 50, 100, 200, 400, 800, and 1,600 over 20 splits per dataset.
Only sparse-GP configurations with fewer inducing points than training observations are run.
This produces 1,520 eligible runs in one result file:
`results/tables/power_plant_experiment.csv`.

### Power Plant and Electrical Grid Stability experiment

A separate experiment replaces Protein Structure with UCI Electrical Grid Stability (ID 471)
while retaining Power Plant and the same training sizes, splits, models, and sparse-GP eligibility
rule:

```bash
python -u experiments/power_plant_grid_stability_experiment.py
```

Electrical Grid Stability uses the continuous `stab` target and drops the documented
non-predictive `p1` field, leaving 11 numerical inputs. This run writes to
`results/tables/power_plant_grid_stability_experiment.csv`; it does not overwrite the existing
Power Plant/Protein results in `results/tables/power_plant_experiment.csv`.

#### Dataset choice

The two datasets create a controlled contrast between a four-feature engineering problem and a
nine-feature biological problem. Power Plant provides a low-dimensional setting in which exact
GPs should be strong, while Protein Structure tests whether sparse GPs and learned neural features
become more useful as dimensionality and functional complexity increase. Both have numerical
inputs, continuous targets, no missing values, and enough observations for an identical protocol.

## Experimental approach

### Shared evaluation protocol

Every model is evaluated using the same data split, training subset, test observations, and
metrics. This isolates differences between models as far as the current scaffold permits.

The benchmark follows this sequence:

1. Load or generate a regression dataset.
2. Divide it into a training pool and a held-out test set.
3. Generate one reproducibly shuffled training-pool ordering per split.
4. Form every requested training size as a prefix of that ordering, producing nested subsets.
5. Assign stable model-fit and validation roles within each prefix (80%/20% by default).
6. Fit feature standardization only on the model-fit partition and transform validation/test data.
7. Select a training duration using validation NLL and a minimum improvement of `1e-4`.
   Every method uses the same 50-epoch warm-up, patience 5, and 400-epoch ceiling for
   validation-based early stopping.
8. Reinitialize the model and feature scaler, then train on the entire requested subset for the
   selected number of epochs.
9. Predict a mean and variance for every test observation.
10. Calculate accuracy and uncertainty metrics and store one row per model, split, and size.

The default sweep uses:

| Setting | Current value |
| --- | --- |
| Dataset | Synthetic nonlinear one-dimensional regression |
| Total observations per split | 300 |
| Training/test pool | 240/60 observations |
| Training subset sizes | 25, 50, 100, 150 |
| Model-fit/validation fraction | 80%/20% within each training subset |
| Repeated seeds | 3 |
| Model families/configurations | 5/7 |
| Result rows | 84 |

For each seed, the synthetic inputs are sampled uniformly from `[-3, 3]` and targets follow:

```text
y = sin(x) + 0.3x + ε,    ε ~ Normal(0, 0.2²)
```

Different seeds generate different synthetic datasets rather than alternative partitions of one
fixed dataset.

### Models being compared

All models implement the same prediction contract:

```python
mean, variance = model.predict(X_test)
```

| Model | Motivation | Current implementation |
| --- | --- | --- |
| Exact GP | Principled small-data probabilistic baseline | BoTorch `SingleTaskGP` with Matérn-5/2 ARD kernel and learned observation noise |
| Sparse GP | Test whether inducing-point approximations retain GP quality at lower cost | BoTorch `SingleTaskVariationalGP` with learned inducing locations, ELBO training, and `m ∈ {16, 64, 256}` |
| Deep ensemble | Strong practical neural uncertainty baseline based on disagreement between independently trained networks | Five independently trained heteroscedastic Gaussian MLPs |
| MC dropout | Inexpensive approximate Bayesian neural baseline using stochastic inference passes | Heteroscedastic MLP with dropout active over repeated inference passes |
| DKL | Combine learned neural representations with a probabilistic GP head | Jointly trained neural feature extractor and exact Matérn-5/2 GP head |

All methods use float64. Neural models have two equal-width hidden layers, and DKL preserves the
input dimension in its learned feature space.

The exact GP and DKL use the full training covariance matrix, which is appropriate for the
small-data focus but will become expensive as the dataset grows. The public NumPy-array interface
keeps the benchmark independent of the internal PyTorch implementation.

Sparse GPs are evaluated only when the requested training size is strictly larger than the
configuration's inducing-point count. This avoids treating a full-size inducing set as a sparse
approximation and prevents duplicate capped configurations at small training sizes.

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

The benchmark records wall-clock time for final full-data training and test-set inference.

### Result format

The generated CSV contains these columns:

| Column | Meaning |
| --- | --- |
| `dataset` | Dataset identifier |
| `split` | Repetition/seed index |
| `train_size` | Requested number of training observations |
| `model` | Model identifier |
| `selected_epoch` | Validation-selected duration used for full-subset retraining (member mean for ensembles) |
| `n_inducing` | Inducing-point count used for eligible SVGP rows; missing otherwise |
| `final_training_time_seconds` | Model reinitialization and training time on the complete subset; excludes epoch selection and feature scaling |
| `inference_time_seconds` | Predictive-distribution computation time; excludes metric calculation |
| `rmse` | Root mean squared error |
| `nll` | Mean Gaussian negative log-likelihood |
| `crps` | Mean Gaussian CRPS |
| `coverage_90` | Empirical coverage of the nominal 90% interval |

The command-line script saves all individual rows but prints metric means grouped by model and
training size. Timing uses a monotonic high-resolution wall clock. In particular, the reported
training time covers only final full-subset training and never includes validation-based epoch
selection. The proof-of-concept script also prints phase-level progress, elapsed time, and a
rolling ETA after every completed model run; epoch-selection timing is shown only as progress
information and is not stored as training time.

### Plotting

The current plotting helpers display learning curves and mean 90% coverage:

```python
import pandas as pd

from plots import plot_coverage_vs_nominal, plot_learning_curve

results = pd.read_csv("results/tables/poc_experiment.csv")

plot_learning_curve(results, metric="rmse", dataset="synthetic")
plot_coverage_vs_nominal(results, dataset="synthetic")
```

Plots are shown interactively and are not automatically saved. Calibration curves, confidence
bands across repetitions, and one-dimensional extrapolation visualizations remain future work.

## Using the experiment runner with another model or dataset

`execute_experiment` accepts dictionaries of model factories and a custom dataset loader:

```python
from execute_experiment import execute_experiment
from models import ExactGPModel

results = execute_experiment(
    model_factories={
        "exact_gp": lambda X_train, y_train, **kwargs: ExactGPModel(
            X_train, y_train, **kwargs
        ),
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
a named OpenML dataset and creates an 80/20 train/test split. It deliberately returns raw features:
the benchmark fits a new scaler on each model-fit partition, preventing preprocessing leakage from
the validation data, test data, or larger training subsets. Model factories receive `X_val`,
`y_val`, and `random_seed` keyword arguments, so custom factories should accept and forward
`**kwargs`.

Each model standardizes the model-fit targets during epoch selection. Before final evaluation, it
is reinitialized, standardizes all targets in the requested subset, and trains on all of them for
the selected duration. Consequently, `train_size` is the number of observations actually used by
the final model.

## Project structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── data_preparation.py
├── metrics.py
├── execute_experiment.py
├── plots.py
├── models/
│   ├── __init__.py
│   ├── _common.py
│   ├── exact_gp.py
│   ├── sparse_gp.py
│   ├── deep_ensemble.py
│   ├── mc_dropout.py
│   └── dkl.py
├── experiments/
│   └── poc_experiment.py
├── tests/
│   └── test_methods.py
└── results/
    └── tables/
        └── poc_experiment.csv
```

### Root files

- **`README.md`** — explains the research motivation, experimental design, and usage.
- **`LICENSE`** — contains the MIT license under which the project is distributed.
- **`requirements.txt`** — lists runtime and development dependencies. NumPy, SciPy, pandas,
  scikit-learn, Matplotlib, and seaborn support the benchmark and metrics; PyTorch, BoTorch, and
  GPyTorch provide the trained neural and GP implementations; JupyterLab supports interactive
  analysis.
- **`.gitignore`** — excludes virtual environments, Python caches, editor state, build output,
  and other generated local files from version control.
- **`data_preparation.py`** — loads named regression datasets, creates reproducible splits,
  constructs nested subsets, assigns stable validation roles, and standardizes input features.
- **`metrics.py`** — implements RMSE, Gaussian NLL, Gaussian CRPS, and interval coverage.
- **`execute_experiment.py`** — contains the shared experiment orchestration. It loads each dataset,
  iterates over seeds and training sizes, constructs every model, evaluates predictions, and
  returns a pandas DataFrame.
- **`plots.py`** — provides Matplotlib helpers for metric learning curves and comparisons between
  empirical and nominal interval coverage.

### `models/`

The model package defines the common set of uncertainty-aware regressors:

- **`models/__init__.py`** — exposes all five model classes through the `models` package.
- **`models/_common.py`** — contains target scaling, seeding, the shared Gaussian MLP, and its
  early-stopping training loop.
- **`models/exact_gp.py`** — wraps BoTorch's exact `SingleTaskGP` baseline.
- **`models/sparse_gp.py`** — wraps BoTorch's variational `SingleTaskVariationalGP` baseline.
- **`models/deep_ensemble.py`** — trains independent heteroscedastic Gaussian MLP members.
- **`models/mc_dropout.py`** — trains the shared MLP architecture and samples with dropout active.
- **`models/dkl.py`** — jointly trains a dimension-preserving neural feature map and exact GP.

### `experiments/`

- **`experiments/poc_experiment.py`** — executable experiment entry point. It generates
  synthetic data, defines seven configurations across five model families, runs the
  3-seed × 4-size sweep, saves the
  detailed CSV, and prints averages by model and training size.

### `results/`

This folder holds generated experiment artifacts:

- **`results/tables/`** — stores machine-readable result tables.
- **`results/tables/poc_experiment.csv`** — output path for the default 84-row synthetic sweep.

A dedicated plot-output folder is not currently used because the plotting functions display
figures rather than saving them.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
