# `quail/` — core library

The Python package behind QuAIL: data loading with quality metadata,
preprocessing, model construction, training with the QuAIL layer, and the
centralized Optuna experiment. Everything here is centralized (one model per
dataset); the federated training and aggregation will be built on top of these
pieces (see the TODO list in the [top-level README](../README.md#todo)).

Run everything from the repository root, so that `import quail` resolves and
the default `data/` and `data_poisoned/` paths are found.

## Contents

| Module | What it does | Main entry points |
|---|---|---|
| [data.py](data.py) | Lists datasets and loads train/val/test splits together with cell-level masks and per-feature / per-sample quality | `get_datasets`, `load_data` |
| [preprocessing.py](preprocessing.py) | Turns a raw tabular frame into a numeric matrix (type detection, clipping, imputation, scaling, one-hot) | `TabularPreprocessor` |
| [nn.py](nn.py) | Builds linear, MLP, residual or transformer networks for tabular data | `build_model` |
| [training.py](training.py) | Training loop with early stopping and metrics; the QuAIL layer and its proximal regularizer | `fit`, `GatedModel`, `predict`, `predict_proba`, `set_seed` |
| [optimization.py](optimization.py) | Centralized Optuna search over `clean` / `baseline` / `gate`, multi-seed evaluation, results on disk | `OptunaExperiment`, `load_config` |
| [comparison_methods.py](comparison_methods.py) | Canonical method keys shared by `scripts/evaluate.py` and `analysis/` | `METHOD_CHOICES`, `resolve_comparison_methods` |

The typical flow is `load_data` → `build_model` → `fit` (which wraps the model
in `GatedModel` when the quail layer is on) → `predict`. `OptunaExperiment`
runs that flow for every trial and seed.

## Data conventions

`load_data` expects the layout produced by `scripts/download_data.py` and
`scripts/poison_data.py`:

```
data/<idx>_<openml_id>_<name>.csv          clean dataset, e.g. 062_40975_car.csv
data_poisoned/ar/<file>.csv                AR-poisoned train+val split
data_poisoned/ar/<file-stem>_mask.csv      True where a cell was corrupted
data_poisoned/nar/...                      same for NAR
data_poisoned/test/<file>.csv              clean hold-out (test_size of config.yaml)
```

Datasets are referred to by `<name>` (`"car"`, `"credit_g"`, …). Column
prefixes set the column types: `num_` numerical, `cat_` categorical, `dat_`
date, `id_` ignored, `cls_` / `reg_` the classification / regression target.
Targets are never poisoned.

Even `mode="clean"` reads the test set from `data_poisoned/test/`, so run the
poisoning step first. In clean mode the train+val pool excludes the rows held
out as test (`poison_test_size`, which must match `test_size` in
`config.yaml`), so the test set never leaks into training.

## Usage

### Train one model with the QuAIL layer

```python
import numpy as np
from sklearn.preprocessing import LabelEncoder

from quail.data import load_data
from quail.nn import build_model
from quail.training import fit, predict

(X_tr, X_val, X_te), (y_tr, y_val, y_te), prep, meta = load_data("car", mode="ar", seed=42)

# Labels -> 0..K-1 (fit() expects integer class indices)
enc = LabelEncoder().fit(np.concatenate([y_tr, y_val, y_te]))
y_tr, y_val, y_te = enc.transform(y_tr), enc.transform(y_val), enc.transform(y_te)

# Per-feature quality in [0, 1], aligned with the preprocessed columns
q = np.array([meta["feature_quality"].get(f, 100.0) / 100.0
              for f in prep.get_feature_names_out()])

model = build_model(input_dim=X_tr.shape[1], output_dim=len(enc.classes_),
                    hidden_neurons=32, num_layers=2)
model, history = fit(
    model, X_tr, y_tr, X_val, y_val, X_te, y_te,
    epochs=20, learning_rate=1e-3,
    use_gate="true", gate_init="quality", feature_quality=q,
    gate_loss_weight=0.01, gate_quality_weighting="linear",
    verbose=0,
)

best = int(np.argmax(history["val_f1"]))       # the epoch fit() restored
print("test F1:", history["test_f1"][best])
print("gates:", model.gate.detach().numpy())   # one gate per preprocessed feature
y_pred = predict(model, X_te)
```

Without `use_gate="true"`, the same call trains the plain `baseline`.

### Run the centralized Optuna experiment

```python
from quail.optimization import OptunaExperiment, load_config

config = load_config("config.yaml")
config.update({"datasets": ["car"], "n_trials": 10, "n_seeds": 3})
results = OptunaExperiment(config).run_all_experiments()   # also written to results/
```

This is what `main.py` does, plus the CLI overrides and a confirmation prompt.

## Modules in detail

### `data.py`

- **`get_datasets(data_dir="data", poisoned_dir="data_poisoned")`** returns a
  DataFrame with one row per CSV in `data/`: name, OpenML id, number of
  samples, features and classes, task type, and whether AR/NAR versions exist.
  It reads every file, so it is slow on the full suite.
- **`load_data(dataset_name, mode, seed, ...)`** returns
  `(X_train, X_val, X_test), (y_train, y_val, y_test), preprocessor, metadata`.
  The train/val split (`val_size=0.2`) is stratified and redrawn per `seed`.
  The test set is a `test_sample_size=0.8` sample of the clean hold-out.
  `clean_val=True` swaps the validation rows for their clean counterparts. The
  `TabularPreprocessor` is fitted on the training rows only.
- **`metadata`** holds:
  - `mask_train` / `mask_val` / `mask_test`: cell-level masks
  - `feature_quality`: % of clean training cells, keyed by *preprocessed*
    feature name; one-hot columns inherit the quality of their source column
  - `sample_quality_*`: % of clean cells per row
  - `overall_quality`, `task_type`, `n_samples`, `n_features_raw`,
    `n_features_preprocessed`

### `preprocessing.py`

`TabularPreprocessor` detects feature types from the column prefixes, turns
dates into numbers, clips numerical outliers at the 1st/99th percentile, groups
rare categories, imputes (mean / most frequent), standardizes numerical columns
(`scale_numerical=True`) and one-hot encodes categorical ones. It exposes
`fit`, `transform`, `fit_transform` (which also splits train/val/test) and
`get_feature_names_out`. `load_data` uses it with `test_size=0` and
`val_size=0` and does the splitting itself.

### `nn.py`

`build_model(input_dim, output_dim, task="classification", hidden_neurons=128,
layer_type="dense", num_layers=2, dropout=0.0, activation="relu",
use_batch_norm="false")`:

- `hidden_neurons=0` gives a linear model.
- `hidden_neurons="256,128"` sets the layer sizes one by one.
- `layer_type` is `"dense"`, `"residual"` or `"transformer"` (experimental).

### `training.py`

**`fit(model, X_train, y_train, X_val, y_val, X_test=None, y_test=None, ...)`**
trains with the chosen optimizer (`adam`, `adamw`, `sgd`, `rmsprop`), LR
scheduler, early stopping, and optional L1, gradient clipping, class weights,
label smoothing and mixup. It evaluates train/val/test every epoch and
restores the weights of the epoch with the best validation primary metric (F1
for classification, R² for regression). It returns `(model, history)`:
`history` holds per-epoch losses and metrics (`train_f1`, `val_auc`, …) and,
with the quail layer, `gate_weights` and `gate_loss_weight`.

**The QuAIL layer.** With `use_gate="true"`, `fit` wraps the model in
`GatedModel`, which computes `base_model(g ⊙ x)` with one learnable gate per
input feature (`model.gate`).

- **Initialization** (`gate_init`): `quality` sets `g = q` (requires
  `feature_quality`), `ones` sets `g = 1`, `random` sets `g ≈ 1 + noise`.
- **Proximal loss.** Each batch adds `λ(t) · mean(w(q) ⊙ (g − q)²)`. The
  anchor `q` is fixed for the whole run; without `feature_quality`, the anchor
  is the initial `g` and `w = 1`.
- **Weights** (`gate_quality_weighting`), with `u = 1 − q`, so that
  low-quality features are pulled harder towards their anchor:

  | Strategy | `w(q)` |
  |---|---|
  | `linear` | `u` |
  | `quadratic` | `u²` |
  | `exp` | `e^{2u} − 1` |
  | `inv_exp` | `1 − e^{−2u}` |

- **Weight schedule.** `λ` starts at `gate_loss_weight` and follows
  `gate_loss_scheduler`: `none`, `decay`, `increase` or `cosine`.
- **Separate regularization.** The gate sits in its own optimizer parameter
  group, so weight decay and L1 do not act on it.
- `gate_anchor_interval` is accepted but currently has no effect.

`predict(model, X, task)` and `predict_proba(model, X)` run batched inference,
and `set_seed(seed)` seeds Python, NumPy and torch.

### `optimization.py`

`OptunaExperiment(config)` takes the dictionary loaded from
[config.yaml](../config.yaml) (`datasets`, `data_modes`, `model_types`,
`n_trials`, `n_seeds`, `hyperparameter_ranges`, …; `data_dir` and
`poisoned_dir` default to `data` and `data_poisoned`).

- **`run_all_experiments()`** runs one Optuna study per dataset × model type
  for `clean`, and for each AR/NAR mode, both `baseline` and `gate`. Each trial
  is scored by the mean validation metric over `n_seeds` seeds, trained in
  parallel.
- **`reuse_params`**: the `gate` studies start from the top baseline trials of
  the same mode and tune only the quail parameters.
- **`run_final_evaluation(top_k, n_final_seeds, seed_offset)`** re-trains the
  best trials on seeds disjoint from the search.
- **Outputs** go to `output_dir`: studies in `optuna_studies.db` (resumable
  with `warm_start`), one `<study>_results.csv` per study, and
  `all_experiments_results.csv`.
- **Naming**: studies keep the `<dataset>_<mode>_<model>_curr0_gate{0,1}` names
  and the CSVs the constant `use_curriculum` / `preparation` columns, because
  `analysis/` parses them.

### `comparison_methods.py`

`METHOD_CHOICES = ["clean", "baseline", "gate"]` is the method vocabulary. The
`resolve_*` functions map study-name tokens and `evaluate.py` labels onto it.
`resolve_comparison_methods(cli_methods, config_path)` reads
`comparison_methods` from `config.yaml` unless the CLI overrides it. Add a key
here when a new method (e.g. a federated baseline) enters the evaluation.

## Notes for the federated adaptation

- **`GatedModel.state_dict()` keys**: `gate` for the gate vector and
  `base_model.*` for the network, so the two parts can be aggregated
  separately.
- **`fit` is a single-run loop.** Each call builds a new optimizer and early
  stopping state and restores the best validation epoch. With
  `use_gate="true"` it also wraps its input in a **new** `GatedModel`, with the
  gate re-initialized from `q`. Calling it again on an already gated model
  therefore nests two gates, so multi-round local training needs a variant
  that accepts an existing `GatedModel`.
- **Quality comes from the local split.** `load_data` computes
  `feature_quality` and `sample_quality_*` from the training split it loads.
  Once the data are partitioned across clients, the same computation on each
  client's rows gives its local `q`.
