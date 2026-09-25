# QuAIL — Quality-Aware Inertial Learning (federated starting point)

This branch (`quailFed`) is the starting workspace for adapting QuAIL to
**federated learning**, where the quality prior has to act at **aggregation**
time rather than only inside a single centralized training run. It is a
cleaned-up copy of the centralized codebase of the paper: the data-preparation
baselines and curriculum learning are gone, and what is left is the data
tooling, the QuAIL layer and the analysis scripts. The federated part itself is
not implemented yet — see [TODO](#todo).

## What QuAIL is

QuAIL augments a base model with a **learnable feature-modulation (gating)
layer** `g`, applied element-wise to the input (`f(g ⊙ x)`). Each gate is
initialized from a per-feature data-quality score `q ∈ [0, 1]` and its updates
are constrained by a **quality-dependent proximal regularizer**:

- **High-quality features** → a weak regularizer, so the gate is free to adapt.
- **Low-quality features** → a strong regularizer, which anchors the gate close
  to its quality-informed initialization and dampens the feature's influence.

See *"QuAIL: Quality-Aware Inertial Learning for Robust Training under Data
Corruption"* (Sabella & Archetti et al., IJCNN 2026).

## Repository layout

```
quail/                    Core library (details and usage in quail/README.md)
  data.py                 get_datasets(), load_data(): train/val/test splits + per-feature
                          and per-sample quality metadata from the poison masks
  preprocessing.py        TabularPreprocessor (imputation, scaling, one-hot encoding)
  nn.py                   build_model(): linear / MLP / residual / transformer networks
  training.py             fit() and GatedModel — the QuAIL layer and its proximal loss
  optimization.py         OptunaExperiment: centralized HPO over clean / baseline / gate
  comparison_methods.py   Canonical method keys shared by evaluate.py and analysis/
scripts/
  download_data.py        Download the OpenML-CC18 suite into data/
  select_datasets.py      Pick the experiment subset and write it into config.yaml
  poison_data.py          AR / NAR corruption of data/ into data_poisoned/
  evaluate.py             Friedman test + critical-difference diagrams
analysis/                 Post-hoc analysis: Elo ratings, evidence tables, reports
data/                     Clean source datasets (OpenML-CC18), not tracked
config.yaml               Settings for poisoning, selection, HPO and analysis
run_pipeline.sh           Select datasets + poison data per preset (rest is TODO)
main.py                   Entry point of the centralized Optuna reference
POISONING.md              AR/NAR corruption mechanisms and preset rates
```

## What was removed

- **Data-preparation baselines**: CP (`data_preparation_pipeline.py`), Saga++,
  Learn2Clean, DiffPrep, CtxPipe (with its released agent weights), their
  loaders in `quail/data.py`, their flags in `main.py`/`config.yaml` and their
  dependencies (`shap`, `miceforest`, `mlxtend`, `lightgbm`, `transformers`).
- **Curriculum learning** (quality-weighted sampling), alone and combined with
  the quail layer.
- The **notebooks**.

All of it is still in the git history (last commit before the cleanup:
`0dc3a4b`, also on `quail_v2`).

## Setup

- Python 3.12–3.14
- [Poetry](https://python-poetry.org/), or the provided `Dockerfile` /
  `environment.yaml` (a conda env with Poetry inside it)

```bash
cd Repository/FrogDQ.Federated_Proximal_Gating_with_Data_Quality
poetry lock      # the lock file predates the dependency cleanup
poetry install
```

## Data

Clean datasets live under `data/`. To refresh them from OpenML-CC18:

```bash
python scripts/download_data.py --dir data
```

The poisoning step writes, for each dataset, the corrupted train+val split and
its cell-level mask under `data_poisoned/{ar,nar}/`, and the clean hold-out
under `data_poisoned/test/`. The mechanisms and the preset rates (10–40 % of
the feature cells) are documented in [POISONING.md](POISONING.md).

```bash
python scripts/poison_data.py --input_dir data --output_dir data_poisoned --config config.yaml
python scripts/poison_data.py --dataset car          # a single dataset
```

## Pipeline

```bash
./run_pipeline.sh                      # all presets in poisoning.presets.run
./run_pipeline.sh --presets 30         # a single preset
./run_pipeline.sh --skip-select        # keep the datasets already in config.yaml
PYTHON=venv/bin/python ./run_pipeline.sh   # default interpreter is ./.venv/bin/python
```

Step 1 selects the datasets once and rewrites the `datasets:` block of
`config.yaml`; step 2 poisons them for each preset. Steps 3–5 (federated
training and aggregation, evaluation, cleanup) are commented placeholders.
Until they exist, every preset overwrites `data_poisoned/`, so only the last
one is kept.

## Centralized reference

`main.py` still runs the centralized Optuna search the paper used, now reduced
to three methods per dataset: `clean` (train on clean data), `baseline` (train
on AR/NAR data) and `gate` (AR/NAR data + quail layer). It is kept as the
non-federated upper bound to compare against. It reads `data_poisoned/`, so run
the poisoning step first.

```bash
python main.py --datasets car --n-trials 10 --n-seeds 3
python main.py --quick-test --datasets car      # 5 trials, 2 seeds, clean mode only
python main.py --list-datasets
```

Results go to `results/` (`optuna_studies.db`, one CSV per study and
`all_experiments_results.csv`). Study names keep the old
`<dataset>_<mode>_<model>_curr0_gate{0,1}` form, and the CSVs keep constant
`use_curriculum` / `preparation` columns, so the analysis scripts below parse
them unchanged.

## Evaluation and analysis

```bash
python scripts/evaluate.py --config config.yaml --results-dir results --output-dir results/evaluation
cd analysis && python run_quail_evidence_report.py
```

`run_quail_evidence_report.py` chains `optuna_progress.py`,
`quail_analysis.py`, `elo_ratings.py`, `latex_evidence_table.py` and
`elo_baseline_table.py`, reading `../results/optuna_studies.db` and writing to
`analysis/plots/` and `analysis/reports/`. The scripts have not been modified.
`comparison_methods` in `config.yaml` now accepts only `clean`, `baseline` and
`gate`.

## TODO

**Federated setting**

- [ ] **Client partitioning.** Split each dataset's poisoned train split across
      clients, choosing how data and quality are distributed (IID or non-IID
      labels; same or different corruption per client, e.g. different presets
      or different corrupted columns).
- [ ] **Per-client quality.** Compute the per-feature quality `q` on each
      client from its own mask. `load_data` already returns `feature_quality`
      and `sample_quality_*`, which may also serve as client-level signals.
- [ ] **Local training.** Train on each client with the quail layer
      (`quail.training.fit` / `GatedModel`), for a number of local epochs per
      round.
- [ ] **Quality-aware aggregation (the core).** Redesign the QuAIL logic for
      the server: how gates `g` and base weights are aggregated (e.g. per-feature
      weights driven by client quality), what the proximal anchor becomes
      across clients, and whether the regularizer acts on the client, the
      server, or both.
- [ ] **Federated baselines.** Choose and implement the aggregation baselines
      to compare against (e.g. FedAvg, FedProx).
- [ ] **Entry point and configuration.** Add the federated entry point and a
      `federated:` block to `config.yaml` (clients, rounds, local epochs,
      aggregation rule), then fill steps 3–5 of `run_pipeline.sh`.
- [ ] **Results format.** Either write results in the layout the analysis
      scripts expect (Optuna DB with `curr0_gate{0,1}` studies, or the CSV
      columns `evaluate.py` groups by), or adapt the scripts to the new
      methods.

**Known issues left as they are**

- [ ] `scripts/evaluate.py` crashes on the "competitors only" plot with the
      current methods: without `clean` only `baseline` and `gate` remain, but
      the guard at line 198 checks for fewer than 2 methods, while the
      Friedman test needs at least 3. The `with_clean` plot is written before
      the crash. It goes away once a third method (e.g. a federated baseline)
      exists, or by changing the guard to `< 3`.
- [ ] `analysis/` still contains labels, regexes and paths for the removed
      baselines. They are harmless (they simply do not match), but can be
      cleaned up. `poisoning_pct_comparison.py` and `method_time_comparison.py`
      read `results_old/` from the previous baseline runs.
- [ ] `gate_anchor_interval` is still in the search space but has no effect in
      `fit()`: the anchor is fixed before training.
- [ ] `xgboost`, `ucimlrepo` and `tabulate` are declared as dependencies but
      are not imported anywhere.

## License

MIT — see [LICENSE](LICENSE).
