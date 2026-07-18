# QuAIL — Quality-Aware Inertial Learning

QuAIL is a quality-informed training mechanism for tabular machine learning.
Real-world tabular data is rarely uniformly reliable: individual columns are
often affected by noise, missingness, or systematic bias, and in practice this
is documented only through coarse, column-level reliability indicators (e.g.
provenance, freshness, sensor calibration) rather than per-instance quality
labels. QuAIL turns that column-level prior directly into a learning signal.

It does so by augmenting a base model with a **learnable feature-modulation
(gating) layer** `g`, applied element-wise to the input (`f(g ⊙ x)`). Each
gate is initialized from a per-feature data-quality score `q ∈ [0, 1]` and its
updates are constrained by a **quality-dependent proximal regularizer**:

- **High-quality features** → a weak regularizer, so the gate is free to
  adapt during training.
- **Low-quality features** → a strong regularizer, which anchors the gate
  close to its quality-informed initialization and dampens the feature's
  influence.

This induces controlled, feature-wise adaptation without requiring explicit
data repair, instance-level quality annotations, or sample reweighting.
Empirically, this stabilizes optimization under both random (AR) and
value-dependent (NAR) corruption, with particularly strong gains in
low-data and systematically biased regimes — see the accompanying paper,
*"QuAIL: Quality-Aware Inertial Learning for Robust Training under Data
Corruption"* (Sabella & Archetti et al., IJCNN 2026).

## Repository layout

```
quail/                    Core library: data loading, preprocessing, the QuAIL
                           gating layer, comparison baselines, Optuna training loop
scripts/                  Standalone pipeline steps (see below), one per stage
analysis/                 Post-hoc analysis: Elo ratings, evidence tables, reports
notebooks/                Exploratory notebooks (dataset summaries, walkthroughs)
data/                     Clean source datasets (OpenML-CC18-derived)
data_poisoned/            AR/NAR-corrupted versions of data/, from scripts/poison_data.py
data_cleaned_cp/          Custom-pipeline-cleaned data (MICE + IQR + rule repair)
data_cleaned_saga/        Saga++-cleaned data (baseline data-repair method)
data_autogluon/           Precomputed AutoGluon features (baseline)
config.yaml               Single source of truth for all experiment settings
run_pipeline.sh           Orchestrates the full pipeline end to end
main.py                   Entry point for the Optuna hyperparameter search
POISONING.md              Details of the AR/NAR corruption mechanisms and rates
```

## Reproducing the experiments

### 1. Requirements

- Python 3.12–3.14
- [Poetry](https://python-poetry.org/) for dependency management (or use the
  provided `Dockerfile` / `environment.yaml`, which set up a conda env with
  Poetry inside it)

### 2. Install

```bash
cd Repository/FrogDQ.Federated_Proximal_Gating_with_Data_Quality
poetry install
```

Or, via conda + Docker:

```bash
conda env create -f environment.yaml && conda activate quail
poetry install --no-root
# or: docker build -t quail . && docker run -it quail
```

### 3. Data

Clean datasets are already provided under `data/`. To refresh them from the
OpenML-CC18 suite instead:

```bash
python scripts/download_data.py --output-dir data
```

### 4. Configure

Edit [config.yaml](config.yaml) to pick datasets, data-quality modes
(`clean`/`ar`/`nar`), model types, and which benchmarks to include
(`run_cp`, `run_saga`, `run_autogluon`, `run_catboost`). Corruption rates are
documented in [POISONING.md](POISONING.md) and must be changed accordingly to config.yaml

### 5. Run the full pipeline

```bash
./run_pipeline.sh -y
```

This runs all seven stages in order: select datasets, poison data, data
preparation (CP), Saga++ and Optuna experiments with the selected methods (QuAIL and Curriculum included) (`main.py`)

```bash
./run_pipeline.sh --skip-poison --skip-cp          # skip specific stages
./run_pipeline.sh --start-from 6                    # resume from a given stage
./run_pipeline.sh --eval-output-dir my_evaluation    # custom plot output dir
```

Each stage can also be run standalone via the corresponding script in
`scripts/`, e.g. `python scripts/poison_data.py --input_dir data --output_dir
data_poisoned --config config.yaml`.



### 6. Quick smoke test

To sanity-check the setup without a full run:

```bash
python main.py --quick-test --datasets iris
```

### 7. Outputs

- `results/` — per-trial and top-M metrics (CSV), keyed by dataset / data
  mode / model type / method

### 8. Deep-dive analysis

Once `results/` is populated, the `analysis/` scripts turn the raw metrics
into the statistical evidence and plots used in the paper (Nemenyi/Friedman
significance, Elo ratings, noise-robustness breakdowns, LaTeX tables). Run
the whole suite end to end with:

```bash
python analysis/run_quail_evidence_report.py
```

This chains `optuna_progress.py`, `quail_analysis.py`, `elo_ratings.py`,
`latex_evidence_table.py`, and `elo_baseline_table.py`, writing all plots to
`analysis/plots/` and a single Markdown report to `analysis/reports/`. Each
of these can also be run individually for a narrower output (e.g.
`python analysis/quail_analysis.py --metric accuracy --latex`).

List available datasets at any time with:

```bash
python main.py --list-datasets
```

## License

MIT — see [LICENSE](LICENSE).
