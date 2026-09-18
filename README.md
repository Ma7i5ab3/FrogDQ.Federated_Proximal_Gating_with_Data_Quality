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
data_cleaned_learn2clean/ Learn2Clean-cleaned data (Q-learning pipeline selection)
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
(`run_cp`, `run_saga`). The corruption levels the pipeline sweeps live under
`poisoning.presets` and are documented in [POISONING.md](POISONING.md).
The Learn2Clean baseline is tuned under the `learn2clean` block (goal model,
Q-learning parameters, per-method thresholds).

### 5. Run the full pipeline

```bash
./run_pipeline.sh -y
```

After selecting datasets once, this loops over every corruption preset in
`poisoning.presets.run` (10%, 20%, 30%, 40%). For each preset it poisons the data,
runs data preparation (CP), Saga++ and Learn2Clean, runs the Optuna experiments
with the selected methods (QuAIL and Curriculum included) into
`results/<preset>pct/`, evaluates them, then wipes `data_poisoned/`,
`data_cleaned_cp/`, `data_cleaned_saga/` and `data_cleaned_learn2clean/` before
the next preset.

```bash
./run_pipeline.sh --presets "10,30"                  # only some presets
./run_pipeline.sh --skip-poison --skip-cp            # skip stages in every preset
./run_pipeline.sh --skip-learn2clean                 # skip the Learn2Clean stage
./run_pipeline.sh --no-cleanup                       # keep the intermediate data dirs
./run_pipeline.sh --eval-output-dir my_evaluation    # plot subdir inside each results dir
```

### Data-preparation baselines

| Baseline | Script | Output | Shape |
|---|---|---|---|
| CP | `scripts/data_preparation_pipeline.py` | `data_cleaned_cp/` | preserved |
| Saga++ | `scripts/saga.py` | `data_cleaned_saga/` | preserved |
| Learn2Clean | `scripts/learn2clean.py` | `data_cleaned_learn2clean/` | **reduced** |

Learn2Clean ([Berti-Equille, WWW '19](https://doi.org/10.1145/3308558.3313602),
ported from [the reference implementation](https://github.com/LaureBerti/Learn2Clean))
picks its preparation pipeline by reinforcement learning: Q-learning explores the
state-action graph of the 18 preparation/cleaning methods, then the greedy
traversal from every starting state is executed and the pipeline maximizing the
goal model's quality metric wins. Unlike CP and Saga++ it is *not*
shape-preserving — outlier detection, deduplication and consistency checking drop
rows, feature selection drops columns. The cleaned CSV, its residual mask and the
surviving row/column labels recorded in `<dataset>_pipeline.pkl` are all aligned
to the reduced frame.

Because the frame it produces is smaller, Learn2Clean has its own loader,
`quail.data.load_learn2clean_data`: it reads the reduced train+val frame, reads
the hold-out from `data_cleaned_learn2clean/test/{mode}/` (already projected onto
the surviving columns and rescaled the same way), and aligns the `clean_val`
substitution through the row positions recorded in `<dataset>_pipeline.pkl`
instead of assuming the clean and cleaned partitions have equal length.

Enable it with `run_learn2clean` in [config.yaml](config.yaml) (or
`--run-learn2clean` on `main.py`), and add `learn2clean` to `comparison_methods`
to include it in the evaluation and analysis outputs.
`scripts/learn2clean.py --help` documents the standalone CLI.

Each stage can also be run standalone via the corresponding script in
`scripts/`, e.g. `python scripts/poison_data.py --input_dir data --output_dir
data_poisoned --config config.yaml`.



### 6. Quick smoke test

To sanity-check the setup without a full run:

```bash
python main.py --quick-test --datasets iris
```

### 7. Outputs

- `results/<preset>pct/` — per-trial and top-M metrics (CSV) for that corruption
  level, keyed by dataset / data mode / model type / method, plus the evaluation
  plots under `results/<preset>pct/evaluation/`

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
