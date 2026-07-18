#!/usr/bin/env python3
"""
AutoGluon data preparation pipeline for Quail benchmarks.

Fits AutoGluon Tabular on poisoned datasets (AR/NAR) tuned for Logistic Regression
(linear) or MLP models, extracts the transformed features, and saves them as
pre-computed NPZ files. These features are then used directly in Optuna experiments
without any further preprocessing.

Both clean and corrupted versions of the validation and test splits are always
saved in each NPZ file, so the caller can choose at load time via the
``clean_val`` / ``clean_test`` flags of ``load_autogluon_data()`` without
having to re-run this script.

One NPZ file is produced per (dataset, mode, model_type, seed), using the same
split logic as ``quail/data.load_data()``:  train+val from ``data_poisoned/``
and test from ``data_poisoned/test/`` (always the clean held-out split).

Usage:
    # Pre-compute features for all datasets using seeds from config.yaml
    python scripts/autogluon.py

    # Pre-compute for specific datasets and modes
    python scripts/autogluon.py --datasets iris wine --data-modes ar nar

    # Override seed start and number of seeds
    python scripts/autogluon.py --seed 42 --n-seeds 5

    # Resume previously interrupted run (already-computed files are skipped)
    python scripts/autogluon.py --resume

Output structure:
    data_autogluon/
      {mode}/               # ar or nar
        {model_type}/       # linear or mlp
          {dataset_name}/
            seed_42.npz     # X_train
            seed_43.npz     # X_val_clean, X_val_corrupted
            ...             # X_test_clean, X_test_corrupted
                            # y_train, y_val, y_test
                            # seed  (the integer seed used)
"""

import argparse
import shutil
import sys
import time
import tracemalloc
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

try:
    from autogluon.tabular import TabularPredictor
except ImportError:
    print(
        "AutoGluon is not installed.\n"
        "Install it with:  pip install autogluon.tabular\n"
        "or:               pip install autogluon"
    )
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading (mirrors quail/data.py split logic for consistency)
# ──────────────────────────────────────────────────────────────────────────────

def _load_raw_splits(
    dataset_name: str,
    mode: str,
    seed: int,
    data_dir: str = "data",
    poisoned_dir: str = "data_poisoned",
    val_size: float = 0.2,
    test_sample_size: float = 0.8,
    poison_test_size: float = 0.3,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
    """Load raw (un-preprocessed) train/val/test splits mirroring load_data() exactly.

    - Train+val pool: from ``poisoned_dir/{mode}/dataset.csv`` (already the 60%
      non-test portion produced by ``poison_data.py``).
    - Test: from ``poisoned_dir/test/dataset.csv`` (always the clean held-out
      split, sampled with ``random_state=seed`` exactly as ``load_data()`` does).
    - Both the corrupted and the clean variants of val are returned so that the
      caller can pass either through AutoGluon and store both in the NPZ.
      For ``mode="clean"`` the two variants are identical.
    - Test is always clean (``data_poisoned/test/`` was split before poisoning),
      so ``df_test_clean`` and ``df_test_corrupted`` are the same object.

    Returns
    -------
    df_train          : poisoned (or clean) training rows
    df_val_clean      : clean validation rows (from original data/)
    df_val_corrupted  : poisoned validation rows
    df_test_clean     : clean held-out test rows (from data_poisoned/test/)
    df_test_corrupted : same as df_test_clean (test is always clean)
    label_col         : name of the target column
    task_type         : 'classification' or 'regression'
    """
    data_path = Path(data_dir)
    poisoned_path = Path(poisoned_dir)

    csv_files = list(data_path.glob(f"*_{dataset_name}.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Dataset '{dataset_name}' not found in {data_dir}")
    if len(csv_files) > 1:
        raise ValueError(f"Multiple files for '{dataset_name}': {[f.name for f in csv_files]}")

    csv_file = csv_files[0]
    dataset_filename = csv_file.name
    na_vals = ["?", "NA", "N/A", "NaN", "nan", "NAN", "", " "]

    # ── Load train+val pool ───────────────────────────────────────────────────
    if mode == "clean":
        full_clean_df = pd.read_csv(csv_file, na_values=na_vals)
        held_out = full_clean_df.sample(frac=poison_test_size, random_state=42)
        poisoned_df = full_clean_df.drop(held_out.index).reset_index(drop=True)
        clean_trainval_df = poisoned_df.copy()
    else:
        poisoned_file = poisoned_path / mode / dataset_filename
        if not poisoned_file.exists():
            raise FileNotFoundError(
                f"Poisoned data not found: {poisoned_file}. "
                "Run scripts/poison_data.py first."
            )
        poisoned_df = pd.read_csv(poisoned_file, na_values=na_vals)
        full_clean_df = pd.read_csv(csv_file, na_values=na_vals)
        held_out = full_clean_df.sample(frac=poison_test_size, random_state=42)
        clean_trainval_df = full_clean_df.drop(held_out.index).reset_index(drop=True)

    # ── Detect label column ───────────────────────────────────────────────────
    label_cols = [c for c in poisoned_df.columns if c.startswith(("cls_", "reg_"))]
    if not label_cols:
        raise ValueError(f"No label column (cls_*/reg_*) found in dataset '{dataset_name}'")
    label_col = label_cols[0]
    task_type = "classification" if label_col.startswith("cls_") else "regression"

    # Drop rows with missing labels before splitting — some datasets (e.g.
    # analcatdata_dmft) have label values like "NA" that are parsed as NaN via
    # na_values.  NaN in the stratify array causes train_test_split to raise
    # "ValueError: Input contains NaN".  poisoned_df and clean_trainval_df are
    # positionally aligned (same 60 % training split), so the same mask applies.
    label_notna = poisoned_df[label_col].notna()
    if not label_notna.all():
        poisoned_df       = poisoned_df[label_notna].reset_index(drop=True)
        clean_trainval_df = clean_trainval_df[label_notna.values].reset_index(drop=True)

    # ── Per-seed train/val split (identical to load_data) ────────────────────
    indices = np.arange(len(poisoned_df))
    y_full = poisoned_df[label_col].values
    stratify = y_full if task_type == "classification" else None
    train_idx, val_idx = train_test_split(
        indices, test_size=val_size, random_state=seed, stratify=stratify
    )

    df_train         = poisoned_df.iloc[train_idx].reset_index(drop=True)
    df_val_corrupted = poisoned_df.iloc[val_idx].reset_index(drop=True)
    df_val_clean     = clean_trainval_df.iloc[val_idx].reset_index(drop=True)

    # ── Test: always from data_poisoned/test/ (always clean) ─────────────────
    test_file = poisoned_path / "test" / dataset_filename
    if not test_file.exists():
        raise FileNotFoundError(
            f"Test file not found: {test_file}. Run scripts/poison_data.py first."
        )
    df_test_full = pd.read_csv(test_file, na_values=na_vals)
    df_test = (
        df_test_full[df_test_full[label_col].notna()]
        .sample(frac=test_sample_size, random_state=seed)
        .reset_index(drop=True)
    )

    return (
        df_train,
        df_val_clean, df_val_corrupted,
        df_test, df_test,  # test is always clean; both variants identical
        label_col, task_type,
    )


# ──────────────────────────────────────────────────────────────────────────────
# NaN imputation for extracted features
# ──────────────────────────────────────────────────────────────────────────────

def _fill_nan_with_train_mean(
    X_train: np.ndarray,
    others: List[np.ndarray],
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Fill NaN values using per-column means computed on X_train."""
    col_means = np.nanmean(X_train, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)  # all-NaN cols → 0

    def _fill(X: np.ndarray) -> np.ndarray:
        nan_mask = np.isnan(X)
        if not nan_mask.any():
            return X
        X = X.copy()
        col_idx = np.where(nan_mask)[1]
        X[nan_mask] = col_means[col_idx]
        return X

    return _fill(X_train), [_fill(X) for X in others]


# ──────────────────────────────────────────────────────────────────────────────
# Core extraction function
# ──────────────────────────────────────────────────────────────────────────────

_AG_LABEL = "__ag_label__"

_MODEL_HYPERPARAMS: Dict[str, Dict] = {
    "linear": {"LR": {}},
    "mlp": {"NN_TORCH": {}},
}


def _to_float32(df: "pd.DataFrame") -> np.ndarray:
    """Convert a DataFrame to a float32 array, preserving NaN."""
    return df.to_numpy(dtype=np.float32, na_value=np.nan)


def extract_features_for_seed(
    dataset_name: str,
    mode: str,
    model_type: str,
    seed: int,
    output_dir: Path,
    data_dir: str = "data",
    poisoned_dir: str = "data_poisoned",
    poison_test_size: float = 0.3,
    time_limit: int = 120,
    verbosity: int = 0,
) -> bool:
    """Fit AutoGluon on one (dataset, mode, model_type, seed) and save features.

    Both clean and corrupted variants of the validation and test splits are
    saved in the NPZ so that ``load_autogluon_data()`` can select either at
    load time via its ``clean_val`` / ``clean_test`` flags.

    Keys saved in the NPZ
    ---------------------
    X_train           – AutoGluon-transformed poisoned training features
    X_val_clean       – AutoGluon-transformed *clean* validation features
    X_val_corrupted   – AutoGluon-transformed *poisoned* validation features
    X_test_clean      – AutoGluon-transformed *clean* test features
    X_test_corrupted  – AutoGluon-transformed *poisoned* test features
    y_train / y_val / y_test – labels (targets are never poisoned)
    task_type         – 'classification' or 'regression'
    feature_names     – column names produced by transform_features()

    Returns True on success, False on failure.  Already-computed seeds are
    skipped automatically.
    """
    out_file = output_dir / mode / model_type / dataset_name / f"seed_{seed}.npz"
    if out_file.exists():
        return True

    # ── 1. Load data ──────────────────────────────────────────────────────────
    try:
        (
            df_train,
            df_val_clean, df_val_corrupted,
            df_test_clean, df_test_corrupted,
            label_col, task_type,
        ) = _load_raw_splits(dataset_name, mode, seed, data_dir, poisoned_dir,
                             poison_test_size=poison_test_size)
    except Exception as exc:
        print(f"      [ERROR] Loading data: {exc}")
        return False

    def _rename(df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns={label_col: _AG_LABEL})

    df_train_ag           = _rename(df_train)
    df_val_clean_ag       = _rename(df_val_clean)
    df_val_corrupted_ag   = _rename(df_val_corrupted)
    df_test_clean_ag      = _rename(df_test_clean)
    df_test_corrupted_ag  = _rename(df_test_corrupted)

    # ── 2. Determine AutoGluon problem type ───────────────────────────────────
    n_classes = df_train_ag[_AG_LABEL].nunique()
    if task_type == "classification":
        problem_type = "binary" if n_classes == 2 else "multiclass"
    else:
        problem_type = "regression"

    # ── 3. Fit AutoGluon predictor (on poisoned training data only) ───────────
    ag_path = output_dir / "_ag_tmp" / f"{dataset_name}_{mode}_{model_type}_s{seed}"

    proc = psutil.Process()
    ram_before_mb = proc.memory_info().rss / 1024 / 1024
    tracemalloc.start()
    wall_t0 = time.perf_counter()
    cpu_t0 = time.process_time()

    try:
        predictor = TabularPredictor(
            label=_AG_LABEL,
            path=str(ag_path),
            verbosity=verbosity,
            problem_type=problem_type,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            predictor.fit(
                train_data=df_train_ag,
                hyperparameters=_MODEL_HYPERPARAMS[model_type],
                time_limit=time_limit,
                num_bag_folds=0,     # disable bagging for simplicity
                num_stack_levels=0,  # disable stacking for simplicity
            )

        # ── 4. Extract transformed features for all splits ────────────────────
        # transform_features() applies AutoGluon's general feature engineering
        # pipeline (imputation, encoding, type coercion) as fitted on the
        # training data.  Both clean and corrupted variants of val/test are
        # passed through the same fitted pipeline.
        def _transform(df: pd.DataFrame) -> np.ndarray:
            return _to_float32(predictor.transform_features(df.drop(columns=[_AG_LABEL])))

        X_train_raw          = _transform(df_train_ag)
        X_val_clean_raw      = _transform(df_val_clean_ag)
        X_val_corrupted_raw  = _transform(df_val_corrupted_ag)
        X_test_clean_raw     = _transform(df_test_clean_ag)
        X_test_corrupted_raw = _transform(df_test_corrupted_ag)

        feature_names = (
            predictor.transform_features(df_train_ag.drop(columns=[_AG_LABEL]))
            .columns.tolist()
        )

        # ── 5. Impute remaining NaN with training column means ─────────────────
        X_train, others = _fill_nan_with_train_mean(
            X_train_raw,
            [X_val_clean_raw, X_val_corrupted_raw,
             X_test_clean_raw, X_test_corrupted_raw],
        )
        X_val_clean, X_val_corrupted, X_test_clean, X_test_corrupted = others

        # ── Collect performance metrics (mirrors saga.py) ──────────────────────
        wall_time_s = time.perf_counter() - wall_t0
        cpu_time_s = time.process_time() - cpu_t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        ram_after_mb = proc.memory_info().rss / 1024 / 1024
        ram_peak_mb = peak_bytes / 1024 / 1024

        n_rows = X_train.shape[0]
        perf_metrics = {
            "n_rows": n_rows,
            "n_cols": len(df_train_ag.columns) - 1,  # exclude label column
            "wall_time_s": round(wall_time_s, 4),
            "cpu_time_s": round(cpu_time_s, 4),
            "ram_before_mb": round(ram_before_mb, 3),
            "ram_after_mb": round(ram_after_mb, 3),
            "ram_peak_mb": round(ram_peak_mb, 3),
            "throughput_rows_per_s": round(n_rows / wall_time_s, 4) if wall_time_s > 0 else float("inf"),
        }

        # ── 6. Labels (targets are never poisoned) ─────────────────────────────
        y_train = df_train[label_col].values
        y_val   = df_val_clean[label_col].values   # identical across variants
        y_test  = df_test_clean[label_col].values

        # ── 7. Save ────────────────────────────────────────────────────────────
        out_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_file,
            X_train=X_train,
            X_val_clean=X_val_clean,
            X_val_corrupted=X_val_corrupted,
            X_test_clean=X_test_clean,
            X_test_corrupted=X_test_corrupted,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            task_type=np.array([task_type]),
            feature_names=np.array(feature_names),
            seed=np.array([seed]),
        )
        perf_file = out_file.with_name("perf_metrics.csv")
        pd.DataFrame([{"dataset": dataset_name, "mode": mode, "model_type": model_type,
                        "seed": seed, **perf_metrics}]).to_csv(perf_file, index=False)
        print(
            f"      Saved {out_file.name}  seed={seed}  "
            f"(train {X_train.shape}, val_clean {X_val_clean.shape}, "
            f"val_corrupted {X_val_corrupted.shape})  "
            f"| {wall_time_s:.1f}s wall | {cpu_time_s:.1f}s CPU | {ram_peak_mb:.1f} MB peak"
        )
        return True

    except Exception as exc:
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        print(f"      [ERROR] AutoGluon: {exc}")
        return False

    finally:
        # Always clean up the AutoGluon model directory to save disk space
        if ag_path.exists():
            shutil.rmtree(ag_path, ignore_errors=True)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset discovery
# ──────────────────────────────────────────────────────────────────────────────

def _discover_datasets(poisoned_dir: str, modes: List[str]) -> List[str]:
    """Return sorted unique dataset names found in poisoned_dir/{mode}/*.csv."""
    base = Path(poisoned_dir)
    names: set = set()
    for mode in modes:
        for csv_file in sorted((base / mode).glob("*.csv")):
            stem = csv_file.stem
            if stem.endswith("_mask"):
                continue
            name = stem.split("_", 1)[-1]
            names.add(name)
    return sorted(names)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-compute AutoGluon features for Quail Optuna benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to experiment config file (default: config.yaml)",
    )
    parser.add_argument(
        "--output-dir", default="data_autogluon",
        help="Directory to save pre-computed features (default: data_autogluon)",
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="Clean data directory (default: data)",
    )
    parser.add_argument(
        "--poisoned-dir", default="data_poisoned",
        help="Poisoned data directory (default: data_poisoned)",
    )
    parser.add_argument(
        "--datasets", nargs="+",
        help="Datasets to process (overrides config)",
    )
    parser.add_argument(
        "--data-modes", nargs="+", choices=["ar", "nar"],
        help="Data modes to process (default: ar nar)",
    )
    parser.add_argument(
        "--model-types", nargs="+", choices=["linear", "mlp"],
        help="Model types to process (default: linear mlp)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Starting seed (overrides config seed_start; default: config value or 42)",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=None,
        help="Number of seeds to generate (overrides config n_seeds; default: config value or 5)",
    )
    parser.add_argument(
        "--test-size", type=float, default=None,
        help="Fraction held out as test set in poison_data.py (overrides config.yaml test_size)",
    )
    parser.add_argument(
        "--time-limit", type=int, default=120,
        help="AutoGluon fitting time limit in seconds per seed (default: 120)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable AutoGluon verbose output",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip already-computed seeds (default behaviour; flag kept for clarity)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Load config (best-effort)
    config: Dict = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}
    else:
        print(f"Config file not found ({args.config}); using defaults / CLI args.")

    # Resolve parameters (CLI overrides config)
    datasets: List[str] = args.datasets or config.get("datasets", [])
    data_modes: List[str] = args.data_modes or [
        m for m in config.get("data_modes", ["ar", "nar"]) if m != "clean"
    ]
    model_types: List[str] = args.model_types or config.get("model_types", ["linear", "mlp"])
    seed_start: int = args.seed if args.seed is not None else config.get("seed_start", 42)
    n_seeds: int = args.n_seeds if args.n_seeds is not None else config.get("n_seeds", 5)
    seeds: List[int] = [seed_start + i for i in range(n_seeds)]
    test_size: float = args.test_size if args.test_size is not None else config.get("test_size", 0.3)

    if not datasets:
        datasets = _discover_datasets(args.poisoned_dir, data_modes)
        if not datasets:
            print(
                f"No datasets found in '{args.poisoned_dir}'. "
                "Use --datasets or populate the poisoned data directory."
            )
            sys.exit(1)
        print(f"Auto-discovered {len(datasets)} dataset(s) from '{args.poisoned_dir}'.")

    if datasets == "all":
        # Lazy import to avoid circular dependency when running standalone
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from quail.data import get_datasets
        datasets = get_datasets(args.data_dir)["dataset_name"].tolist()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    verbosity = 2 if args.verbose else 0

    # ── Summary ───────────────────────────────────────────────────────────────
    total = len(datasets) * len(data_modes) * len(model_types) * len(seeds)
    print(f"\nAutoGluon Feature Extraction")
    print("=" * 60)
    print(f"Datasets    : {datasets}")
    print(f"Data modes  : {data_modes}")
    print(f"Model types : {model_types}")
    print(f"Seeds       : {seeds}")
    print(f"Time limit  : {args.time_limit}s per seed")
    print(f"Output      : {output_dir}/")
    print(f"Total tasks : {total}")
    print(f"Val/test    : both clean and corrupted variants saved per NPZ")
    print(f"              (select at load time via clean_val / clean_test flags)")
    print("=" * 60 + "\n")

    # ── Main loop ─────────────────────────────────────────────────────────────
    n_done = 0
    n_skip = 0
    n_err = 0

    for dataset in datasets:
        for mode in data_modes:
            for model_type in model_types:
                print(f"[{dataset}]  mode={mode}  model={model_type}")
                for seed in seeds:
                    out_file = output_dir / mode / model_type / dataset / f"seed_{seed}.npz"
                    if out_file.exists():
                        print(f"      seed={seed}: already done, skipping")
                        n_skip += 1
                        continue

                    print(f"      seed={seed}: extracting …")
                    ok = extract_features_for_seed(
                        dataset_name=dataset,
                        mode=mode,
                        model_type=model_type,
                        seed=seed,
                        output_dir=output_dir,
                        data_dir=args.data_dir,
                        poisoned_dir=args.poisoned_dir,
                        poison_test_size=test_size,
                        time_limit=args.time_limit,
                        verbosity=verbosity,
                    )
                    if ok:
                        n_done += 1
                    else:
                        n_err += 1

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Completed : {n_done}  |  Skipped : {n_skip}  |  Errors : {n_err}")
    print(f"Output directory: {output_dir}/")
    if n_err > 0:
        print(
            "\nSome extractions failed.  Check error messages above.\n"
            "Re-run the script to retry failed datasets (successful ones are skipped)."
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
