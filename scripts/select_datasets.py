#!/usr/bin/env python3
"""
Dataset subset selection for FrogDQ experiments.

Three configurable, sequential filtering steps:

  1. Size filter   — drop datasets exceeding row / column thresholds.
  2. Family filter — keep one representative per redundant dataset family
                     (the member whose row-count is closest to the family median).
  3. Pilot filter  — if survivors still exceed --pilot-threshold, rank them by
                     clean-vs-poisoned balanced-accuracy gap (logistic-regression
                     proxy) and keep the top --pilot-select.

Run from the project root:
  python scripts/select_datasets.py [options]

Output:
  A YAML ``datasets:`` block printed to stdout, ready to paste into config.yaml.
  Use --output to additionally save it to a file.
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from loguru import logger

# ── project imports ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frogdq.data import get_datasets
from frogdq.preprocessing import TabularPreprocessor
from scripts.poison_data import DataPoisoner

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
)

# ── default redundancy families ───────────────────────────────────────────────
# Each key names a group of structurally similar datasets.
# One member per surviving group is kept (median-row-count representative).
# Override via --families-file (YAML dict mapping name → [dataset, ...]).
DEFAULT_FAMILIES: Dict[str, List[str]] = {
    "mfeat": [
        "mfeat_factors",
        "mfeat_fourier",
        "mfeat_karhunen",
        "mfeat_morphological",
        "mfeat_zernike",
        "mfeat_pixel",
    ],
    "nasa_defect": ["pc1", "pc3", "pc4", "kc1", "kc2", "jm1"],
    "image_derived": [
        "mnist_784",
        "fashion_mnist",
        "cifar_10",
        "devnagari_script",
        "semeion",
    ],
    "analcatdata": ["analcatdata_authorship", "analcatdata_dmft"],
}


# ── Step 1: size filter ───────────────────────────────────────────────────────

def step1_size_filter(df: pd.DataFrame, max_rows: int, max_cols: int) -> pd.DataFrame:
    """Drop datasets with more than *max_rows* samples or *max_cols* features."""
    before = len(df)
    mask = (df["n_samples"] <= max_rows) & (df["n_features"] <= max_cols)
    removed = df[~mask][["dataset_name", "n_samples", "n_features"]]

    result = df[mask].reset_index(drop=True)
    logger.info(
        f"Step 1 — size filter (max_rows={max_rows:,}, max_cols={max_cols:,}): "
        f"{before} → {len(result)} datasets  ({before - len(result)} removed)"
    )
    for _, row in removed.iterrows():
        logger.debug(
            f"  Dropped {row['dataset_name']}  "
            f"({row['n_samples']:,} rows, {row['n_features']} cols)"
        )
    return result


# ── Step 2: family / redundancy filter ───────────────────────────────────────

def step2_family_filter(
    df: pd.DataFrame,
    families: Dict[str, List[str]],
) -> pd.DataFrame:
    """Keep one representative per redundant family (closest to family median rows)."""
    before = len(df)
    drop_names: set = set()

    for family_name, members in families.items():
        surviving = df[df["dataset_name"].isin(members)]
        if len(surviving) <= 1:
            continue

        median_rows = surviving["n_samples"].median()
        keeper_idx = (surviving["n_samples"] - median_rows).abs().idxmin()
        keeper = surviving.loc[keeper_idx, "dataset_name"]
        to_drop = set(surviving["dataset_name"]) - {keeper}
        drop_names |= to_drop
        logger.info(
            f"  Family '{family_name}':  kept '{keeper}',  "
            f"dropped {sorted(to_drop)}"
        )

    result = df[~df["dataset_name"].isin(drop_names)].reset_index(drop=True)
    logger.info(
        f"Step 2 — family filter: {before} → {len(result)} datasets  "
        f"({before - len(result)} removed)"
    )
    return result


# ── Step 3: pilot filter ─────────────────────────────────────────────────────

def _score_pilot(
    X_tr_raw: pd.DataFrame,
    y_tr: np.ndarray,
    X_te_raw: pd.DataFrame,
    y_te: np.ndarray,
    seed: int,
) -> float:
    """
    Fit a logistic-regression proxy on (X_tr_raw, y_tr) and evaluate on
    (X_te_raw, y_te).  Returns balanced accuracy, or NaN on any error.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score

    classes, counts = np.unique(y_tr, return_counts=True)
    if len(classes) < 2 or counts.min() < 3:
        return np.nan

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prep = TabularPreprocessor(random_state=seed, test_size=0.0, val_size=0.0)
            prep.fit(X_tr_raw, y_tr)
            X_tr = prep.transform(X_tr_raw)
            X_te = prep.transform(X_te_raw)

            clf = LogisticRegression(max_iter=300, random_state=seed, n_jobs=1)
            clf.fit(X_tr, y_tr)
            return float(balanced_accuracy_score(y_te, clf.predict(X_te)))
    except Exception:
        return np.nan


def _pilot_gap(
    dataset_name: str,
    data_dir: str,
    poison_test_size: float,
    seed: int,
) -> Tuple[str, float]:
    """
    Compute the performance gap (clean balanced-acc minus min poisoned balanced-acc)
    for one dataset using AR and NAR poisoning.

    Returns (dataset_name, gap).  gap is NaN on error or non-classification datasets.
    """
    # Suppress DataPoisoner INFO logs inside pilot runs
    from loguru import logger as _log
    _log.disable("scripts.poison_data")

    try:
        data_path = Path(data_dir)
        csv_files = sorted(data_path.glob(f"*_{dataset_name}.csv"))
        if not csv_files:
            return dataset_name, np.nan

        df = pd.read_csv(
            csv_files[0],
            na_values=["?", "NA", "N/A", "NaN", "nan", "NAN", "", " "],
        )

        label_cols = [c for c in df.columns if c.startswith("cls_")]
        if not label_cols:
            return dataset_name, np.nan
        label_col = label_cols[0]

        # Hold out test split with the same fixed seed used by poison_data.py
        df_test = df.sample(frac=poison_test_size, random_state=42)
        df_train = df.drop(df_test.index).reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)

        y_train = df_train[label_col].values
        y_test = df_test[label_col].values
        X_train_raw = df_train.drop(columns=[label_col])
        X_test_raw = df_test.drop(columns=[label_col])

        clean_acc = _score_pilot(X_train_raw, y_train, X_test_raw, y_test, seed)
        if np.isnan(clean_acc):
            return dataset_name, np.nan

        poisoner = DataPoisoner(seed=seed)

        df_ar, _ = poisoner.poison_ar(df_train)
        ar_acc = _score_pilot(
            df_ar.drop(columns=[label_col]), y_train, X_test_raw, y_test, seed
        )

        df_nar, _ = poisoner.poison_nar(df_train)
        nar_acc = _score_pilot(
            df_nar.drop(columns=[label_col]), y_train, X_test_raw, y_test, seed
        )

        poisoned_best = float(np.nanmin([ar_acc, nar_acc]))
        if np.isnan(poisoned_best):
            return dataset_name, np.nan

        gap = clean_acc - poisoned_best
        return dataset_name, float(gap)

    except Exception as e:
        logger.warning(f"  [{dataset_name}] Pilot error: {e}")
        return dataset_name, np.nan
    finally:
        _log.enable("scripts.poison_data")


def step3_pilot_filter(
    df: pd.DataFrame,
    data_dir: str,
    n_select: int,
    poison_test_size: float,
    seed: int,
    n_jobs: int,
) -> pd.DataFrame:
    """
    Rank datasets by clean-vs-poisoned balanced-accuracy gap.
    Keeps the top *n_select* most discriminative datasets.
    Datasets where the pilot fails (NaN) are placed last.
    """
    n_select = min(n_select, len(df))
    logger.info(
        f"Step 3 — pilot filter: evaluating {len(df)} datasets "
        f"(n_jobs={n_jobs}), keeping top {n_select}…"
    )

    names = df["dataset_name"].tolist()
    results: List[Tuple[str, float]]

    try:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(_pilot_gap)(name, data_dir, poison_test_size, seed)
            for name in names
        )
    except ImportError:
        logger.warning("joblib not found — running pilot sequentially.")
        results = [_pilot_gap(name, data_dir, poison_test_size, seed) for name in names]

    gap_map = {name: gap for name, gap in results}
    df = df.copy()
    df["pilot_gap"] = df["dataset_name"].map(gap_map)

    valid = df[df["pilot_gap"].notna()].sort_values("pilot_gap", ascending=False)
    invalid = df[df["pilot_gap"].isna()]
    ranked = pd.concat([valid, invalid], ignore_index=True)

    # Summary table
    col_w = max(len(n) for n in names) + 2
    header = f"{'Dataset':<{col_w}} {'Rows':>7}  {'Gap':>8}"
    logger.info(f"\n{header}\n{'─' * len(header)}")
    for _, row in ranked.iterrows():
        gap_str = f"{row['pilot_gap']:+.4f}" if pd.notna(row["pilot_gap"]) else "   N/A "
        logger.info(f"  {row['dataset_name']:<{col_w - 2}} {int(row['n_samples']):>7}  {gap_str:>8}")

    result = ranked.head(n_select).reset_index(drop=True)
    logger.info(
        f"Step 3 — pilot filter: {len(df)} → {len(result)} datasets selected"
    )
    return result


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a meaningful dataset subset for FrogDQ experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="Directory containing the clean CSV files.",
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Experiment config file (read for test_size).",
    )
    parser.add_argument(
        "--output", default=None, metavar="FILE",
        help="Write the final YAML dataset list to this file.",
    )

    # ── step 1 ──
    g1 = parser.add_argument_group("Step 1 — size filter")
    g1.add_argument(
        "--max-rows", type=int, default=100_000,
        help="Drop datasets with more rows than this.",
    )
    g1.add_argument(
        "--max-cols", type=int, default=500,
        help="Drop datasets with more features than this (label excluded).",
    )

    # ── step 2 ──
    g2 = parser.add_argument_group("Step 2 — redundancy / family filter")
    g2.add_argument(
        "--no-family-filter", action="store_true",
        help="Skip the family deduplication step.",
    )
    g2.add_argument(
        "--families-file", default=None, metavar="FILE",
        help=(
            "YAML file with custom families "
            "(dict mapping family_name → [dataset_name, ...])."
        ),
    )

    # ── step 3 ──
    g3 = parser.add_argument_group("Step 3 — pilot filter")
    g3.add_argument(
        "--pilot-threshold", type=int, default=20,
        help="Run pilot only when surviving datasets exceed this count.",
    )
    g3.add_argument(
        "--pilot-select", type=int, default=20,
        help="Number of datasets to retain after the pilot.",
    )
    g3.add_argument(
        "--no-pilot", action="store_true",
        help="Skip the pilot step entirely.",
    )
    g3.add_argument("--pilot-seed", type=int, default=42)
    g3.add_argument(
        "--pilot-jobs", type=int, default=4,
        help="Parallel workers for the pilot (requires joblib).",
    )

    args = parser.parse_args()

    # ── read config ──────────────────────────────────────────────────────────
    cfg: dict = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
    poison_test_size: float = cfg.get("test_size", 0.3)

    # ── load families ─────────────────────────────────────────────────────────
    families = DEFAULT_FAMILIES
    if args.families_file:
        with open(args.families_file) as f:
            families = yaml.safe_load(f)

    # ── catalogue ─────────────────────────────────────────────────────────────
    logger.info(f"Loading dataset catalogue from '{args.data_dir}'…")
    try:
        df = get_datasets(data_dir=args.data_dir)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    logger.info(f"Found {len(df)} datasets.")

    # ── step 1 ───────────────────────────────────────────────────────────────
    df = step1_size_filter(df, max_rows=args.max_rows, max_cols=args.max_cols)
    if df.empty:
        logger.error(
            "No datasets survive the size filter. "
            "Relax --max-rows / --max-cols and retry."
        )
        sys.exit(1)

    # ── step 2 ───────────────────────────────────────────────────────────────
    if not args.no_family_filter:
        df = step2_family_filter(df, families)

    # ── step 3 ───────────────────────────────────────────────────────────────
    if args.no_pilot:
        logger.info("Step 3 — pilot skipped (--no-pilot).")
    elif len(df) <= args.pilot_threshold:
        logger.info(
            f"Step 3 — pilot skipped: {len(df)} datasets ≤ "
            f"threshold {args.pilot_threshold}."
        )
    else:
        logger.info(
            f"{len(df)} datasets remain (> threshold {args.pilot_threshold}): "
            "running pilot…"
        )
        df = step3_pilot_filter(
            df,
            data_dir=args.data_dir,
            n_select=args.pilot_select,
            poison_test_size=poison_test_size,
            seed=args.pilot_seed,
            n_jobs=args.pilot_jobs,
        )

    # ── output ────────────────────────────────────────────────────────────────
    final_names = df["dataset_name"].tolist()

    logger.success(
        f"Final subset: {len(final_names)} datasets — "
        + ", ".join(final_names)
    )

    yaml_block = "datasets:\n" + "".join(f"  - {n}\n" for n in final_names)
    print()
    print("# ── paste into config.yaml ─────────────────────────────────────────────")
    print(yaml_block)

    if args.output:
        import re
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and out_path.suffix in (".yaml", ".yml"):
            content = out_path.read_text()
            pattern = r'^datasets:\n(?:[ \t]+-[ \t]+[^\n]+\n)+'
            if re.search(pattern, content, flags=re.MULTILINE):
                content = re.sub(pattern, yaml_block, content, flags=re.MULTILINE)
            else:
                content += "\n" + yaml_block
            out_path.write_text(content)
        else:
            with open(out_path, "w") as f:
                f.write(yaml_block)
        logger.info(f"Written to '{out_path}'.")


if __name__ == "__main__":
    main()
