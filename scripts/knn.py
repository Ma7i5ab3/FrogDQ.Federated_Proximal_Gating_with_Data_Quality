#!/usr/bin/env python3
"""
KNN Imputation Baseline for ML Datasets

Applies K-Nearest Neighbours imputation (sklearn.impute.KNNImputer) to handle
missing values in poisoned tabular datasets.

Categorical columns are ordinal-encoded before imputation and decoded afterwards
(imputed float values are rounded to the nearest valid category index).

Input:  data_poisoned/{ar,nar}/  (poisoned CSV + mask produced by poison_data.py)
Output: data_knn/{ar,nar}/       (imputed CSV + residual mask + metrics)

Note: Target columns (cls_*, reg_*) are never modified.
"""
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Dict, Tuple
import argparse
import yaml

import numpy as np
import pandas as pd
import psutil
from loguru import logger

try:
    from sklearn.impute import KNNImputer
except ImportError as e:
    raise ImportError(
        "scikit-learn is required for KNN imputation. "
        "Install it with: pip install scikit-learn"
    ) from e

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
)


class KNNImputation:
    """
    K-Nearest Neighbours imputation baseline.

    Numerical and categorical features are imputed jointly:
      1. Categorical columns are ordinal-encoded (observed values → integer indices;
         NaN stays as NaN so KNNImputer can handle them).
      2. KNNImputer fills all NaN values using the k nearest neighbours (Euclidean
         distance on the encoded matrix).
      3. Imputed categorical values are rounded to the nearest integer, clipped to
         the valid index range, and decoded back to the original category labels.

    Target columns (cls_*, reg_*) are never modified.
    """

    def __init__(self, n_neighbors: int = 5, seed: int = 42):
        self.n_neighbors = n_neighbors
        self.seed = seed

    def identify_column_types(self, df: pd.DataFrame) -> Dict[str, list]:
        """Identify column types from the prefix naming convention."""
        col_types: Dict[str, list] = {
            "numerical": [],
            "categorical": [],
            "target_cls": [],
            "target_reg": [],
            "date": [],
            "id": [],
        }
        for col in df.columns:
            if col.startswith("num_"):
                col_types["numerical"].append(col)
            elif col.startswith("cat_"):
                col_types["categorical"].append(col)
            elif col.startswith("cls_"):
                col_types["target_cls"].append(col)
            elif col.startswith("reg_"):
                col_types["target_reg"].append(col)
            elif col.startswith("dat_"):
                col_types["date"].append(col)
            elif col.startswith("id_"):
                col_types["id"].append(col)
        return col_types

    def calculate_residual_metrics(self, residual_mask: pd.DataFrame) -> Dict:
        """Calculate cleanliness metrics on the residual mask."""
        col_clean = (1 - residual_mask.sum(axis=0) / len(residual_mask)) * 100
        row_clean = (1 - residual_mask.sum(axis=1) / len(residual_mask.columns)) * 100
        overall_clean = (1 - residual_mask.sum().sum() / residual_mask.size) * 100
        return {
            "column_cleanliness": col_clean.to_dict(),
            "row_cleanliness": row_clean.to_dict(),
            "overall_cleanliness": overall_clean,
        }

    def prepare(
        self,
        df_poisoned: pd.DataFrame,
        mask_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Apply KNN imputation to a poisoned dataset.

        Categorical columns are ordinal-encoded, imputed together with numerical
        columns, and then decoded back to their original label space.

        Args:
            df_poisoned : Poisoned dataframe loaded from data_poisoned/{mode}/
            mask_df     : Boolean poison mask (True = originally poisoned cell)

        Returns:
            df_cleaned    : Imputed dataframe
            residual_mask : Boolean mask — True for cells that were originally
                            poisoned but could not be resolved
            perf_metrics  : Dict with timing and memory measurements
        """
        proc = psutil.Process()
        ram_before_mb = proc.memory_info().rss / 1024 / 1024
        tracemalloc.start()
        wall_t0 = time.perf_counter()
        cpu_t0 = time.process_time()

        col_types = self.identify_column_types(df_poisoned)
        n_rows = len(df_poisoned)

        num_cols = col_types["numerical"]
        cat_cols = col_types["categorical"]
        feature_cols = num_cols + cat_cols

        if not feature_cols:
            logger.warning("  No numerical or categorical feature columns found; skipping imputation")
            df_clean = df_poisoned.copy()
            residual_mask = mask_df & df_clean.isna()
            wall_time_s = time.perf_counter() - wall_t0
            cpu_time_s = time.process_time() - cpu_t0
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            ram_after_mb = proc.memory_info().rss / 1024 / 1024
            ram_peak_mb = peak_bytes / 1024 / 1024
            perf_metrics = {
                "n_rows": n_rows,
                "n_cols": df_clean.shape[1],
                "wall_time_s": round(wall_time_s, 4),
                "cpu_time_s": round(cpu_time_s, 4),
                "ram_before_mb": round(ram_before_mb, 3),
                "ram_after_mb": round(ram_after_mb, 3),
                "ram_peak_mb": round(ram_peak_mb, 3),
                "throughput_rows_per_s": round(n_rows / wall_time_s, 4) if wall_time_s > 0 else float("inf"),
            }
            return df_clean, residual_mask, perf_metrics

        # Build the encoding matrix: numerical as-is, categorical as ordinal integers
        X = pd.DataFrame(index=df_poisoned.index, dtype=float)

        # Copy numerical columns (NaN preserved for KNNImputer)
        for col in num_cols:
            X[col] = df_poisoned[col].astype(float)

        # Ordinal-encode categorical columns: observed value → integer; NaN stays NaN
        cat_mappings: Dict[str, Dict] = {}     # col → {label: int}
        cat_inv_mappings: Dict[str, Dict] = {} # col → {int: label}
        for col in cat_cols:
            observed = sorted(df_poisoned[col].dropna().unique().tolist(), key=str)
            cat_mappings[col] = {v: i for i, v in enumerate(observed)}
            cat_inv_mappings[col] = {i: v for i, v in enumerate(observed)}
            X[col] = df_poisoned[col].map(cat_mappings[col])  # NaN → NaN

        n_missing_before = int(X.isna().sum().sum())
        logger.info(f"  KNN imputing {n_missing_before} missing values across {len(feature_cols)} feature columns")

        # Apply KNN imputation
        # keep_empty_features=True retains all-NaN columns (e.g. num_TBG in sick) in
        # the output as NaN instead of silently dropping them, preventing a shape
        # mismatch when rebuilding the DataFrame with the original feature_cols list.
        imputer = KNNImputer(n_neighbors=min(self.n_neighbors, n_rows - 1), keep_empty_features=True)
        X_arr = imputer.fit_transform(X[feature_cols].values)
        X_imputed = pd.DataFrame(X_arr, columns=feature_cols, index=df_poisoned.index)

        # Decode categorical columns back to original label space
        for col in cat_cols:
            n_cats = len(cat_mappings[col])
            if n_cats == 0:
                logger.warning(f"  {col}: no observed categories, cannot decode")
                continue
            encoded_ints = X_imputed[col].round().clip(0, n_cats - 1).astype(int)
            X_imputed[col] = encoded_ints.map(cat_inv_mappings[col])

        # Reconstruct full dataframe: replace only feature columns
        df_clean = df_poisoned.copy()
        df_clean[feature_cols] = X_imputed[feature_cols]

        # Residual mask: originally poisoned cells that are still NaN after imputation
        residual_mask = mask_df & df_clean.isna()

        n_remaining = int(df_clean[feature_cols].isna().sum().sum())
        if n_remaining > 0:
            logger.warning(f"  {n_remaining} NaN values remain after KNN imputation")

        wall_time_s = time.perf_counter() - wall_t0
        cpu_time_s = time.process_time() - cpu_t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        ram_after_mb = proc.memory_info().rss / 1024 / 1024
        ram_peak_mb = peak_bytes / 1024 / 1024

        perf_metrics = {
            "n_rows": n_rows,
            "n_cols": df_clean.shape[1],
            "wall_time_s": round(wall_time_s, 4),
            "cpu_time_s": round(cpu_time_s, 4),
            "ram_before_mb": round(ram_before_mb, 3),
            "ram_after_mb": round(ram_after_mb, 3),
            "ram_peak_mb": round(ram_peak_mb, 3),
            "throughput_rows_per_s": round(n_rows / wall_time_s, 4) if wall_time_s > 0 else float("inf"),
        }

        return df_clean, residual_mask, perf_metrics


def check_dataset_complete(csv_file: Path, output_dir: str) -> bool:
    """Return True if all four output files for both AR and NAR already exist."""
    expected = [
        os.path.join(output_dir, "ar", csv_file.name),
        os.path.join(output_dir, "ar", csv_file.stem + "_mask.csv"),
        os.path.join(output_dir, "metrics", csv_file.stem + "_ar_metrics.csv"),
        os.path.join(output_dir, "metrics", csv_file.stem + "_ar_perf_metrics.csv"),
        os.path.join(output_dir, "nar", csv_file.name),
        os.path.join(output_dir, "nar", csv_file.stem + "_mask.csv"),
        os.path.join(output_dir, "metrics", csv_file.stem + "_nar_metrics.csv"),
        os.path.join(output_dir, "metrics", csv_file.stem + "_nar_perf_metrics.csv"),
    ]
    return all(os.path.exists(f) for f in expected)


def process_all_datasets(
    input_dir: str = "data_poisoned",
    output_dir: str = "data_knn",
    datasets=None,
    n_neighbors: int = 5,
    seed: int = 42,
):
    """
    Apply KNN imputation to all poisoned datasets.

    Args:
        input_dir    : Root poisoned-data directory (default: data_poisoned)
        output_dir   : Root output directory for imputed data (default: data_knn)
        datasets     : List of dataset names to process (from config.yaml); None = all
        n_neighbors  : Number of nearest neighbours for KNNImputer (default: 5)
        seed         : Kept for API consistency; KNNImputer is deterministic
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "ar"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "nar"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)

    preparer = KNNImputation(n_neighbors=n_neighbors, seed=seed)

    ar_dir = Path(input_dir) / "ar"
    nar_dir = Path(input_dir) / "nar"

    if not ar_dir.exists():
        logger.error(f"AR poisoned directory not found: {ar_dir}")
        return
    if not nar_dir.exists():
        logger.error(f"NAR poisoned directory not found: {nar_dir}")
        return

    csv_files = sorted(ar_dir.glob("*.csv"))
    csv_files = [f for f in csv_files if not f.stem.endswith("_mask")]

    if not csv_files:
        logger.error(f"No poisoned CSV files found in {ar_dir}")
        return

    if datasets:
        dataset_set = set(datasets)
        csv_files = [f for f in csv_files if f.stem[10:] in dataset_set]
        if not csv_files:
            logger.error(f"None of the specified datasets found in {ar_dir}")
            return
        logger.info(f"Processing {len(csv_files)} configured dataset(s)")
    else:
        logger.info(f"Found {len(csv_files)} poisoned datasets to impute (KNN, k={n_neighbors})")

    skipped = sum(1 for f in csv_files if check_dataset_complete(f, output_dir))
    if skipped > 0:
        logger.info(f"Skipped {skipped} already-processed dataset(s)")

    for csv_file in csv_files:
        if check_dataset_complete(csv_file, output_dir):
            logger.info(f"Skipping {csv_file.name} (already complete)")
            continue

        logger.info(f"Processing {csv_file.name}")

        try:
            for mode in ("ar", "nar"):
                mode_dir = Path(input_dir) / mode
                mode_csv = mode_dir / csv_file.name
                mode_mask_file = mode_dir / f"{csv_file.stem}_mask.csv"

                out_files = [
                    os.path.join(output_dir, mode, csv_file.name),
                    os.path.join(output_dir, mode, csv_file.stem + "_mask.csv"),
                    os.path.join(output_dir, "metrics", csv_file.stem + f"_{mode}_metrics.csv"),
                    os.path.join(output_dir, "metrics", csv_file.stem + f"_{mode}_perf_metrics.csv"),
                ]

                if all(os.path.exists(f) for f in out_files):
                    logger.info(f"  {mode.upper()} already processed, skipping: {csv_file.name}")
                    continue
                if not mode_csv.exists():
                    logger.warning(f"  {mode.upper()} CSV not found, skipping: {mode_csv}")
                    continue
                if not mode_mask_file.exists():
                    logger.warning(f"  {mode.upper()} mask not found, skipping: {mode_mask_file}")
                    continue

                df = pd.read_csv(
                    mode_csv, na_values=["?", "NA", "N/A", "NaN", "nan", "NAN", "", " "]
                )
                mask = pd.read_csv(mode_mask_file).astype(bool)
                logger.info(f"  {mode.upper()} loaded: {df.shape[0]} rows × {df.shape[1]} columns")

                df_clean, residual, perf = preparer.prepare(df, mask)
                metrics = preparer.calculate_residual_metrics(residual)

                df_clean.to_csv(os.path.join(output_dir, mode, csv_file.name), index=False)
                residual.to_csv(
                    os.path.join(output_dir, mode, csv_file.stem + "_mask.csv"), index=False
                )
                pd.DataFrame(
                    {
                        "column": list(metrics["column_cleanliness"].keys()),
                        "cleanliness_pct": list(metrics["column_cleanliness"].values()),
                    }
                ).to_csv(
                    os.path.join(output_dir, "metrics", csv_file.stem + f"_{mode}_metrics.csv"),
                    index=False,
                )
                pd.DataFrame([{"dataset": csv_file.stem, "corruption": mode, **perf}]).to_csv(
                    os.path.join(output_dir, "metrics", csv_file.stem + f"_{mode}_perf_metrics.csv"),
                    index=False,
                )
                logger.info(
                    f"  {mode.upper()} imputed: {metrics['overall_cleanliness']:.2f}% clean overall "
                    f"| {perf['wall_time_s']:.1f}s wall | {perf['cpu_time_s']:.1f}s CPU "
                    f"| {perf['ram_peak_mb']:.1f} MB peak"
                )

            logger.success(f"  Completed {csv_file.name}")

        except Exception as e:
            logger.error(f"  Error processing {csv_file.name}: {e}")

    logger.success(f"All datasets imputed! Output in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="KNN imputation baseline for AR and NAR poisoned datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python knn.py
  python knn.py --input_dir data_poisoned --output_dir data_knn
  python knn.py --dataset iris
  python knn.py --n-neighbors 10

Imputation strategy:
  Numerical and categorical columns are imputed jointly via KNNImputer.
  Categoricals are ordinal-encoded before imputation and decoded afterwards.

Note: Target columns (cls_*, reg_*) are NEVER modified.
        """,
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default="data_poisoned",
        help="Root directory of poisoned data (default: data_poisoned)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data_knn",
        help="Root directory for imputed output (default: data_knn)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to experiment config file (default: config.yaml)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Process only this dataset (name without .csv, e.g. 'iris')",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=5,
        help="Number of nearest neighbours for KNNImputer (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (kept for API consistency; default: 42)",
    )

    args = parser.parse_args()

    _config: dict = {}
    if Path(args.config).exists():
        with open(args.config) as _f:
            _config = yaml.safe_load(_f) or {}

    datasets = [args.dataset] if args.dataset else (_config.get("datasets") or None)

    process_all_datasets(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        datasets=datasets,
        n_neighbors=args.n_neighbors,
        seed=args.seed,
    )
