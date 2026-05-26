#!/usr/bin/env python3
"""
Baseline 0: Zero/Random Imputation for ML Datasets

Imputes missing values using the simplest possible strategy:
  - Numerical NaN  → 0
  - Categorical NaN → random value drawn uniformly from observed values in that column

Input:  data_poisoned/{ar,nar}/  (poisoned CSV + mask produced by poison_data.py)
Output: data_baseline_zero/{ar,nar}/   (imputed CSV + residual mask + metrics)

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

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
)


class BaselineZero:
    """
    Zero/random imputation baseline.

    - Numerical NaN  → 0
    - Categorical NaN → random value drawn uniformly from observed column values

    Target columns (cls_*, reg_*) are never modified.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

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
        Apply zero/random imputation to a poisoned dataset.

        Numerical NaN → 0
        Categorical NaN → random value drawn uniformly from observed column values

        Args:
            df_poisoned : Poisoned dataframe loaded from data_poisoned/{mode}/
            mask_df     : Boolean poison mask (True = originally poisoned cell)

        Returns:
            df_cleaned    : Imputed dataframe
            residual_mask : Boolean mask — True for cells that were originally
                            poisoned but could not be resolved (should be empty
                            for this baseline)
            perf_metrics  : Dict with timing and memory measurements
        """
        proc = psutil.Process()
        ram_before_mb = proc.memory_info().rss / 1024 / 1024
        tracemalloc.start()
        wall_t0 = time.perf_counter()
        cpu_t0 = time.process_time()

        col_types = self.identify_column_types(df_poisoned)
        df_clean = df_poisoned.copy()
        n_rows = len(df_clean)

        # Numerical: fill NaN with 0
        for col in col_types["numerical"]:
            n_missing = int(df_clean[col].isna().sum())
            if n_missing > 0:
                if df_clean[col].dtype in ["int64", "int32", "int16", "int8"]:
                    df_clean[col] = df_clean[col].astype("float64")
                df_clean[col] = df_clean[col].fillna(0.0)
                logger.info(f"    {col}: imputed {n_missing} NaN → 0")

        # Categorical: fill NaN with a random observed value
        for col in col_types["categorical"]:
            n_missing = int(df_clean[col].isna().sum())
            if n_missing > 0:
                observed = df_poisoned[col].dropna().unique()
                if len(observed) == 0:
                    logger.warning(f"    {col}: no observed values, skipping imputation")
                    continue
                fill_values = self.rng.choice(observed, size=n_missing)
                na_idx = df_clean.index[df_clean[col].isna()]
                df_clean.loc[na_idx, col] = fill_values
                logger.info(f"    {col}: imputed {n_missing} NaN → random observed value")

        # Residual mask: originally poisoned cells that are still NaN after imputation
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
    output_dir: str = "data_baseline_zero",
    datasets=None,
    seed: int = 42,
):
    """
    Apply zero/random imputation to all poisoned datasets.

    Args:
        input_dir  : Root poisoned-data directory (default: data_poisoned)
        output_dir : Root output directory for imputed data (default: data_baseline_zero)
        datasets   : List of dataset names to process (from config.yaml); None = all
        seed       : Random seed for reproducible categorical sampling (default: 42)
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "ar"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "nar"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)

    preparer = BaselineZero(seed=seed)

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
        logger.info(f"Found {len(csv_files)} poisoned datasets to impute")

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

                # Reinitialise per dataset/mode so the RNG state is consistent
                preparer.rng = np.random.default_rng(seed)
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
                    f"| {perf['wall_time_s']:.3f}s wall | {perf['ram_peak_mb']:.1f} MB peak"
                )

            logger.success(f"  Completed {csv_file.name}")

        except Exception as e:
            logger.error(f"  Error processing {csv_file.name}: {e}")

    logger.success(f"All datasets imputed! Output in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baseline 0: zero/random imputation for AR and NAR poisoned datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python baseline_zero.py
  python baseline_zero.py --input_dir data_poisoned --output_dir data_baseline_zero
  python baseline_zero.py --dataset iris
  python baseline_zero.py --seed 0

Imputation strategy:
  Numerical NaN  → 0
  Categorical NaN → random value drawn uniformly from observed column values

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
        default="data_baseline_zero",
        help="Root directory for imputed output (default: data_baseline_zero)",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible categorical sampling (default: 42)",
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
        seed=args.seed,
    )
