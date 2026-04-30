#!/usr/bin/env python3
import argparse
import os
import sys

import openml
import pandas as pd
from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
)

OPENML_CC18_SUITE_ID = 99


def get_prefix(feature_type, is_target):
    if is_target:
        return "cls_"
    t = str(feature_type).lower()
    if t in ("numeric", "integer"):
        return "num_"
    if t in ("nominal", "string"):
        return "cat_"
    if t == "date":
        return "dat_"
    raise ValueError(f"Unknown feature type: {feature_type}")


def download_all(output_dir, max_rows, max_columns):
    os.makedirs(output_dir, exist_ok=True)

    suite = openml.study.get_suite(OPENML_CC18_SUITE_ID)
    task_ids = suite.tasks
    logger.info(f"OpenML-CC18 suite contains {len(task_ids)} tasks.")

    for index, task_id in enumerate(task_ids, start=1):
        try:
            task = openml.tasks.get_task(task_id)
            dataset = task.get_dataset()
            name = dataset.name.lower().replace(" ", "_").replace("-", "_")
            ds_id = dataset.dataset_id

            out_path = os.path.join(output_dir, f"{index:03d}_{ds_id:05d}_{name}.csv")
            if os.path.exists(out_path):
                logger.info(f"Skipping {name}, already exists.")
                continue

            logger.info(f"Processing {index}/{len(task_ids)}: {name} (dataset_id={ds_id})")

            X, y, _, _ = dataset.get_data(
                target=task.target_name, dataset_format="dataframe"
            )

            df = pd.concat([X, y], axis=1)

            # Filter out datasets that exceed maximum number of rows and columns
            if df.shape[0] > max_rows and df.shape[1] > max_columns:
                continue

            # Build rename map using OpenML feature metadata
            features_meta = {f.name: f for f in dataset.features.values()}
            target_name = task.target_name

            rename_map = {}
            for col in df.columns:
                is_target = col == target_name
                if col in features_meta:
                    ftype = features_meta[col].data_type
                else:
                    # fallback: infer from dtype
                    ftype = "nominal" if df[col].dtype == object else "numeric"
                prefix = get_prefix(ftype, is_target)
                rename_map[col] = f"{prefix}{col}"

            df = df.rename(columns=rename_map)
            df.to_csv(out_path, index=False)
            logger.success(f"Saved {name}")

        except Exception as e:
            logger.error(f"Error on task {task_id}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download all OpenML-CC18 benchmark suite datasets."
    )
    parser.add_argument(
        "--dir", type=str, help="Directory to save CSV files.", default="data", nargs="?"
    )
    parser.add_argument(
        "--max_rows", type=int, help="Filter out datasets that have a number of rows higher 'max_rows'", default=100000000
    )
    parser.add_argument(
        "--max_columns", type=int, help="Filter out datasets that have a number of columns higher 'max_columns'", default=100000000
    )
    args = parser.parse_args()

    download_all(args.dir, args.max_rows, args.max_columns)
