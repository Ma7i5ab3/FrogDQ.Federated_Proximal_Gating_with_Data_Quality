#!/usr/bin/env python3
"""
data_profiling.py — Basic data profiling for the datasets listed in config.yaml.

For each dataset in config.yaml's ``datasets:`` list, loads the raw clean CSV
from ``data/`` and computes basic profiling statistics: size, feature
composition (numerical vs. categorical, using the repo's num_/cat_/cls_/reg_
column-prefix convention — see quail/data.py), missing-value rate, duplicate
row rate, and task-specific stats (number of classes + imbalance ratio for
classification, target mean/std for regression).

Results are saved as a CSV (analysis/reports/dataset_profile.csv) and as a
paper-ready LaTeX table (analysis/reports/dataset_profile_table.tex).

Usage
-----
    python data_profiling.py
    python data_profiling.py --config ../config.yaml --data-dir ../data
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
REPORTS_DIR = HERE / "reports"

NA_VALUES = ["?", "NA", "N/A", "NaN", "nan", "NAN", "", " "]


def load_dataset_names(config_path: str) -> list:
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    datasets = config.get("datasets")
    if not datasets or datasets == "all":
        raise ValueError(
            f"No explicit dataset list found in {config_path} (datasets: {datasets!r}). "
            "This script profiles the datasets named in config.yaml's `datasets:` block."
        )
    return list(datasets)


def find_csv(data_dir: Path, dataset_name: str) -> Path:
    matches = list(data_dir.glob(f"*_{dataset_name}.csv"))
    if not matches:
        raise FileNotFoundError(f"Dataset '{dataset_name}' not found in {data_dir}")
    if len(matches) > 1:
        raise ValueError(f"Multiple files match '{dataset_name}': {[m.name for m in matches]}")
    return matches[0]


def profile_dataset(csv_path: Path, dataset_name: str) -> dict:
    df = pd.read_csv(csv_path, na_values=NA_VALUES)

    parts = csv_path.stem.split("_", 2)
    uci_id = parts[1] if len(parts) >= 2 else "000"

    numerical_cols = [c for c in df.columns if c.startswith("num_")]
    categorical_cols = [c for c in df.columns if c.startswith("cat_")]
    cls_cols = [c for c in df.columns if c.startswith("cls_")]
    reg_cols = [c for c in df.columns if c.startswith("reg_")]
    label_col = (cls_cols + reg_cols)[0]
    feature_cols = numerical_cols + categorical_cols

    n_samples = len(df)
    missing_pct = df[feature_cols].isna().to_numpy().mean() * 100 if feature_cols else 0.0
    duplicate_pct = df.duplicated().mean() * 100

    if cls_cols:
        task_type = "classification"
        class_counts = df[label_col].value_counts()
        n_classes = len(class_counts)
        imbalance_ratio = class_counts.max() / class_counts.min()
        target_mean = np.nan
        target_std = np.nan
    else:
        task_type = "regression"
        n_classes = 0
        imbalance_ratio = np.nan
        target_mean = df[label_col].mean()
        target_std = df[label_col].std()

    return {
        "dataset_name": dataset_name,
        "uci_id": uci_id,
        "task_type": task_type,
        "n_samples": n_samples,
        "n_features": len(feature_cols),
        "n_numerical": len(numerical_cols),
        "n_categorical": len(categorical_cols),
        "n_classes": n_classes,
        "imbalance_ratio": imbalance_ratio,
        "missing_pct": missing_pct,
        "duplicate_pct": duplicate_pct,
        "target_mean": target_mean,
        "target_std": target_std,
    }


def build_latex(profile: pd.DataFrame) -> str:
    lines = [
        "% Requires \\usepackage{booktabs} in the document preamble.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Basic profiling statistics for the datasets used in this study. "
        "\\#Feat.\\ is split into numerical (Num) and categorical (Cat) columns; "
        "\\#Cls.\\ is the number of classes (classification) or `--' (regression); "
        "IR is the class imbalance ratio (majority/minority count); "
        "Miss.\\ and Dup.\\ are the percentage of missing feature cells and "
        "duplicate rows respectively.}",
        "\\label{tab:dataset_profile}",
        "\\begin{tabular}{l r r r r r r r}",
        "\\toprule",
        "Dataset & \\#Inst. & \\#Feat. (Num/Cat) & \\#Cls. & IR & Miss.\\ (\\%) & "
        "Dup.\\ (\\%) & Task \\\\",
        "\\midrule",
    ]

    for _, row in profile.iterrows():
        name = row["dataset_name"].replace("_", "\\_")
        ir = "--" if pd.isna(row["imbalance_ratio"]) else f"{row['imbalance_ratio']:.1f}"
        n_cls = "--" if row["n_classes"] == 0 else f"{int(row['n_classes'])}"
        task = "Class." if row["task_type"] == "classification" else "Reg."
        cells = [
            name,
            f"{int(row['n_samples'])}",
            f"{int(row['n_features'])} ({int(row['n_numerical'])}/{int(row['n_categorical'])})",
            n_cls,
            ir,
            f"{row['missing_pct']:.1f}",
            f"{row['duplicate_pct']:.1f}",
            task,
        ]
        lines.append(" & ".join(cells) + " \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="../config.yaml",
                         help="Path to config.yaml (default: ../config.yaml)")
    parser.add_argument("--data-dir", default="../data",
                         help="Directory containing the clean dataset CSVs (default: ../data)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    dataset_names = load_dataset_names(args.config)
    print(f"Profiling {len(dataset_names)} datasets from {args.config}: {dataset_names}")

    rows = []
    for name in dataset_names:
        try:
            csv_path = find_csv(data_dir, name)
        except (FileNotFoundError, ValueError) as e:
            print(f"  Skipping '{name}': {e}")
            continue
        rows.append(profile_dataset(csv_path, name))
        print(f"  Profiled: {name}")

    profile = pd.DataFrame(rows)

    REPORTS_DIR.mkdir(exist_ok=True)

    csv_out = REPORTS_DIR / "dataset_profile.csv"
    profile.to_csv(csv_out, index=False)
    print(f"\nSaved: {csv_out}")

    latex = build_latex(profile)
    tex_out = REPORTS_DIR / "dataset_profile_table.tex"
    tex_out.write_text(latex)
    print(f"Saved: {tex_out}")

    print("\n" + latex)


if __name__ == "__main__":
    main()
