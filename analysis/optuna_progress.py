import argparse
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import math
import pandas as pd
from IPython.display import display

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

plt.rcParams.update({"figure.dpi": 130, "font.size": 10})

DB_PATH = Path("..") / "results" / "optuna_studies.db"
STORAGE  = f"sqlite:///{DB_PATH.resolve()}"
print(f"DB: {DB_PATH.resolve()}  |  exists: {DB_PATH.exists()}")

# Config suffix → human-readable label (shown in legend)
CONFIG_LABELS = {
    "curr0_gate0": "Baseline",
    "curr1_gate0": "+ Curriculum",
    "curr0_gate1": "+ Gate",
    "curr1_gate1": "+ Gate + Curr",
    "ag":          "AutoGluon prep",
    "saga":        "Saga++ prep",
    "cp":          "CP prep",
}

BAR_ORDER = [
    "Clean",
    "Baseline",
    "+ Curriculum",
    "AutoGluon prep",
    "Saga++ prep",
    "CP prep",
    "+ Gate",
    "+ Gate + Curr",
]

BAR_COLORS = {
    "Clean":          "#4c9bcd",
    "Baseline":       "#aaaaaa",
    "+ Curriculum":   "#e07b39",
    "AutoGluon prep": "#9b59b6",
    "Saga++ prep":    "#1abc9c",
    "CP prep":        "#f39c12",
    "+ Gate":         "#e74c3c",
    "+ Gate + Curr":  "#2ecc71",
}

NCOLS = 5  # datasets per row


def f1_studies():
    # Decode a study name into its components
    _RE = re.compile(
        r"^(?P<dataset>.+)_(?P<data_mode>clean|ar|nar)_(?P<model_type>linear|mlp)"
        r"_(?P<config>curr[01]_gate[01]|ag|saga|cp)$"
    )

    rows = []
    for name in optuna.get_all_study_names(storage=STORAGE):
        m = _RE.match(name)
        if not m:
            continue
        study = optuna.load_study(study_name=name, storage=STORAGE)
        try:
            best = study.best_trial
        except Exception:
            continue

        seed_results = best.user_attrs.get("seed_results", [])
        test_f1_values = [
            r["test_f1"] for r in seed_results
            if "test_f1" in r and r["test_f1"] is not None
        ]
        if test_f1_values:
            mean_f1 = float(np.mean(test_f1_values))
            # ddof=1: sample std (unbiased for N-1); falls back to 0 for a single seed
            std_f1  = float(np.std(test_f1_values, ddof=min(1, len(test_f1_values) - 1)))
        else:
            mean_f1 = best.user_attrs.get("avg_test_metric")
            std_f1  = best.user_attrs.get("std_test_metric", 0.0)

        if mean_f1 is None:
            continue

        d = m.groupdict()
        rows.append({
            "dataset":      d["dataset"],
            "data_mode":    d["data_mode"],
            "model_type":   d["model_type"],
            "config":       d["config"],
            "config_label": CONFIG_LABELS.get(d["config"], d["config"]),
            "test_f1_mean": mean_f1,
            "test_f1_std":  std_f1,
            "n_seeds":      len(test_f1_values),
        })

    df = pd.DataFrame(rows)
    print(f"{len(df)} studies with results.")

    return df


def plot_comparison(df: pd.DataFrame, model_type: str, noise_mode: str) -> None:
    """One subplot per dataset, configurations on x-axis, y-axis auto-zoomed."""

    clean_rows = df[
        (df["model_type"] == model_type)
        & (df["data_mode"] == "clean")
        & (df["config"] == "curr0_gate0")
    ].copy()
    clean_rows["config_label"] = "Clean"

    noise_rows = df[
        (df["model_type"] == model_type)
        & (df["data_mode"] == noise_mode)
    ].copy()

    combined = pd.concat([clean_rows, noise_rows], ignore_index=True)
    if combined.empty:
        print(f"No data for {model_type} / {noise_mode}.")
        return

    datasets   = sorted(combined["dataset"].unique())
    n_datasets = len(datasets)

    present_labels = combined["config_label"].unique()
    bar_labels = [b for b in BAR_ORDER if b in present_labels]
    colors     = [BAR_COLORS.get(l, "#cccccc") for l in bar_labels]
    x_pos      = np.arange(len(bar_labels))

    lut   = combined.set_index(["dataset", "config_label"])
    nrows = math.ceil(n_datasets / NCOLS)
    fig, axes = plt.subplots(
        nrows, NCOLS,
        figsize=(NCOLS * 5, nrows * 5),
        squeeze=False,
    )

    for idx, dataset in enumerate(datasets):
        ax = axes[idx // NCOLS][idx % NCOLS]

        means = []
        stds  = []
        for label in bar_labels:
            key = (dataset, label)
            if key in lut.index:
                means.append(lut.loc[key, "test_f1_mean"])
                stds.append(lut.loc[key, "test_f1_std"])
            else:
                means.append(np.nan)
                stds.append(0.0)

        ax.bar(
            x_pos, means, width=0.65,
            yerr=stds, capsize=3,
            color=colors,
            error_kw={"elinewidth": 0.9, "ecolor": "#333"},
        )

        # Annotation: "0.xxxx ± 0.xxxx" above each bar (above the error cap)
        valid_tops = []
        for xi, mean, std in zip(x_pos, means, stds):
            if not np.isnan(mean):
                cap_top = mean + std
                valid_tops.append(cap_top)
                ax.text(
                    xi, cap_top + 0.002,
                    f"{mean:.4f} ± {std:.4f}",
                    ha="center", va="bottom",
                    fontsize=5, rotation=90,
                    color="#222",
                )

        # Auto-zoom: start just below the lowest bar, leave headroom for text
        valid_pairs = [(m, s) for m, s in zip(means, stds) if not np.isnan(m)]
        if valid_pairs:
            y_min_data = min(m - s for m, s in valid_pairs)
            y_max_data = max(m + s for m, s in valid_pairs)
            span       = max(y_max_data - y_min_data, 0.005)
            bottom     = max(0.0, y_min_data - span * 0.15)
            # headroom = space for the rotated annotation text above the cap
            headroom   = max(0.07, span * 1.2)
            ax.set_ylim(bottom, y_max_data + headroom)

        ax.set_title(dataset, fontsize=8, fontweight="bold", pad=4)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bar_labels, rotation=40, ha="right", fontsize=6)
        ax.set_ylabel("Test F1", fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", linewidth=0.3, alpha=0.5)

    # Hide unused axes
    for idx in range(n_datasets, nrows * NCOLS):
        axes[idx // NCOLS][idx % NCOLS].set_visible(False)

    fig.suptitle(
        f"{model_type.upper()} — Clean vs {noise_mode.upper()} configurations",
        fontsize=13, fontweight="bold", y=1.005,
    )
    plt.tight_layout()
    plt.savefig(f'plots/{model_type}_{noise_mode}.png')


def filter_complete_datasets(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only datasets that have all configs completed for each (model_type, data_mode)."""
    parts = []
    for (model_type, data_mode), group in df.groupby(["model_type", "data_mode"]):
        all_configs = set(group["config"].unique())
        complete = (
            group.groupby("dataset")["config"]
            .apply(set)
            .pipe(lambda s: s[s.apply(lambda c: c >= all_configs)].index)
        )
        n_dropped = group["dataset"].nunique() - len(complete)
        if n_dropped:
            print(f"[complete-only] {model_type}/{data_mode}: dropping {n_dropped} incomplete dataset(s)")
        parts.append(group[group["dataset"].isin(complete)])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def overall_aggr_results(df):
    
    # Macro-average F1 across datasets (unweighted) per (model_type, data_mode, config)
    macro_avg = (
        df.groupby(["model_type", "data_mode", "config", "config_label"])["test_f1_mean"]
        .mean()
        .reset_index()
        .rename(columns={"test_f1_mean": "macro_avg_f1"})
    )

    # Pivot so each data_mode becomes its own column
    pivot = (
        macro_avg
        .pivot_table(index=["model_type", "config", "config_label"],
                    columns="data_mode", values="macro_avg_f1")
        .reset_index()
    )
    pivot.columns.name = None

    print(pivot)

    # Macro-median F1 across datasets (unweighted) per (model_type, data_mode, config)
    macro_median = (
        df.groupby(["model_type", "data_mode", "config", "config_label"])["test_f1_mean"]
        .median()
        .reset_index()
        .rename(columns={"test_f1_mean": "macro_median_f1"})
    )

    # Pivot so each data_mode becomes its own column
    pivot = (
        macro_median
        .pivot_table(index=["model_type", "config", "config_label"],
                    columns="data_mode", values="macro_median_f1")
        .reset_index()
    )
    pivot.columns.name = None

    print(pivot)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--complete-only",
        action="store_true",
        help="Restrict aggregated results to datasets that have completed all configurations, "
             "ensuring an equal comparison across configs.",
    )
    args = parser.parse_args()

    df = f1_studies()

    for model_type in sorted(df["model_type"].unique()):
        for noise_mode in ["ar", "nar"]:
            plot_comparison(df, model_type, noise_mode)

    df_agg = filter_complete_datasets(df) if args.complete_only else df
    if args.complete_only:
        print(f"\n[complete-only] {df_agg['dataset'].nunique()} dataset(s) retained for aggregation.\n")
    overall_aggr_results(df_agg)
    