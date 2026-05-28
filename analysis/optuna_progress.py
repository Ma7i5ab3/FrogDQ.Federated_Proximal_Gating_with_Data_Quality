#!/usr/bin/env python3
"""
optuna_progress.py — Aggregate and visualise Optuna study results.

CLI arguments
-------------
--split {test,val}
    Evaluation split to use for all metric extraction (default: test).
--metric {f1,accuracy,precision,recall,auc,loss}
    Performance metric to aggregate across seeds (default: f1).
--complete-only
    Restrict aggregated results and plots to datasets that have completed
    all configurations, ensuring a fair comparison.
--model-type {linear,mlp,all}
    Restrict plots to a single model family (default: all).

Examples
--------
# Default: test F1, all models, all datasets
python optuna_progress.py

# Validation accuracy, complete datasets only
python optuna_progress.py --split val --metric accuracy --complete-only

# Test recall for MLP only
python optuna_progress.py --metric recall --model-type mlp

# Test precision, complete datasets, linear model
python optuna_progress.py --metric precision --complete-only --model-type linear
"""

import argparse
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import math
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

plt.rcParams.update({"figure.dpi": 130, "font.size": 10})

DB_PATH = Path("..") / "results" / "optuna_studies.db"
STORAGE = f"sqlite:///{DB_PATH.resolve()}"
print(f"DB: {DB_PATH.resolve()}  |  exists: {DB_PATH.exists()}")

# Config suffix → human-readable label (shown in legend / axes)
CONFIG_LABELS = {
    "curr0_gate0":  "Baseline",
    "curr1_gate0":  "+ Curriculum",
    "curr0_gate1":  "+ Gate",
    "curr1_gate1":  "+ Gate + Curr",
    "ag":           "AutoGluon prep",
    "saga":         "Saga++ prep",
    "cp":           "CP prep",
    "baseline_zero": "Zero imputation",
    "knn":          "KNN imputation",
}

BAR_ORDER = [
    "Clean",
    "Baseline",
    "+ Curriculum",
    "AutoGluon prep",
    "Saga++ prep",
    "CP prep",
    "Zero imputation",
    "KNN imputation",
    "+ Gate",
    "+ Gate + Curr",
]

BAR_COLORS = {
    "Clean":            "#4c9bcd",
    "Baseline":         "#aaaaaa",
    "+ Curriculum":     "#e07b39",
    "AutoGluon prep":   "#9b59b6",
    "Saga++ prep":      "#1abc9c",
    "CP prep":          "#f39c12",
    "Zero imputation":  "#778ca3",
    "KNN imputation":   "#00acc1",
    "+ Gate":           "#e74c3c",
    "+ Gate + Curr":    "#2ecc71",
}

NCOLS = 5

AVAILABLE_METRICS = ["f1", "accuracy", "precision", "recall", "auc", "loss"]
AVAILABLE_SPLITS  = ["test", "val"]

_METRIC_LABELS = {
    "f1":        "F1",
    "accuracy":  "Accuracy",
    "precision": "Precision",
    "recall":    "Recall",
    "auc":       "AUC-ROC",
    "loss":      "Loss",
}

# Keys tried in order when extracting per-seed CPU/wall time
_TIME_KEYS = ["cpu_time", "train_time", "elapsed", "time"]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_studies(split: str = "test", metric: str = "f1") -> pd.DataFrame:
    """
    Load all Optuna studies from the SQLite database and extract performance
    metrics and timing information for the best trial of each study.

    Study names must follow the pattern::

        {dataset}_{data_mode}_{model_type}_{config}

    where the valid tokens are:

    * **data_mode**  : ``clean`` | ``ar`` | ``nar``
    * **model_type** : ``linear`` | ``mlp``
    * **config**     : ``curr0_gate0`` | ``curr1_gate0`` | ``curr0_gate1`` |
                       ``curr1_gate1`` | ``ag`` | ``saga`` | ``cp`` |
                       ``baseline_zero`` | ``knn``

    Metrics are read from ``best_trial.user_attrs["seed_results"]``, a list
    of per-seed dicts whose keys follow ``{split}_{metric}`` (e.g.
    ``test_f1``, ``val_accuracy``).  If that list is empty the function falls
    back to scalar attrs ``avg_{split}_metric`` / ``std_{split}_metric``.

    CPU/wall time is resolved in priority order:

    1. ``seed_results[i]["cpu_time"]`` or ``["train_time"]`` / ``["elapsed"]`` / ``["time"]``
    2. ``best_trial.user_attrs["cpu_time"]``
    3. ``best_trial.duration.total_seconds()``  (wall clock)

    Parameters
    ----------
    split : {'test', 'val'}, default 'test'
        Evaluation split to use:

        * ``'test'`` – held-out test set
        * ``'val'``  – validation set used during HPO

    metric : {'f1', 'accuracy', 'precision', 'recall', 'auc', 'loss'}, default 'f1'
        Performance metric to extract:

        * ``'f1'``        – macro / weighted F1 score
        * ``'accuracy'``  – classification accuracy
        * ``'precision'`` – precision score
        * ``'recall'``    – recall score
        * ``'auc'``       – area under ROC curve
        * ``'loss'``      – cross-entropy / task loss (lower is better)

    Returns
    -------
    pd.DataFrame
        One row per study.  Columns: ``dataset``, ``data_mode``,
        ``model_type``, ``config``, ``config_label``, ``metric_mean``,
        ``metric_std``, ``n_seeds``, ``cpu_time_mean``.
    """
    _RE = re.compile(
        r"^(?P<dataset>.+)_(?P<data_mode>clean|ar|nar)_(?P<model_type>linear|mlp)"
        r"_(?P<config>curr[01]_gate[01]|ag|saga|cp|baseline_zero|knn)$"
    )
    metric_key = f"{split}_{metric}"

    rows, skipped = [], 0
    for name in optuna.get_all_study_names(storage=STORAGE):
        m = _RE.match(name)
        if not m:
            skipped += 1
            continue
        study = optuna.load_study(study_name=name, storage=STORAGE)
        try:
            best = study.best_trial
        except Exception:
            skipped += 1
            continue

        seed_results = best.user_attrs.get("seed_results", [])
        metric_values = [
            r[metric_key]
            for r in seed_results
            if metric_key in r and r[metric_key] is not None
        ]
        if metric_values:
            mean_val = float(np.mean(metric_values))
            std_val  = float(np.std(metric_values, ddof=min(1, len(metric_values) - 1)))
        else:
            mean_val = best.user_attrs.get(f"avg_{split}_metric")
            std_val  = best.user_attrs.get(f"std_{split}_metric", 0.0)

        if mean_val is None:
            skipped += 1
            continue

        # Resolve per-seed CPU / wall time
        cpu_vals = []
        for r in seed_results:
            for k in _TIME_KEYS:
                if r.get(k) is not None:
                    cpu_vals.append(float(r[k]))
                    break
        if cpu_vals:
            cpu_time_mean = float(np.mean(cpu_vals))
        elif best.user_attrs.get("cpu_time") is not None:
            cpu_time_mean = float(best.user_attrs["cpu_time"])
        elif best.duration is not None:
            cpu_time_mean = best.duration.total_seconds()
        else:
            cpu_time_mean = float("nan")

        d = m.groupdict()
        rows.append({
            "dataset":       d["dataset"],
            "data_mode":     d["data_mode"],
            "model_type":    d["model_type"],
            "config":        d["config"],
            "config_label":  CONFIG_LABELS.get(d["config"], d["config"]),
            "metric_mean":   mean_val,
            "metric_std":    std_val,
            "n_seeds":       len(metric_values),
            "cpu_time_mean": cpu_time_mean,
        })

    df = pd.DataFrame(rows)
    n_ds   = df["dataset"].nunique()    if not df.empty else 0
    n_cfg  = df["config_label"].nunique() if not df.empty else 0
    has_t  = df["cpu_time_mean"].notna().sum() if not df.empty else 0
    print(
        f"\n{'─'*60}\n"
        f" Studies loaded : {len(df):>4}  matched  |  {skipped:>3} skipped\n"
        f" Metric key     : '{metric_key}'\n"
        f" Datasets       : {n_ds}\n"
        f" Configs        : {n_cfg}\n"
        f" With CPU time  : {has_t} / {len(df)}\n"
        f"{'─'*60}"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Filtering
# ─────────────────────────────────────────────────────────────────────────────

def filter_complete_datasets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only datasets that have *all* configs present for every
    (model_type, data_mode) group, ensuring a fair cross-config comparison.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`load_studies`.

    Returns
    -------
    pd.DataFrame
        Filtered subset of *df*.
    """
    parts = []
    for (model_type, data_mode), group in df.groupby(["model_type", "data_mode"]):
        all_configs = set(group["config"].unique())
        complete_datasets = (
            group.groupby("dataset")["config"]
            .apply(set)
            .pipe(lambda s: s[s.apply(lambda c: c >= all_configs)].index)
        )
        n_dropped = group["dataset"].nunique() - len(complete_datasets)
        if n_dropped:
            print(f"  [complete-only] {model_type}/{data_mode}: "
                  f"dropping {n_dropped} incomplete dataset(s)")
        parts.append(group[group["dataset"].isin(complete_datasets)])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Bar-chart comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(
    df: pd.DataFrame,
    model_type: str,
    noise_mode: str,
    metric_col:   str = "metric_mean",
    metric_label: str = "Metric",
) -> None:
    """
    One subplot per dataset, configurations on x-axis, metric on y-axis.

    The y-axis is auto-zoomed to the actual data range so small differences
    remain visible.  Each bar is annotated with ``mean ± std`` above the
    error cap.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`load_studies`.
    model_type : {'linear', 'mlp'}
        Model family to include.
    noise_mode : {'ar', 'nar'}
        Noise regime shown alongside the clean reference.
    metric_col : str, default 'metric_mean'
        Column in *df* holding the scalar performance value.
    metric_label : str, default 'Metric'
        Human-readable name printed on the y-axis.
    """
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
        print(f"  No data for {model_type} / {noise_mode}.")
        return

    datasets    = sorted(combined["dataset"].unique())
    n_datasets  = len(datasets)
    present_lbl = combined["config_label"].unique()
    bar_labels  = [b for b in BAR_ORDER if b in present_lbl]
    colors      = [BAR_COLORS.get(l, "#cccccc") for l in bar_labels]
    x_pos       = np.arange(len(bar_labels))

    std_col = metric_col.replace("_mean", "_std")
    lut     = combined.set_index(["dataset", "config_label"])
    nrows   = math.ceil(n_datasets / NCOLS)
    fig, axes = plt.subplots(
        nrows, NCOLS,
        figsize=(NCOLS * 5, nrows * 5),
        squeeze=False,
    )

    for idx, dataset in enumerate(datasets):
        ax = axes[idx // NCOLS][idx % NCOLS]

        means, stds = [], []
        for label in bar_labels:
            key = (dataset, label)
            if key in lut.index:
                row = lut.loc[key]
                means.append(float(row[metric_col]) if not isinstance(row, pd.DataFrame)
                             else float(row[metric_col].iloc[0]))
                stds.append(float(row[std_col]) if not isinstance(row, pd.DataFrame)
                            else float(row[std_col].iloc[0]))
            else:
                means.append(np.nan)
                stds.append(0.0)

        ax.bar(
            x_pos, means, width=0.65,
            yerr=stds, capsize=3,
            color=colors,
            error_kw={"elinewidth": 0.9, "ecolor": "#333"},
        )

        for xi, mean, std in zip(x_pos, means, stds):
            if not np.isnan(mean):
                cap_top = mean + std
                ax.text(
                    xi, cap_top + 0.002,
                    f"{mean:.4f}\n±{std:.4f}",
                    ha="center", va="bottom",
                    fontsize=5, rotation=90, color="#222",
                )

        valid_pairs = [(m, s) for m, s in zip(means, stds) if not np.isnan(m)]
        if valid_pairs:
            y_min_data = min(m - s for m, s in valid_pairs)
            y_max_data = max(m + s for m, s in valid_pairs)
            span       = max(y_max_data - y_min_data, 0.005)
            bottom     = max(0.0, y_min_data - span * 0.15)
            headroom   = max(0.07, span * 1.2)
            ax.set_ylim(bottom, y_max_data + headroom)

        ax.set_title(dataset, fontsize=8, fontweight="bold", pad=4)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bar_labels, rotation=40, ha="right", fontsize=6)
        ax.set_ylabel(f"{metric_label}", fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", linewidth=0.3, alpha=0.5)

    for idx in range(n_datasets, nrows * NCOLS):
        axes[idx // NCOLS][idx % NCOLS].set_visible(False)

    fig.suptitle(
        f"{model_type.upper()} — Clean vs {noise_mode.upper()}  [{metric_label}]",
        fontsize=13, fontweight="bold", y=1.005,
    )
    plt.tight_layout()
    out = f"plots/{model_type}_{noise_mode}_{metric_col}.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmap(
    df: pd.DataFrame,
    model_type: str,
    noise_mode: str,
    metric_col:   str = "metric_mean",
    metric_label: str = "Metric",
) -> None:
    """
    Two-panel heatmap making inter-method gaps immediately visible.

    **Left panel — row-normalised absolute values**
      Each row (dataset) is independently rescaled so the weakest config maps
      to 0 and the strongest maps to 1.  This removes the confound of varying
      dataset difficulty and reveals *relative rankings* even when the global
      value range is narrow.  Cells show the raw (un-normalised) metric value.

    **Right panel — delta vs Baseline**
      Cells show ``value − Baseline`` for every non-Baseline config.  A
      diverging ``RdBu`` colourmap centred at zero makes it instantly clear
      which methods beat (blue) or lose to (red) the noisy Baseline.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`load_studies`.
    model_type : {'linear', 'mlp'}
        Model family to filter on.
    noise_mode : {'ar', 'nar'}
        Noise regime to compare against the clean reference.
    metric_col : str, default 'metric_mean'
        Column in *df* that contains the scalar metric to display.
    metric_label : str, default 'Metric'
        Human-readable name used in colour-bar labels.
    """
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
        print(f"  No data for {model_type} / {noise_mode}.")
        return

    present_lbl = combined["config_label"].unique()
    col_labels  = [b for b in BAR_ORDER if b in present_lbl]
    datasets    = sorted(combined["dataset"].unique())

    pivot = (
        combined
        .pivot_table(index="dataset", columns="config_label",
                     values=metric_col, aggfunc="mean")
        .reindex(index=datasets, columns=col_labels)
    )
    data = pivot.values.astype(float)
    n_rows, n_cols = data.shape

    # Row-normalise: each row spans [0, 1] relative to its own min/max
    row_min  = np.nanmin(data, axis=1, keepdims=True)
    row_max  = np.nanmax(data, axis=1, keepdims=True)
    row_span = np.where(row_max - row_min > 1e-9, row_max - row_min, 1e-9)
    data_norm = (data - row_min) / row_span

    # Delta vs Baseline
    baseline_idx = col_labels.index("Baseline") if "Baseline" in col_labels else None
    has_delta = baseline_idx is not None
    if has_delta:
        baseline_col = data[:, baseline_idx : baseline_idx + 1]
        data_delta   = data - baseline_col

    # Layout
    n_panels  = 2 if has_delta else 1
    cell_w, cell_h = 1.6, 0.45
    fig_w = n_panels * (n_cols * cell_w + 2.5) + 0.5
    fig_h = n_rows * cell_h + 2.5
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, fig_h))
    if n_panels == 1:
        axes = [axes]

    def _draw_panel(ax, mat, cmap, vmin, vmax, cbar_label, title, fmt):
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label=cbar_label, pad=0.02, fraction=0.046)
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(datasets, fontsize=8)
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        ax.set_title(title, fontsize=10, fontweight="bold", pad=14)
        ax.set_xlabel("Configuration", fontsize=8, labelpad=6)
        ax.set_ylabel("Dataset", fontsize=8)

        for i in range(n_rows):
            for j in range(n_cols):
                val = mat[i, j]
                if not np.isnan(val):
                    # Contrast-aware text colour
                    norm_val = (val - vmin) / max(vmax - vmin, 1e-9)
                    txt_color = "white" if abs(norm_val - 0.5) > 0.3 else "black"
                    raw_val = data[i, j]
                    ax.text(j, i, fmt(raw_val, val),
                            ha="center", va="center",
                            fontsize=6, color=txt_color)

    # Left: row-normalised
    _draw_panel(
        axes[0], data_norm,
        cmap="RdYlGn", vmin=0, vmax=1,
        cbar_label=f"Row-normalised {metric_label}  (0=worst, 1=best)",
        title=f"{model_type.upper()} / {noise_mode.upper()} — Row-normalised {metric_label}",
        fmt=lambda raw, _: f"{raw:.3f}",
    )

    # Right: delta vs Baseline
    if has_delta:
        vabs = np.nanmax(np.abs(data_delta))
        vabs = vabs if vabs > 1e-9 else 0.01
        _draw_panel(
            axes[1], data_delta,
            cmap="RdBu", vmin=-vabs, vmax=vabs,
            cbar_label=f"Δ {metric_label} vs Baseline",
            title=f"{model_type.upper()} / {noise_mode.upper()} — Δ vs Baseline",
            fmt=lambda _, delta: (f"+{delta:.3f}" if delta > 0
                                     else ("±0.000" if abs(delta) < 5e-4
                                           else f"{delta:.3f}")),
        )

    fig.suptitle(
        f"{model_type.upper()} — Clean vs {noise_mode.upper()}  [{metric_label}]",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out = f"plots/{model_type}_{noise_mode}_{metric_col}_heatmap.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Efficiency scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_efficiency(
    df: pd.DataFrame,
    model_type: str,
    noise_mode: str,
    metric_col:   str = "metric_mean",
    metric_label: str = "Metric",
) -> None:
    """
    Scatter plot of CPU/wall time vs metric per methodology.

    Aggregates each (config, data_mode) cell as the macro-average across all
    datasets, producing one point per methodology.  Points in the top-left
    corner dominate: higher metric *and* lower training time.  A Pareto
    frontier is drawn to highlight the efficiency–performance trade-off.

    Skips the plot if fewer than two configs have timing data available.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`load_studies`.  Must contain ``cpu_time_mean``.
    model_type : {'linear', 'mlp'}
        Model family to filter on.
    noise_mode : {'ar', 'nar'}
        Noise regime to include alongside the clean reference.
    metric_col : str, default 'metric_mean'
        Column in *df* that contains the scalar metric.
    metric_label : str, default 'Metric'
        Human-readable name shown on the y-axis.
    """
    noise_rows = df[
        (df["model_type"] == model_type)
        & (df["data_mode"] == noise_mode)
        & df["cpu_time_mean"].notna()
    ].copy()

    if noise_rows["config_label"].nunique() < 2:
        print(f"  [efficiency] Not enough timing data for {model_type}/{noise_mode}. Skipping.")
        return

    agg = (
        noise_rows
        .groupby("config_label")
        .agg(
            metric_macro=("metric_mean", "mean"),
            cpu_macro=("cpu_time_mean", "mean"),
            n=("dataset", "count"),
        )
        .reset_index()
    )

    _, ax = plt.subplots(figsize=(7, 5))

    for _, row in agg.iterrows():
        lbl   = row["config_label"]
        color = BAR_COLORS.get(lbl, "#999999")
        ax.scatter(row["cpu_macro"], row["metric_macro"],
                   color=color, s=120, zorder=3, label=lbl,
                   edgecolors="white", linewidths=0.6)
        ax.annotate(
            lbl,
            (row["cpu_macro"], row["metric_macro"]),
            textcoords="offset points", xytext=(6, 4),
            fontsize=7, color="#333",
        )

    # Pareto frontier (min cpu, max metric)
    pts = agg[["cpu_macro", "metric_macro"]].dropna().values
    if len(pts) >= 2:
        pts = pts[pts[:, 0].argsort()]
        pareto = [pts[0]]
        for p in pts[1:]:
            if p[1] >= pareto[-1][1]:
                pareto.append(p)
        if len(pareto) >= 2:
            px, py = zip(*pareto)
            ax.plot(px, py, "k--", linewidth=0.9, alpha=0.5,
                    label="Pareto frontier", zorder=2)

    cpu_vals = agg["cpu_macro"].dropna()
    if cpu_vals.max() / max(cpu_vals.min(), 1e-3) > 10:
        ax.set_xscale("log")
        ax.set_xlabel("Mean CPU / wall time  (s, log scale)", fontsize=9)
    else:
        ax.set_xlabel("Mean CPU / wall time  (s)", fontsize=9)

    ax.set_ylabel(f"Macro-avg {metric_label} across datasets", fontsize=9)
    ax.set_title(
        f"{model_type.upper()} / {noise_mode.upper()} — Efficiency vs {metric_label}",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=7, loc="best")
    ax.grid(linewidth=0.3, alpha=0.4)
    plt.tight_layout()
    out = f"plots/{model_type}_{noise_mode}_{metric_col}_efficiency.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate results table
# ─────────────────────────────────────────────────────────────────────────────

def overall_aggr_results(
    df: pd.DataFrame,
    metric_col:   str = "metric_mean",
    metric_label: str = "Metric",
) -> None:
    """
    Print aggregate statistics across all datasets to stdout.

    For every (model_type, data_mode, config) cell reports:

    * **Macro mean**   – unweighted mean of per-dataset means
    * **Macro median** – unweighted median of per-dataset means
    * **Rank**         – rank within its (model_type, data_mode) group  (1 = best)
    * **Win count**    – number of datasets on which this config achieves
                         the highest value

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`load_studies`.
    metric_col : str, default 'metric_mean'
        Column containing the per-study scalar metric.
    metric_label : str, default 'Metric'
        Human-readable metric name used in printed headers.
    """
    if df.empty:
        print("  No data to aggregate.")
        return

    # Macro mean and median per (model_type, data_mode, config)
    agg = (
        df.groupby(["model_type", "data_mode", "config", "config_label"])[metric_col]
        .agg(macro_mean="mean", macro_median="median")
        .reset_index()
    )

    # Win count: config that achieves the max per (model_type, data_mode, dataset)
    wins_df = (
        df.loc[
            df.groupby(["model_type", "data_mode", "dataset"])[metric_col].idxmax()
        ]
        .groupby(["model_type", "data_mode", "config"])
        .size()
        .reset_index(name="win_count")
    )
    agg = agg.merge(wins_df, on=["model_type", "data_mode", "config"], how="left")
    agg["win_count"] = agg["win_count"].fillna(0).astype(int)

    # Rank (1 = best) within each (model_type, data_mode)
    agg["rank"] = (
        agg.groupby(["model_type", "data_mode"])["macro_mean"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    for model_type, model_group in agg.groupby("model_type"):
        header = f"  Model: {model_type.upper()}"
        print(f"\n{'═'*70}")
        print(f"{header}   [{metric_label}]")
        print(f"{'═'*70}")

        # Pivot macro_mean across data_modes
        pivot_mean = (
            model_group
            .pivot_table(
                index=["config", "config_label", "rank"],
                columns="data_mode",
                values="macro_mean",
            )
            .reset_index()
        )
        pivot_mean.columns.name = None
        pivot_mean = pivot_mean.sort_values("rank")

        modes = [c for c in ["clean", "ar", "nar"] if c in pivot_mean.columns]
        col_w = 10

        # Header row
        print(f"\n  {'Rank':<5} {'Config':<22}", end="")
        for mode in modes:
            print(f"  {mode.upper():>{col_w}}", end="")
        print(f"  {'Wins':>6}")
        print(f"  {'-'*5} {'-'*22}", end="")
        for _ in modes:
            print(f"  {'-'*col_w}", end="")
        print(f"  {'-'*6}")

        for _, row in pivot_mean.iterrows():
            wins = int(
                model_group.loc[model_group["config"] == row["config"], "win_count"].sum()
            )
            print(f"  {int(row['rank']):<5} {row['config_label']:<22}", end="")
            for mode in modes:
                val = row.get(mode, float("nan"))
                print(f"  {val:>{col_w}.4f}" if not np.isnan(val) else f"  {'N/A':>{col_w}}", end="")
            print(f"  {wins:>6}")

        # Pivot macro_median
        pivot_med = (
            model_group
            .pivot_table(
                index=["config", "config_label"],
                columns="data_mode",
                values="macro_median",
            )
            .reset_index()
        )
        pivot_med.columns.name = None

        print(f"\n  Macro-median {metric_label}:")
        print(f"  {'Config':<27}", end="")
        for mode in modes:
            print(f"  {mode.upper():>{col_w}}", end="")
        print()
        print(f"  {'-'*27}", end="")
        for _ in modes:
            print(f"  {'-'*col_w}", end="")
        print()

        for _, row in pivot_med.iterrows():
            print(f"  {row['config_label']:<27}", end="")
            for mode in modes:
                val = row.get(mode, float("nan"))
                print(f"  {val:>{col_w}.4f}" if not np.isnan(val) else f"  {'N/A':>{col_w}}", end="")
            print()

    print(f"\n{'═'*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--split",
        choices=AVAILABLE_SPLITS,
        default="test",
        help=(
            "Evaluation split to read metrics from.  "
            "'test' = held-out test set (default);  "
            "'val'  = validation set used during HPO."
        ),
    )
    parser.add_argument(
        "--metric",
        choices=AVAILABLE_METRICS,
        default="f1",
        help=(
            "Performance metric to aggregate across seeds.  "
            "Choices: f1 (default), accuracy, precision, recall, auc, loss."
        ),
    )
    parser.add_argument(
        "--complete-only",
        action="store_true",
        help=(
            "Restrict aggregated results and plots to datasets that have "
            "completed all configurations, ensuring a fair comparison."
        ),
    )
    parser.add_argument(
        "--model-type",
        choices=["linear", "mlp", "all"],
        default="all",
        help="Restrict plots to a single model family (default: all).",
    )
    args = parser.parse_args()

    metric_label = _METRIC_LABELS.get(args.metric, args.metric.upper())
    metric_col   = "metric_mean"

    df = load_studies(split=args.split, metric=args.metric)
    if df.empty:
        print("No studies found. Check DB path and study naming convention.")
        raise SystemExit(1)

    if args.complete_only:
        print("\n[complete-only] Filtering datasets…")
        df = filter_complete_datasets(df)
        print(f"[complete-only] {df['dataset'].nunique()} dataset(s) retained.\n")

    model_types = (
        [args.model_type]
        if args.model_type != "all"
        else sorted(df["model_type"].unique())
    )

    Path("plots").mkdir(exist_ok=True)

    for model_type in model_types:
        for noise_mode in ["ar", "nar"]:
            print(f"\n── {model_type.upper()} / {noise_mode.upper()} ──")
            plot_comparison(df, model_type, noise_mode, metric_col, metric_label)
            plot_heatmap(df, model_type, noise_mode, metric_col, metric_label)
            plot_efficiency(df, model_type, noise_mode, metric_col, metric_label)

    overall_aggr_results(df, metric_col, metric_label)
