#!/usr/bin/env python3
"""
gate_analysis.py — Statistical evidence for the Proximal Gating approach.

Loads seed-level data from the Optuna database and produces:
  1. Statistical comparison: Gate vs Baseline (Wilcoxon signed-rank, effect size)
  2. Per-dataset delta table (printable LaTeX)
  3. Box/strip plots of Δ F1 (Gate − Baseline) and gate vs all competitors
  4. Noise robustness: relative degradation from clean baseline per method
  5. Gate behaviour analysis: sparsity, change rate, gate_std per dataset
  6. Training efficiency: epochs-to-90% across methods
  7. Critical Difference (rank-based) plot for all five methods
  8. Post-hoc Nemenyi test following the Friedman test (pairwise significance)

Usage
-----
    python gate_analysis.py
    python gate_analysis.py --metric accuracy
    python gate_analysis.py --metric auc --latex
    python gate_analysis.py --comparison-methods baseline gate saga catboost_dirty

Every competitor list in this file can be restricted to a chosen subset of
methods via --comparison-methods (or the comparison_methods key in
config.yaml) — see frogdq/comparison_methods.py for the full list of keys.
The "Clean" reference stays on regardless of this setting, since the
noise-robustness analysis here is built around measuring degradation *from*
clean.
"""

import argparse
import re
import sys
import warnings
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import scikit_posthocs as sp
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from frogdq.comparison_methods import METHOD_CHOICES, resolve_comparison_methods, resolve_from_config_token

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size":  10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

DB_PATH = Path("..") / "results" / "optuna_studies.db"
STORAGE = f"sqlite:///{DB_PATH.resolve()}"

_RE = re.compile(
    r"^(?P<dataset>.+)_(?P<data_mode>clean|ar|nar)_(?P<model_type>linear|mlp)"
    r"_(?P<config>curr[01]_gate[01]|ag|saga|cp|baseline_zero|knn)$"
)

# CatBoost is a standalone model baseline: study names are
# {dataset}_{data_mode}_catboost (no model_type/curr/gate suffix), so it
# needs its own pattern. Its "config" is synthesized as catboost_clean /
# catboost_dirty in load_seed_data() below.
_RE_CATBOOST = re.compile(
    r"^(?P<dataset>.+)_(?P<data_mode>clean|ar|nar)_catboost$"
)

CONFIG_LABELS = {
    "curr0_gate0": "Standard prep",
    "curr1_gate0": "Curriculum",
    "curr0_gate1": "QuAIL",
    "curr1_gate1": "+ Gate + Curr",
    "saga":        "Saga++",
    "cp":          "CP prep",
    "ag":          "AutoGluon",
    "catboost_clean": "CatBoost Clean",
    "catboost_dirty":  "CatBoost Dirty",
}

COLORS = {
    "Standard prep":     "#aaaaaa",
    "QuAIL":       "#e74c3c",
    "Curriculum": "#e07b39",
    "Saga++":       "#1abc9c",
    "CP prep":      "#f39c12",
    "CatBoost Clean": "#27ae60",
    "CatBoost Dirty": "#8e44ad",
}

# comparison_methods selection (canonical keys from frogdq.comparison_methods),
# set once by __main__ from --comparison-methods / config.yaml. None = all
# methods. Every one of this file's hardcoded competitor lists is filtered
# through _filter_methods()/_filter_tokens() below, EXCEPT the "Clean"
# reference itself (config == "curr0_gate0" at data_mode == "clean"), which
# stays structurally always-on: the noise-robustness analysis in this file is
# built around measuring degradation *from* clean, so dropping it would break
# half the plots rather than just narrow the comparison.
_SELECTED_METHODS: Optional[List[str]] = None


def _filter_methods(items):
    """Filter a list of (config_token, label) tuples by the comparison_methods selection."""
    if _SELECTED_METHODS is None:
        return items
    # These lists are only ever used for AR/NAR competitor comparisons, so
    # "curr0_gate0" unambiguously means "baseline" here (never "clean").
    return [
        (tok, lbl) for tok, lbl in items
        if resolve_from_config_token(tok, data_mode="ar") in _SELECTED_METHODS
    ]


def _filter_tokens(tokens):
    """Filter a bare list of config tokens (methods_order lists) the same way."""
    if _SELECTED_METHODS is None:
        return tokens
    return [t for t in tokens if resolve_from_config_token(t, data_mode="ar") in _SELECTED_METHODS]


Path("plots").mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_seed_data(metric: str = "f1") -> pd.DataFrame:
    """
    Load per-seed results from the best trial of every matched study.

    Returns one row per (study × seed). Columns include the chosen
    performance metric, training-convergence fields, and gate-behaviour
    fields (NaN for non-gate configs).
    """
    perf_key   = f"test_{metric}"
    gate_keys  = ["final_gate_sparsity", "final_gate_std", "gate_change_rate"]
    conv_keys  = ["n_epochs_trained", "best_epoch",
                  "epochs_to_90pct", "epochs_to_95pct", "epochs_to_99pct",
                  "convergence_stability", "cpu_time"]

    rows = []
    for name in optuna.get_all_study_names(storage=STORAGE):
        m = _RE.match(name)
        if m:
            d = m.groupdict()
        else:
            m_cb = _RE_CATBOOST.match(name)
            if not m_cb:
                continue
            gd = m_cb.groupdict()
            d = {
                "dataset":    gd["dataset"],
                "data_mode":  gd["data_mode"],
                "model_type": "catboost",
                "config":     "catboost_clean" if gd["data_mode"] == "clean" else "catboost_dirty",
            }
        study = optuna.load_study(study_name=name, storage=STORAGE)
        try:
            best = study.best_trial
        except Exception:
            continue

        seed_results = best.user_attrs.get("seed_results", [])
        if not seed_results:
            continue

        label = CONFIG_LABELS.get(d["config"], d["config"])

        for sr in seed_results:
            if perf_key not in sr or sr[perf_key] is None:
                continue
            row = {
                "dataset":     d["dataset"],
                "data_mode":   d["data_mode"],
                "model_type":  d["model_type"],
                "config":      d["config"],
                "label":       label,
                "seed":        sr.get("seed"),
                "metric":      float(sr[perf_key]),
            }
            for k in conv_keys:
                row[k] = float(sr[k]) if sr.get(k) is not None else float("nan")
            for k in gate_keys:
                row[k] = float(sr[k]) if sr.get(k) is not None else float("nan")
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} seed rows | "
          f"{df['dataset'].nunique()} datasets | "
          f"{df['config'].nunique()} configs | "
          f"metric: test_{metric}")
    return df


def dataset_means(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse seeds → one (dataset, config, data_mode) row with mean/std."""
    return (
        df.groupby(["dataset", "data_mode", "config", "label"])
        .agg(
            metric_mean=("metric",         "mean"),
            metric_std=("metric",          "std"),
            epochs90_mean=("epochs_to_90pct", "mean"),
            epochs95_mean=("epochs_to_95pct", "mean"),
            epochs99_mean=("epochs_to_99pct", "mean"),
            conv_stab_mean=("convergence_stability", "mean"),
            cpu_mean=("cpu_time",          "mean"),
            gate_sparsity=("final_gate_sparsity", "mean"),
            gate_std=("final_gate_std",    "mean"),
            gate_change=("gate_change_rate","mean"),
            n_seeds=("metric",             "count"),
        )
        .reset_index()
    )


def broadcast_catboost_clean(dm: pd.DataFrame) -> pd.DataFrame:
    """
    CatBoost-Clean has no AR/NAR variant (it's trained once, on clean data
    only), but every comparison function below filters by (config, data_mode)
    together. Duplicate its single clean-mode row into synthetic AR/NAR rows
    (same scores) so it can sit alongside "catboost_dirty" and every other
    method as a fixed reference — exactly like the "clean" baseline already
    does via plot_noise_robustness's fixed-reference lookup.
    """
    clean_rows = dm[dm["config"] == "catboost_clean"]
    if clean_rows.empty:
        return dm
    dupes = []
    for noise_mode in ["ar", "nar"]:
        dup = clean_rows.copy()
        dup["data_mode"] = noise_mode
        dupes.append(dup)
    return pd.concat([dm] + dupes, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Statistical helpers
# ─────────────────────────────────────────────────────────────────────────────

def wilcoxon_and_effect(x: np.ndarray, y: np.ndarray):
    """
    Paired Wilcoxon signed-rank test and rank-biserial correlation (r).

    Returns (statistic, p_value, effect_r).
    """
    diffs = x - y
    diffs = diffs[~np.isnan(diffs)]
    diffs = diffs[diffs != 0]
    if len(diffs) < 4:
        return float("nan"), float("nan"), float("nan")
    stat, p = stats.wilcoxon(diffs, alternative="two-sided")
    n  = len(diffs)
    # Rank-biserial: r = 1 - 2W / (n*(n+1)/2)
    r  = 1 - (2 * stat) / (n * (n + 1) / 2)
    return float(stat), float(p), float(r)


def _fmt_p(p: float) -> str:
    """Format a p-value to 4 decimals; report '< 0.05' if it rounds to 0.0000."""
    if np.isnan(p):
        return "—"
    s = f"{p:.4f}"
    return "< 0.05" if s == "0.0000" else s


def sign_test(x: np.ndarray, y: np.ndarray):
    """Returns (n_wins_x, n_draws, n_wins_y) where draws = |diff| < 1e-4."""
    diffs = x - y
    diffs = diffs[~np.isnan(diffs)]
    wins  = int((diffs >  1e-4).sum())
    draws = int((np.abs(diffs) <= 1e-4).sum())
    losses= int((diffs < -1e-4).sum())
    return wins, draws, losses


# ─────────────────────────────────────────────────────────────────────────────
# 1. Statistical comparison: Gate vs all others
# ─────────────────────────────────────────────────────────────────────────────

def stats_summary(dm: pd.DataFrame, metric_name: str) -> None:
    """
    Print a statistical summary table: Gate vs every other method,
    separately for AR and NAR noise modes.
    """
    print("\n" + "═" * 72)
    print(f"  STATISTICAL COMPARISON — Gate vs competitors  [{metric_name}]")
    print("═" * 72)

    gate_config = "curr0_gate0"   # Baseline
    test_config = "curr0_gate1"   # + Gate

    for noise_mode in ["ar", "nar"]:
        gate_rows = dm[(dm["config"] == test_config) & (dm["data_mode"] == noise_mode)]
        gate_vals = gate_rows.set_index("dataset")["metric_mean"]

        print(f"\n  Noise mode: {noise_mode.upper()}")
        print(f"  {'Competitor':<20} {'N':>3}  {'Δ mean':>8}  {'Δ median':>9}  "
              f"{'W/D/L':>9}  {'p-value':>9}  {'|r|':>6}  {'sig':>4}")
        print(f"  {'-'*20} {'-'*3}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*6}  {'-'*4}")

        for comp_cfg, comp_label in _filter_methods([
            ("curr0_gate0", "Baseline"),
            ("curr1_gate0", "+ Curriculum"),
            ("saga",        "Saga++"),
            ("cp",          "CP prep"),
            ("catboost_clean", "CatBoost Clean"),
            ("catboost_dirty", "CatBoost Dirty"),
        ]):
            comp_rows = dm[(dm["config"] == comp_cfg) & (dm["data_mode"] == noise_mode)]
            comp_vals = comp_rows.set_index("dataset")["metric_mean"]

            shared_ds = gate_vals.index.intersection(comp_vals.index)
            if len(shared_ds) < 4:
                continue

            g = gate_vals[shared_ds].values
            c = comp_vals[shared_ds].values
            deltas = g - c

            stat, p, r = wilcoxon_and_effect(g, c)
            w, d, l    = sign_test(g, c)
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))

            print(
                f"  {comp_label:<20} {len(shared_ds):>3}  "
                f"{np.nanmean(deltas):>+8.4f}  {np.nanmedian(deltas):>+9.4f}  "
                f"{w}/{d}/{l:>2}  {p:>9.4f}  {abs(r):>6.3f}  {sig:>4}"
            )

    print()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Per-dataset delta table
# ─────────────────────────────────────────────────────────────────────────────

def delta_table(dm: pd.DataFrame, metric_name: str, latex: bool = False) -> None:
    """Print per-dataset Δ(Gate − Baseline) for AR and NAR."""
    print("\n" + "═" * 72)
    print(f"  PER-DATASET Δ (Gate − Baseline)  [{metric_name}]")
    print("═" * 72)

    rows_out = []
    for noise_mode in ["ar", "nar"]:
        gate = dm[(dm["config"] == "curr0_gate1") & (dm["data_mode"] == noise_mode)].set_index("dataset")
        base = dm[(dm["config"] == "curr0_gate0") & (dm["data_mode"] == noise_mode)].set_index("dataset")
        shared = gate.index.intersection(base.index)
        for ds in sorted(shared):
            g_val = gate.loc[ds, "metric_mean"]
            b_val = base.loc[ds, "metric_mean"]
            rows_out.append({
                "Dataset":     ds,
                "Mode":        noise_mode.upper(),
                "Baseline":    f"{b_val:.4f}",
                "Gate":        f"{g_val:.4f}",
                "Δ":           f"{g_val - b_val:+.4f}",
                "Direction":   "↑" if g_val > b_val + 1e-4 else ("↓" if g_val < b_val - 1e-4 else "="),
            })

    df_out = pd.DataFrame(rows_out)
    if latex:
        print(df_out.to_latex(index=False))
    else:
        print(df_out.to_string(index=False))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Box/strip plot: Δ metric per method vs Baseline
# ─────────────────────────────────────────────────────────────────────────────

def plot_deltas(dm: pd.DataFrame, metric_name: str) -> None:
    """
    Box + strip plot of Δ (method − Baseline) across datasets,
    separately for AR and NAR, one panel per noise mode.
    """
    competitors = _filter_methods([
        ("curr0_gate1", "+ Gate"),
        ("curr1_gate0", "+ Curriculum"),
        ("saga",        "Saga++"),
        ("cp",          "CP prep"),
        ("catboost_clean", "CatBoost Clean"),
        ("catboost_dirty", "CatBoost Dirty"),
    ])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, noise_mode in zip(axes, ["ar", "nar"]):
        base = dm[(dm["config"] == "curr0_gate0") & (dm["data_mode"] == noise_mode)].set_index("dataset")

        all_labels, all_deltas = [], []
        for cfg, lbl in competitors:
            comp = dm[(dm["config"] == cfg) & (dm["data_mode"] == noise_mode)].set_index("dataset")
            shared = base.index.intersection(comp.index)
            deltas = (comp.loc[shared, "metric_mean"] - base.loc[shared, "metric_mean"]).values
            all_labels.append(lbl)
            all_deltas.append(deltas)

        bp = ax.boxplot(
            all_deltas, patch_artist=True, notch=False,
            widths=0.45, medianprops={"color": "black", "linewidth": 1.5},
            whiskerprops={"linewidth": 0.8}, capprops={"linewidth": 0.8},
            flierprops={"marker": ".", "markersize": 4, "alpha": 0.4},
        )
        for patch, lbl in zip(bp["boxes"], all_labels):
            patch.set_facecolor(COLORS.get(lbl, "#cccccc"))
            patch.set_alpha(0.7)

        # Strip overlay
        for xi, (lbl, deltas) in enumerate(zip(all_labels, all_deltas), start=1):
            jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(deltas))
            ax.scatter(xi + jitter, deltas,
                       color=COLORS.get(lbl, "#888"), s=22, alpha=0.6, zorder=3)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xticks(range(1, len(all_labels) + 1))
        ax.set_xticklabels(all_labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(f"Δ {metric_name} vs Baseline", fontsize=9)
        ax.set_title(f"{noise_mode.upper()} noise", fontsize=11, fontweight="bold")
        ax.grid(axis="y", linewidth=0.3, alpha=0.5)

    fig.suptitle(
        f"Performance gain over Baseline  [{metric_name}]  (each point = one dataset)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out = "plots/gate_delta_boxplot.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Noise robustness: degradation from clean
# ─────────────────────────────────────────────────────────────────────────────

def plot_noise_robustness(dm: pd.DataFrame, metric_name: str) -> None:
    """
    For each method and noise mode, compute the relative performance drop
    vs its own clean-baseline reference:
        degradation = (clean_baseline − noisy_method) / clean_baseline

    Positive = method improved vs clean (gate can recover degradation);
    negative = method is *worse* than clean.
    """
    clean = (
        dm[(dm["config"] == "curr0_gate0") & (dm["data_mode"] == "clean")]
        .set_index("dataset")["metric_mean"]
    )

    methods = _filter_methods([
        ("curr0_gate0", "Baseline"),
        ("curr0_gate1", "+ Gate"),
        ("curr1_gate0", "+ Curriculum"),
        ("saga",        "Saga++"),
        ("cp",          "CP prep"),
        ("catboost_clean", "CatBoost Clean"),
        ("catboost_dirty", "CatBoost Dirty"),
    ])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, noise_mode in zip(axes, ["ar", "nar"]):
        positions, medians_neg, labels_used = [], [], []
        all_drops = []

        for xi, (cfg, lbl) in enumerate(methods):
            comp = dm[(dm["config"] == cfg) & (dm["data_mode"] == noise_mode)].set_index("dataset")
            shared = clean.index.intersection(comp.index)
            if len(shared) < 3:
                continue
            # negative = gap remaining vs clean; positive = exceeded clean
            drops = (comp.loc[shared, "metric_mean"] - clean[shared]).values
            all_drops.append(drops)
            positions.append(xi)
            labels_used.append(lbl)

        bp = ax.boxplot(
            all_drops, positions=list(range(len(all_drops))),
            patch_artist=True, notch=False,
            widths=0.45, medianprops={"color": "black", "linewidth": 1.5},
            whiskerprops={"linewidth": 0.8}, capprops={"linewidth": 0.8},
            flierprops={"marker": ".", "markersize": 4, "alpha": 0.4},
        )
        for patch, lbl in zip(bp["boxes"], labels_used):
            patch.set_facecolor(COLORS.get(lbl, "#cccccc"))
            patch.set_alpha(0.7)

        for xi, (lbl, drops) in enumerate(zip(labels_used, all_drops)):
            jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(drops))
            ax.scatter(xi + jitter, drops,
                       color=COLORS.get(lbl, "#888"), s=22, alpha=0.6, zorder=3)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xticks(range(len(labels_used)))
        ax.set_xticklabels(labels_used, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(f"{metric_name} (noisy method) − {metric_name} (clean)", fontsize=8)
        ax.set_title(f"{noise_mode.upper()} noise — gap to clean performance",
                     fontsize=11, fontweight="bold")
        ax.grid(axis="y", linewidth=0.3, alpha=0.5)

    fig.suptitle(
        "Noise robustness: recovery of clean performance  "
        "(0 = matches clean; above = exceeds; below = degraded)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    out = "plots/gate_noise_robustness.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Gate behaviour analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_gate_behaviour(dm: pd.DataFrame) -> None:
    """
    Three scatter/bar panels for gate-specific metrics (only + Gate rows):
      - final_gate_sparsity per dataset (sorted)
      - gate_change_rate per dataset
      - Scatter: sparsity vs Δ F1 (Gate − Baseline)
    """
    gate_ar  = dm[(dm["config"] == "curr0_gate1") & (dm["data_mode"] == "ar")].copy()
    gate_nar = dm[(dm["config"] == "curr0_gate1") & (dm["data_mode"] == "nar")].copy()
    base_ar  = dm[(dm["config"] == "curr0_gate0") & (dm["data_mode"] == "ar")].set_index("dataset")
    base_nar = dm[(dm["config"] == "curr0_gate0") & (dm["data_mode"] == "nar")].set_index("dataset")

    gate_ar["delta"] = gate_ar.apply(
        lambda r: r["metric_mean"] - base_ar.loc[r["dataset"], "metric_mean"]
        if r["dataset"] in base_ar.index else float("nan"), axis=1
    )
    gate_nar["delta"] = gate_nar.apply(
        lambda r: r["metric_mean"] - base_nar.loc[r["dataset"], "metric_mean"]
        if r["dataset"] in base_nar.index else float("nan"), axis=1
    )

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    for row_idx, (gdf, mode) in enumerate([(gate_ar, "AR"), (gate_nar, "NAR")]):
        gdf = gdf.dropna(subset=["gate_sparsity"]).sort_values("gate_sparsity")
        ds_names = [d.replace("_", " ") for d in gdf["dataset"]]
        xi = np.arange(len(ds_names))

        # Panel 1: Sparsity bar chart
        ax = axes[row_idx][0]
        bars = ax.bar(xi, gdf["gate_sparsity"], color="#e74c3c", alpha=0.75, width=0.7)
        ax.set_xticks(xi)
        ax.set_xticklabels(ds_names, rotation=45, ha="right", fontsize=6)
        ax.set_ylabel("Gate sparsity (mean)", fontsize=8)
        ax.set_title(f"{mode} — Gate sparsity per dataset", fontsize=9, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.axhline(gdf["gate_sparsity"].mean(), color="black",
                   linestyle="--", linewidth=0.9, label=f"mean={gdf['gate_sparsity'].mean():.2f}")
        ax.legend(fontsize=7)
        ax.grid(axis="y", linewidth=0.3, alpha=0.5)

        # Panel 2: Gate change rate
        ax = axes[row_idx][1]
        gdf2 = gdf.sort_values("gate_change")
        ds2 = [d.replace("_", " ") for d in gdf2["dataset"]]
        ax.barh(ds2, gdf2["gate_change"], color="#3498db", alpha=0.75, height=0.7)
        ax.set_xlabel("Gate change rate (mean)", fontsize=8)
        ax.set_title(f"{mode} — Gate change rate", fontsize=9, fontweight="bold")
        ax.axvline(gdf2["gate_change"].mean(), color="black",
                   linestyle="--", linewidth=0.9, label=f"mean={gdf2['gate_change'].mean():.3f}")
        ax.legend(fontsize=7)
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(axis="x", linewidth=0.3, alpha=0.5)

        # Panel 3: Sparsity vs Δ metric
        ax = axes[row_idx][2]
        sub = gdf.dropna(subset=["gate_sparsity", "delta"])
        scatter = ax.scatter(
            sub["gate_sparsity"], sub["delta"],
            c=sub["gate_change"], cmap="viridis", s=60, alpha=0.8, zorder=3,
        )
        plt.colorbar(scatter, ax=ax, label="Gate change rate", pad=0.02)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        # Annotate dataset names
        for _, r in sub.iterrows():
            ax.annotate(r["dataset"].replace("_", "\n"),
                        (r["gate_sparsity"], r["delta"]),
                        fontsize=5, ha="center", va="bottom",
                        xytext=(0, 4), textcoords="offset points", color="#555")
        # Pearson r
        if len(sub) >= 4:
            rho, pv = stats.pearsonr(sub["gate_sparsity"], sub["delta"])
            ax.set_title(
                f"{mode} — Sparsity vs Δ metric  (r={rho:+.2f}, p={pv:.3f})",
                fontsize=9, fontweight="bold",
            )
        else:
            ax.set_title(f"{mode} — Sparsity vs Δ metric", fontsize=9, fontweight="bold")
        ax.set_xlabel("Gate sparsity", fontsize=8)
        ax.set_ylabel("Δ metric (Gate − Baseline)", fontsize=8)
        ax.grid(linewidth=0.3, alpha=0.4)

    plt.tight_layout()
    out = "plots/gate_behaviour.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Training efficiency: epochs-to-90%
# ─────────────────────────────────────────────────────────────────────────────

def _plot_training_efficiency_for_pct(dm: pd.DataFrame, metric_name: str, pct: int) -> None:
    """
    Compare training convergence speed (epochs to reach `pct`% of final
    validation metric) and convergence stability across methods.
    """
    methods = _filter_methods([
        ("curr0_gate0", "Baseline"),
        ("curr0_gate1", "+ Gate"),
        ("curr1_gate0", "+ Curriculum"),
        ("saga",        "Saga++"),
        ("cp",          "CP prep"),
        ("catboost_clean", "CatBoost Clean"),
        ("catboost_dirty", "CatBoost Dirty"),
    ])

    epochs_col = f"epochs{pct}_mean"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, noise_mode in zip(axes, ["ar", "nar"]):
        all_epochs, all_stab, labels_used = [], [], []
        for cfg, lbl in methods:
            sub = dm[(dm["config"] == cfg) & (dm["data_mode"] == noise_mode)]
            epochs = sub[epochs_col].dropna().values
            stab = sub["conv_stab_mean"].dropna().values
            if len(epochs) < 3:
                continue
            all_epochs.append(epochs)
            all_stab.append(stab)
            labels_used.append(lbl)

        xi = np.arange(len(labels_used))
        w = 0.35
        bars1 = ax.bar(xi - w / 2,
                       [np.nanmean(v) for v in all_epochs],
                       width=w, yerr=[np.nanstd(v) for v in all_epochs],
                       capsize=3, label=f"Epochs to {pct}%",
                       color=[COLORS.get(l, "#ccc") for l in labels_used],
                       alpha=0.85, error_kw={"elinewidth": 0.8})
        ax2 = ax.twinx()
        bars2 = ax2.bar(xi + w / 2,
                        [np.nanmean(v) for v in all_stab],
                        width=w, yerr=[np.nanstd(v) for v in all_stab],
                        capsize=3, label="Conv. stability",
                        color=[COLORS.get(l, "#ccc") for l in labels_used],
                        alpha=0.4, hatch="///", error_kw={"elinewidth": 0.8})
        ax.set_xticks(xi)
        ax.set_xticklabels(labels_used, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(f"Mean epochs to {pct}% val-metric", fontsize=8, color="#333")
        ax2.set_ylabel("Convergence stability (lower = smoother)", fontsize=8, color="#888")
        ax.set_title(f"{noise_mode.upper()} — Training efficiency",
                     fontsize=11, fontweight="bold")
        ax.grid(axis="y", linewidth=0.3, alpha=0.4)
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, fontsize=7, loc="upper right")

    fig.suptitle(
        f"Training efficiency comparison — {pct}% threshold  [{metric_name}]",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out = f"plots/gate_training_efficiency_{pct}.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


def plot_training_efficiency(dm: pd.DataFrame, metric_name: str) -> None:
    """
    Generate one training-efficiency plot per convergence threshold
    (90%, 95%, 99% of final validation metric), each comparing epochs-to-
    threshold and convergence stability across methods.
    """
    for pct in (90, 95, 99):
        _plot_training_efficiency_for_pct(dm, metric_name, pct)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Rank-based critical difference analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_rank_distribution(dm: pd.DataFrame, metric_name: str) -> None:
    """
    For each (dataset, noise_mode) rank all methods (1 = best).
    Plot mean rank ± std and distribution of ranks as a stacked bar.
    Also print Friedman test p-value.
    """
    methods_order = _filter_tokens([
        "curr0_gate0", "curr0_gate1", "curr1_gate0", "saga", "cp",
        "catboost_clean", "catboost_dirty",
    ])
    method_labels = [CONFIG_LABELS.get(c, c) for c in methods_order]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, noise_mode in zip(axes, ["ar", "nar"]):
        sub = dm[dm["data_mode"] == noise_mode]
        pivot = sub.pivot_table(index="dataset", columns="config",
                                values="metric_mean", aggfunc="mean")
        available = [c for c in methods_order if c in pivot.columns]
        pivot = pivot[available].dropna()

        if pivot.empty or len(pivot) < 4:
            ax.set_title(f"{noise_mode.upper()} — insufficient data")
            continue

        # Rank: 1 = highest metric
        ranked = pivot.rank(axis=1, ascending=False, method="min")

        avail_labels = [CONFIG_LABELS.get(c, c) for c in available]
        mean_ranks = ranked.mean(axis=0).values
        std_ranks  = ranked.std(axis=0).values

        # Friedman test
        try:
            _, p_friedman = stats.friedmanchisquare(*[pivot[c].values for c in available])
        except Exception:
            p_friedman = float("nan")

        # Mean rank bar (bars span [1, mean_rank] — rank can never be < 1)
        colors_used = [COLORS.get(l, "#ccc") for l in avail_labels]
        ax.bar(avail_labels, mean_ranks - 1, bottom=1, yerr=std_ranks, capsize=4,
               color=colors_used, alpha=0.8,
               error_kw={"elinewidth": 0.9, "ecolor": "#333"}, width=0.6)
        ax.set_ylim(0.5, len(available) + 0.5)
        ax.invert_yaxis()
        ax.axhline(1, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
        for xi, (mr, sl) in enumerate(zip(mean_ranks, std_ranks)):
            ax.text(xi, mr + sl + 0.1, f"{mr:.2f}", ha="center", va="top",
                    fontsize=8, fontweight="bold")
        ax.set_xticks(range(len(avail_labels)))
        ax.set_xticklabels(avail_labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Mean rank  (1 = best)", fontsize=9)
        ax.set_title(
            f"{noise_mode.upper()} — Mean rank across {len(pivot)} datasets\n"
            f"Friedman p = {_fmt_p(p_friedman)}",
            fontsize=10, fontweight="bold",
        )
        ax.grid(axis="y", linewidth=0.3, alpha=0.4)

    fig.suptitle(
        f"Rank-based comparison across datasets  [{metric_name}]",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out = "plots/gate_rank_distribution.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()

    # Print rank table
    print("\n" + "═" * 60)
    print(f"  RANK TABLE  [{metric_name}]")
    print("═" * 60)
    for noise_mode in ["ar", "nar"]:
        sub = dm[dm["data_mode"] == noise_mode]
        pivot = sub.pivot_table(index="dataset", columns="config",
                                values="metric_mean", aggfunc="mean")
        available = [c for c in methods_order if c in pivot.columns]
        pivot = pivot[available].dropna()
        if pivot.empty:
            continue
        ranked = pivot.rank(axis=1, ascending=False, method="min")
        avail_labels = [CONFIG_LABELS.get(c, c) for c in available]
        try:
            _, pf = stats.friedmanchisquare(*[pivot[c].values for c in available])
        except Exception:
            pf = float("nan")
        print(f"\n  {noise_mode.upper()} (N={len(pivot)}, Friedman p={_fmt_p(pf)})")
        print(f"  {'Method':<22} {'Mean rank':>10}  {'Std':>7}  {'Best count':>10}")
        print(f"  {'-'*22} {'-'*10}  {'-'*7}  {'-'*10}")
        for c, l in zip(available, avail_labels):
            mr  = ranked[c].mean()
            sr  = ranked[c].std()
            best = int((ranked[c] == 1).sum())
            print(f"  {l:<22} {mr:>10.3f}  {sr:>7.3f}  {best:>10}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 8. Post-hoc Nemenyi test
# ─────────────────────────────────────────────────────────────────────────────

def nemenyi_posthoc(dm: pd.DataFrame, metric_name: str) -> None:
    """
    Post-hoc Nemenyi test following a significant Friedman result.

    The Friedman test (already reported by :func:`plot_rank_distribution`)
    only tells us *some* method differs from the others — it does not say
    which pairs. For each noise mode this function restricts to datasets
    where all five methods are present, re-runs Friedman as a gate, and (only
    if significant at alpha=0.05) runs the pairwise Nemenyi test on ranks.

    Prints the full pairwise p-value matrix, lists significant pairs
    (p < 0.05), and saves a p-value heatmap to
    ``plots/gate_nemenyi_{noise_mode}.png``.
    """
    methods_order = _filter_tokens([
        "curr0_gate0", "curr0_gate1", "curr1_gate0", "saga", "cp",
        "catboost_clean", "catboost_dirty",
    ])

    print("\n" + "═" * 72)
    print(f"  POST-HOC NEMENYI TEST (after Friedman)  [{metric_name}]")
    print("═" * 72)

    for noise_mode in ["ar", "nar"]:
        sub = dm[dm["data_mode"] == noise_mode]
        pivot = sub.pivot_table(index="dataset", columns="config",
                                values="metric_mean", aggfunc="mean")
        available = [c for c in methods_order if c in pivot.columns]
        pivot = pivot[available].dropna()

        if pivot.empty or len(pivot) < 4 or len(available) < 3:
            print(f"\n  {noise_mode.upper()}: insufficient data, skipping.")
            continue

        labels = [CONFIG_LABELS.get(c, c) for c in available]
        stat, p_friedman = stats.friedmanchisquare(*[pivot[c].values for c in available])

        print(f"\n  {noise_mode.upper()}  (N={len(pivot)} datasets, {len(available)} methods)")
        print(f"  Friedman chi2={stat:.3f}, p={p_friedman:.4f}")

        if p_friedman >= 0.05:
            print("  Friedman not significant at alpha=0.05 — skipping post-hoc test.")
            continue

        nem = sp.posthoc_nemenyi_friedman(pivot.values)
        nem.index   = labels
        nem.columns = labels

        print("\n  Pairwise Nemenyi p-values:")
        print("  " + nem.round(4).to_string().replace("\n", "\n  "))

        print("\n  Significant pairs (p < 0.05):")
        sig_found = False
        for i, li in enumerate(labels):
            for j, lj in enumerate(labels):
                if j <= i:
                    continue
                pv = nem.iloc[i, j]
                if pv < 0.05:
                    sig_found = True
                    print(f"    {li:<14} vs {lj:<14}  p={pv:.4f}")
        if not sig_found:
            print("    (none — Friedman significant overall, but no pair survives correction)")

        # Heatmap
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(nem.values, cmap="viridis_r", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label="p-value", pad=0.02, fraction=0.046)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        for i in range(len(labels)):
            for j in range(len(labels)):
                pv = nem.values[i, j]
                txt_color = "white" if pv < 0.5 else "black"
                marker = "*" if (pv < 0.05 and i != j) else ""
                ax.text(j, i, f"{pv:.3f}{marker}", ha="center", va="center",
                        fontsize=7, color=txt_color)
        ax.set_title(
            f"{noise_mode.upper()} — Nemenyi post-hoc p-values  [{metric_name}]\n"
            f"(* = p<0.05; Friedman p={p_friedman:.4f})",
            fontsize=10, fontweight="bold",
        )
        plt.tight_layout()
        out = f"plots/gate_nemenyi_{noise_mode}.png"
        plt.savefig(out, bbox_inches="tight")
        print(f"\n  Saved: {out}")
        plt.close()

    print()


# ─────────────────────────────────────────────────────────────────────────────
# 9. Summary evidence table (printer-friendly / LaTeX-ready)
# ─────────────────────────────────────────────────────────────────────────────

def evidence_summary(dm: pd.DataFrame, metric_name: str, latex: bool = False) -> None:
    """
    Print a compact summary of the key evidence supporting the Gate approach:
      - macro mean metric per method per noise mode
      - win rate vs Baseline
      - Wilcoxon p-value and effect size
    """
    methods = _filter_methods([
        ("curr0_gate0", "Baseline"),
        ("curr0_gate1", "+ Gate"),
        ("curr1_gate0", "+ Curriculum"),
        ("saga",        "Saga++"),
        ("cp",          "CP prep"),
        ("catboost_clean", "CatBoost Clean"),
        ("catboost_dirty", "CatBoost Dirty"),
    ])

    print("\n" + "═" * 72)
    print(f"  EVIDENCE SUMMARY  [{metric_name}]")
    print("═" * 72)

    rows_out = []
    for noise_mode in ["ar", "nar"]:
        base = dm[(dm["config"] == "curr0_gate0") & (dm["data_mode"] == noise_mode)].set_index("dataset")
        for cfg, lbl in methods:
            sub = dm[(dm["config"] == cfg) & (dm["data_mode"] == noise_mode)]
            if sub.empty:
                continue
            macro_mean   = sub["metric_mean"].mean()
            macro_median = sub["metric_mean"].median()

            if cfg != "curr0_gate0":
                shared = sub.set_index("dataset").index.intersection(base.index)
                if len(shared) >= 4:
                    g_vals = sub.set_index("dataset").loc[shared, "metric_mean"].values
                    b_vals = base.loc[shared, "metric_mean"].values
                    _, p, r = wilcoxon_and_effect(g_vals, b_vals)
                    w, d, l = sign_test(g_vals, b_vals)
                else:
                    p, r, w, d, l = [float("nan")] * 5
            else:
                p, r, w, d, l = float("nan"), float("nan"), 0, 0, 0

            rows_out.append({
                "Mode":        noise_mode.upper(),
                "Method":      lbl,
                "Macro mean":  f"{macro_mean:.4f}",
                "Macro med.":  f"{macro_median:.4f}",
                "W/D/L":       f"{int(w)}/{int(d)}/{int(l)}" if not np.isnan(p) else "—",
                "p (Wilcoxon)": f"{p:.4f}" if not np.isnan(p) else "—",
                "|r|":         f"{abs(r):.3f}" if not np.isnan(r) else "—",
            })

    df_out = pd.DataFrame(rows_out)
    if latex:
        print(df_out.to_latex(index=False, escape=False))
    else:
        print(df_out.to_string(index=False))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metric", default="f1",
                        choices=["f1", "accuracy", "precision", "recall", "auc", "loss"],
                        help="Performance metric to analyse (default: f1).")
    parser.add_argument("--latex", action="store_true",
                        help="Print tables in LaTeX format.")
    parser.add_argument("--config", default="../config.yaml",
                        help="Path to config.yaml (default: ../config.yaml, i.e. the project "
                             "root — this script is meant to be run from analysis/), used to "
                             "read comparison_methods when --comparison-methods is omitted.")
    parser.add_argument("--comparison-methods", nargs="+", choices=METHOD_CHOICES, default=None,
                        help="Restrict every comparison in this script to these methods "
                             "(overrides config.yaml's comparison_methods). The 'Clean' "
                             "reference stays on regardless — see module docstring.")
    args = parser.parse_args()

    _SELECTED_METHODS = resolve_comparison_methods(args.comparison_methods, args.config)
    if _SELECTED_METHODS is not None:
        print(f"Restricting to comparison_methods: {_SELECTED_METHODS}")

    metric_name = {"f1": "F1", "accuracy": "Accuracy", "precision": "Precision",
                   "recall": "Recall", "auc": "AUC-ROC", "loss": "Loss"}[args.metric]

    print(f"\nDB: {DB_PATH.resolve()}  |  exists: {DB_PATH.exists()}")
    seed_df = load_seed_data(metric=args.metric)
    dm      = dataset_means(seed_df)
    dm      = broadcast_catboost_clean(dm)

    print("\nRunning analyses…")
    stats_summary(dm, metric_name)
    delta_table(dm, metric_name, latex=args.latex)
    evidence_summary(dm, metric_name, latex=args.latex)

    print("Generating plots…")
    plot_deltas(dm, metric_name)
    plot_noise_robustness(dm, metric_name)
    plot_gate_behaviour(dm)
    plot_training_efficiency(dm, metric_name)
    plot_rank_distribution(dm, metric_name)
    nemenyi_posthoc(dm, metric_name)

    print("\nDone. All plots saved to plots/")
