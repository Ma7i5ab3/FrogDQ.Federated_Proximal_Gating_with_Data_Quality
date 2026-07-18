#!/usr/bin/env python3
"""
elo_ratings.py — Bradley-Terry / Elo rating evaluation across methods.

Complements the rank-based Friedman/Nemenyi analysis in quail_analysis.py with
a pairwise-comparison rating. Every pair of methods that shares a (dataset,
seed) under a given noise mode is one "match": whichever has the higher
metric on that seed wins (a difference within --tie-eps counts as a draw,
split 0.5/0.5) — matches are decided at the individual-seed level rather than
on the cross-seed mean, so the per-seed noise contributes real match-to-match
variation instead of being averaged away before a single win/loss verdict is
made. All matches are pooled and fit in one shot with the Bradley-Terry
model — the exact MLE behind Elo, and the same approach LMSYS's Chatbot Arena
leaderboard uses to rank LLMs from pairwise battles — then rescaled onto the
familiar Elo scale (400 points per factor of 10 in win odds, centred at 1500).

Unlike a Friedman rank sum, Bradley-Terry tolerates partial method coverage
(a method that was never run on some dataset simply contributes no match
there, rather than forcing a complete-case dataset intersection across all
methods) and yields a continuous, probabilistic score: a 400-point gap
between two ratings means the higher-rated method is expected to win ~91% of
head-to-head dataset comparisons.

Ratings are bootstrapped by resampling *datasets* with replacement (cluster
bootstrap — every match drawn from the same dataset, including its several
per-seed matches, is correlated, so resampling individual matches would
understate uncertainty) to get a 95% CI per method.

Usage
-----
    python elo_ratings.py
    python elo_ratings.py --metric accuracy
    python elo_ratings.py --comparison-methods baseline gate saga catboost_dirty
    python elo_ratings.py --n-boot 1000 --latex
"""

import argparse
import re
import sys
import warnings
from itertools import combinations
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quail.comparison_methods import METHOD_CHOICES, resolve_comparison_methods, resolve_from_config_token

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
    r"_(?P<config>curr[01]_gate[01]|ag|saga|cp)$"
)
_RE_CATBOOST = re.compile(
    r"^(?P<dataset>.+)_(?P<data_mode>clean|ar|nar)_catboost$"
)

CONFIG_LABELS = {
    "curr0_gate0": "Standard prep",
    "curr1_gate0": "Curriculum",
    "curr0_gate1": "QuAIL",
    "curr1_gate1": "+ Quail + Curr",
    "saga":        "Saga++",
    "cp":          "CP prep",
    "ag":          "AutoGluon",
    "catboost_clean": "CatBoost Clean",
    "catboost_dirty":  "CatBoost Dirty",
}

COLORS = {
    "Standard prep":  "#aaaaaa",
    "QuAIL":          "#e74c3c",
    "Curriculum":     "#e07b39",
    "Saga++":         "#1abc9c",
    "CP prep":        "#f39c12",
    "CatBoost Clean": "#27ae60",
    "CatBoost Dirty": "#8e44ad",
}

ELO_BASE  = 1500.0
ELO_SCALE = 400.0 / np.log(10.0)  # Elo points per unit of natural-log BT strength

_SELECTED_METHODS: Optional[List[str]] = None


def _filter_tokens(tokens):
    """Filter a bare list of config tokens by the comparison_methods selection."""
    if _SELECTED_METHODS is None:
        return tokens
    return [t for t in tokens if resolve_from_config_token(t, data_mode="ar") in _SELECTED_METHODS]


Path("plots").mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (same per-seed shape as quail_analysis.py's load_seed_data)
# ─────────────────────────────────────────────────────────────────────────────

def load_seed_data(metric: str = "f1", field: Optional[str] = None) -> pd.DataFrame:
    """
    Load per-seed results from the best trial of every matched study.

    By default reads the `test_{metric}` performance field. Pass `field`
    to read a different per-seed scalar instead (e.g. "epochs_to_90pct"
    for the convergence-speed Elo) — the rest of the pipeline (matches,
    Bradley-Terry fit, bootstrap, plots) is agnostic to what "metric" means.
    """
    perf_key = field if field is not None else f"test_{metric}"
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
            rows.append({
                "dataset":    d["dataset"],
                "data_mode":  d["data_mode"],
                "model_type": d["model_type"],
                "config":     d["config"],
                "label":      label,
                "seed":       sr.get("seed"),
                "metric":     float(sr[perf_key]),
            })

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} seed rows | "
          f"{df['dataset'].nunique()} datasets | "
          f"{df['config'].nunique()} configs | "
          f"metric: {perf_key}")
    return df


def broadcast_catboost_clean(seed_df: pd.DataFrame) -> pd.DataFrame:
    """CatBoost-Clean has no AR/NAR variant; duplicate its per-seed rows into
    synthetic "ar"/"nar" rows, exactly like quail_analysis.py does at the
    dataset-mean level, so it can sit in the same per-noise-mode rating pool
    as every other method (each seed keeps its own clean-run metric)."""
    clean_rows = seed_df[seed_df["config"] == "catboost_clean"]
    if clean_rows.empty:
        return seed_df
    dupes = []
    for noise_mode in ["ar", "nar"]:
        dup = clean_rows.copy()
        dup["data_mode"] = noise_mode
        dupes.append(dup)
    return pd.concat([seed_df] + dupes, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Bradley-Terry / Elo fitting
# ─────────────────────────────────────────────────────────────────────────────

def build_matches(pivot: pd.DataFrame, methods: List[str], tie_eps: float = 1e-4,
                   lower_is_better: bool = False) -> pd.DataFrame:
    """
    From a (dataset, seed) x method metric pivot table (MultiIndex rows),
    build one row per match: every pair of methods present for a given
    (dataset, seed), whichever has the higher metric on that seed "wins"
    (difference within tie_eps counts as a 0.5/0.5 draw). Methods missing for
    a (dataset, seed) simply contribute no match there. `dataset` is carried
    through as its own column (not just the index) so bootstrap resampling
    can cluster matches by it.

    Set `lower_is_better=True` for metrics where a smaller value wins (e.g.
    epochs-to-convergence): the sign of the difference is flipped before the
    win/tie decision, so everything downstream (Bradley-Terry fit, ratings,
    win-rate bookkeeping) stays "higher strength = better" without change.
    """
    rows = []
    for (ds, seed), row in pivot.iterrows():
        present = [m for m in methods if pd.notna(row.get(m))]
        for a, b in combinations(present, 2):
            diff = row[a] - row[b]
            if lower_is_better:
                diff = -diff
            outcome = 0.5 if abs(diff) <= tie_eps else float(diff > 0)
            rows.append({"dataset": ds, "seed": seed, "i": a, "j": b, "outcome": outcome})
    return pd.DataFrame(rows, columns=["dataset", "seed", "i", "j", "outcome"])


def fit_bradley_terry(matches: pd.DataFrame, methods: List[str], reg: float = 0.5) -> np.ndarray:
    """
    MLE Bradley-Terry log-strengths via L-BFGS on the exact (possibly
    fractional, for ties) pairwise log-likelihood:

        outcome * log(sigmoid(beta_i - beta_j)) + (1 - outcome) * log(sigmoid(beta_j - beta_i))

    Ridge-regularised (`reg`) so an always-winning/always-losing method
    doesn't drive its strength to +-infinity. The likelihood is invariant to
    adding a constant to every beta, so the fit is mean-centred afterwards to
    fix an arbitrary origin without changing any pairwise win probability.
    """
    idx = {m: k for k, m in enumerate(methods)}
    n = len(methods)
    if matches.empty:
        return np.zeros(n)

    I = matches["i"].map(idx).to_numpy()
    J = matches["j"].map(idx).to_numpy()
    O = matches["outcome"].to_numpy()

    def nll_grad(beta):
        d = beta[I] - beta[J]
        p = expit(d)
        eps = 1e-12
        nll = -(O * np.log(p + eps) + (1 - O) * np.log(1 - p + eps)).sum() \
              + 0.5 * reg * np.sum(beta ** 2)
        resid = O - p  # positive => i under-predicted to win => push beta_i up
        grad = np.zeros(n)
        np.add.at(grad, I, -resid)
        np.add.at(grad, J, resid)
        grad += reg * beta
        return nll, grad

    res = minimize(nll_grad, np.zeros(n), jac=True, method="L-BFGS-B")
    beta = res.x
    beta -= beta.mean()
    return beta


def bootstrap_ratings(
    pivot: pd.DataFrame, methods: List[str], tie_eps: float, reg: float,
    n_boot: int, rng: np.random.Generator, lower_is_better: bool = False,
) -> np.ndarray:
    """
    Cluster-bootstrap (resample datasets with replacement) Bradley-Terry
    fits. `pivot` is indexed by (dataset, seed); resampling a dataset pulls
    in *all* of its seed rows together (a dataset drawn twice contributes all
    of its seed-level matches twice), which is what keeps the per-seed
    matches from being treated as independent draws across datasets in the
    uncertainty estimate. Returns an (n_boot, n_methods) array of Elo
    ratings; a method absent from a given resample is NaN in that row.
    """
    datasets = pivot.index.get_level_values("dataset").unique().to_numpy()
    blocks = {ds: pivot.xs(ds, level="dataset", drop_level=False) for ds in datasets}
    out = np.full((n_boot, len(methods)), np.nan)
    for b in range(n_boot):
        sample = rng.choice(datasets, size=len(datasets), replace=True)
        sub = pd.concat([blocks[ds] for ds in sample])
        matches = build_matches(sub, methods, tie_eps=tie_eps, lower_is_better=lower_is_better)
        present = set(matches["i"]).union(matches["j"])
        beta = fit_bradley_terry(matches, methods, reg=reg)
        ratings = ELO_BASE + beta * ELO_SCALE
        ratings = np.array([r if m in present else np.nan for r, m in zip(ratings, methods)])
        out[b] = ratings
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Reporting: ratings table + plots
# ─────────────────────────────────────────────────────────────────────────────

def compute_and_report(
    seed_df: pd.DataFrame, metric_name: str, methods_order: List[str],
    tie_eps: float, reg: float, n_boot: int, seed: int, latex: bool,
    lower_is_better: bool = False, plot_prefix: str = "gate_elo",
) -> None:
    print("\n" + "═" * 72)
    print(f"  BRADLEY-TERRY / ELO RATINGS  [{metric_name}]")
    print("═" * 72)

    rng = np.random.default_rng(seed)
    all_tables = []

    for noise_mode in ["ar", "nar"]:
        sub = seed_df[seed_df["data_mode"] == noise_mode]
        pivot = sub.pivot_table(index=["dataset", "seed"], columns="config",
                                 values="metric", aggfunc="mean")
        available = [c for c in methods_order if c in pivot.columns]
        pivot = pivot[available]

        n_datasets = pivot.index.get_level_values("dataset").nunique()
        if pivot.empty or len(available) < 2:
            print(f"\n  {noise_mode.upper()}: insufficient data, skipping.")
            continue

        matches = build_matches(pivot, available, tie_eps=tie_eps, lower_is_better=lower_is_better)
        beta = fit_bradley_terry(matches, available, reg=reg)
        point_ratings = ELO_BASE + beta * ELO_SCALE

        boot = bootstrap_ratings(pivot, available, tie_eps, reg, n_boot, rng,
                                  lower_is_better=lower_is_better)
        lo = np.nanpercentile(boot, 2.5, axis=0)
        hi = np.nanpercentile(boot, 97.5, axis=0)
        med = np.nanmedian(boot, axis=0)

        labels = [CONFIG_LABELS.get(c, c) for c in available]

        n_matches = {m: int(((matches["i"] == m) | (matches["j"] == m)).sum()) for m in available}
        n_wins = {}
        for m in available:
            as_i = matches[matches["i"] == m]["outcome"].sum()
            as_j = (1.0 - matches[matches["j"] == m]["outcome"]).sum()
            n_wins[m] = as_i + as_j

        rows_out = []
        for c, lbl, r, lo_, hi_, md in zip(available, labels, point_ratings, lo, hi, med):
            played = n_matches[c]
            win_rate = n_wins[c] / played if played else float("nan")
            rows_out.append({
                "Mode":       noise_mode.upper(),
                "Method":     lbl,
                "Elo":        round(float(r), 1),
                "95% CI low": round(float(lo_), 1),
                "95% CI high": round(float(hi_), 1),
                "Boot median": round(float(md), 1),
                "Matches":    played,
                "Win rate":   round(float(win_rate), 3) if played else float("nan"),
            })
        rows_out.sort(key=lambda r: -r["Elo"])
        df_out = pd.DataFrame(rows_out)
        all_tables.append(df_out)

        print(f"\n  {noise_mode.upper()}  (N={n_datasets} datasets, "
              f"{len(available)} methods, {len(matches)} seed-level matches, "
              f"{n_boot} bootstrap resamples)")
        if latex:
            print(df_out.to_latex(index=False))
        else:
            print(df_out.to_string(index=False))

        _plot_ratings(df_out, noise_mode, metric_name, plot_prefix=plot_prefix)
        _plot_win_probabilities(available, labels, beta, noise_mode, metric_name, plot_prefix=plot_prefix)

    print()
    return all_tables


def _plot_ratings(df_out: pd.DataFrame, noise_mode: str, metric_name: str,
                   plot_prefix: str = "gate_elo") -> None:
    """Bar chart of Elo ratings with 95% bootstrap CI, one file per noise mode."""
    fig, ax = plt.subplots(figsize=(7, 5))
    df_sorted = df_out.sort_values("Elo", ascending=True)
    y = np.arange(len(df_sorted))
    err_low = df_sorted["Elo"] - df_sorted["95% CI low"]
    err_high = df_sorted["95% CI high"] - df_sorted["Elo"]
    colors = [COLORS.get(lbl, "#888") for lbl in df_sorted["Method"]]

    ax.barh(y, df_sorted["Elo"], color=colors, alpha=0.85, height=0.6)
    ax.errorbar(df_sorted["Elo"], y, xerr=[err_low, err_high],
                fmt="none", ecolor="black", elinewidth=1.2, capsize=4)
    ax.set_yticks(y)
    ax.set_yticklabels(df_sorted["Method"], fontsize=9)
    ax.axvline(ELO_BASE, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    label_x = df_sorted["95% CI high"].max() * 1.02
    for yi, elo in zip(y, df_sorted["Elo"]):
        ax.text(label_x, yi, f"{elo:.0f}", va="center", ha="left", fontsize=8)
    ax.set_xlim(right=label_x * 1.08)
    ax.set_xlabel("Elo rating (Bradley-Terry MLE, bootstrap 95% CI)", fontsize=9)
    ax.set_title(
        f"{noise_mode.upper()} — Bradley-Terry / Elo ratings  [{metric_name}]",
        fontsize=11, fontweight="bold",
    )
    ax.grid(axis="x", linewidth=0.3, alpha=0.5)
    plt.tight_layout()
    out = f"plots/{plot_prefix}_ratings_{noise_mode}.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


def _plot_win_probabilities(available, labels, beta, noise_mode, metric_name,
                             plot_prefix: str = "gate_elo") -> None:
    """Heatmap of P(row beats column) implied by the fitted BT strengths."""
    n = len(available)
    probs = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            probs[a, b] = expit(beta[a] - beta[b]) if a != b else np.nan

    fig, ax = plt.subplots(figsize=(1.1 * n + 2, 1.1 * n + 1))
    im = ax.imshow(probs, cmap="RdBu_r", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="P(row beats column)", pad=0.02, fraction=0.046)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            val = probs[a, b]
            ax.text(b, a, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(val - 0.5) > 0.25 else "black")
    ax.set_title(
        f"{noise_mode.upper()} — Implied pairwise win probability  [{metric_name}]",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    out = f"plots/{plot_prefix}_win_prob_{noise_mode}.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Training-efficiency Elo (epochs_to_XXpct), one rating pool per threshold
# ─────────────────────────────────────────────────────────────────────────────

def compute_epochs_efficiency_elo(
    pct: int, methods_order: List[str],
    tie_eps: float, reg: float, n_boot: int, seed: int, latex: bool,
) -> None:
    """
    Elo ratings for training-convergence speed at a given threshold: for
    each (dataset, seed), whichever method needed fewer epochs to reach
    `pct`% of its own final validation metric wins that match. Reuses the
    exact same per-seed Bradley-Terry mechanism as the performance Elo in
    compute_and_report — only the loaded field (`epochs_to_{pct}pct`
    instead of `test_{metric}`) and the win polarity (lower is better)
    differ.
    """
    field = f"epochs_to_{pct}pct"
    seed_df = load_seed_data(field=field)
    seed_df = broadcast_catboost_clean(seed_df)

    compute_and_report(
        seed_df, metric_name=f"Epochs to {pct}%", methods_order=methods_order,
        tie_eps=tie_eps, reg=reg, n_boot=n_boot, seed=seed, latex=latex,
        lower_is_better=True, plot_prefix=f"gate_elo_epochs{pct}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metric", default="f1",
                         choices=["f1", "accuracy", "precision", "recall", "auc", "loss"],
                         help="Performance metric to rate methods on (default: f1).")
    parser.add_argument("--latex", action="store_true",
                         help="Print rating tables in LaTeX format.")
    parser.add_argument("--config", default="../config.yaml",
                         help="Path to config.yaml (default: ../config.yaml), used to read "
                              "comparison_methods when --comparison-methods is omitted.")
    parser.add_argument("--comparison-methods", nargs="+", choices=METHOD_CHOICES, default=None,
                         help="Restrict the rating pool to these methods (overrides "
                              "config.yaml's comparison_methods).")
    parser.add_argument("--tie-eps", type=float, default=1e-4,
                         help="Metric difference below which a dataset counts as a draw "
                              "(default: 1e-4, matching quail_analysis.py's sign_test).")
    parser.add_argument("--reg", type=float, default=0.5,
                         help="Ridge regularisation strength on Bradley-Terry strengths "
                              "(default: 0.5; guards against undefeated/always-losing methods).")
    parser.add_argument("--n-boot", type=int, default=500,
                         help="Number of dataset-cluster bootstrap resamples for the "
                              "95%% CI (default: 500).")
    parser.add_argument("--seed", type=int, default=42,
                         help="Bootstrap RNG seed (default: 42, matching the project convention).")
    parser.add_argument("--efficiency-pcts", type=int, nargs="+", default=[90, 95, 99],
                         help="Convergence thresholds to also rate on training-efficiency "
                              "Elo (epochs_to_XXpct, lower is better). Default: 90 95 99.")
    parser.add_argument("--skip-efficiency-elo", action="store_true",
                         help="Skip the epochs_to_XXpct efficiency Elo ratings entirely.")
    args = parser.parse_args()

    _SELECTED_METHODS = resolve_comparison_methods(args.comparison_methods, args.config)
    if _SELECTED_METHODS is not None:
        print(f"Restricting to comparison_methods: {_SELECTED_METHODS}")

    metric_name = {"f1": "F1", "accuracy": "Accuracy", "precision": "Precision",
                   "recall": "Recall", "auc": "AUC-ROC", "loss": "Loss"}[args.metric]

    print(f"\nDB: {DB_PATH.resolve()}  |  exists: {DB_PATH.exists()}")
    seed_df = load_seed_data(metric=args.metric)
    seed_df = broadcast_catboost_clean(seed_df)

    methods_order = _filter_tokens([
        "curr0_gate0", "curr0_gate1", "curr1_gate0", "saga", "cp",
        "catboost_clean", "catboost_dirty",
    ])

    compute_and_report(
        seed_df, metric_name, methods_order,
        tie_eps=args.tie_eps, reg=args.reg, n_boot=args.n_boot,
        seed=args.seed, latex=args.latex,
    )

    if not args.skip_efficiency_elo:
        for pct in args.efficiency_pcts:
            compute_epochs_efficiency_elo(
                pct, methods_order,
                tie_eps=args.tie_eps, reg=args.reg, n_boot=args.n_boot,
                seed=args.seed, latex=args.latex,
            )

    print("Done. All plots saved to plots/")
