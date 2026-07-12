#!/usr/bin/env python3
"""
latex_evidence_table.py — Cross-metric per-dataset LaTeX evidence table.

For every dataset (rows) and every performance metric (column groups: F1,
Accuracy, Precision, Recall, AUC-ROC), reports each comparison_methods
baseline as one compact column, stacked two lines per cell:

  - Mean (analytic 95% CI across seeds: mean +/- 1.96 * std / sqrt(n))
  - Win/Defeat% (effective win count / defeat count) underneath, in smaller
    type, computed over the same pairwise per-seed "matches"
    elo_ratings.py's Bradley-Terry model is fit on (every pair of methods
    sharing a (dataset, seed): whichever has the higher metric wins; a
    difference within --tie-eps is a draw, split 0.5/0.5), restricted to
    that one dataset and pooled across every other selected competitor.
    "clean" (see NON_COMPETING_METHODS) never enters the match pool — same
    as elo_ratings.py's own default rating pool — so it has no win/defeat
    line of its own, and it never dilutes anyone else's.

One table is produced per noise mode (AR, NAR); pass --split-by-metric to
instead produce one (much narrower) table per (noise mode, metric). "clean"
and "catboost_clean" have no AR/NAR variant of their own, so their per-seed
scores are broadcast into both noise-mode match pools, exactly like
gate_analysis.py/elo_ratings.py already do for catboost_clean.

In every table, the "clean" column (noise-free reference) is always bolded
and shaded light grey (\\cellcolor{gray!15}); the best mean per (dataset,
metric) row among the remaining, actually-competing methods is bolded and
shaded green (\\cellcolor{green!20}).

Tables render as a table*/tabularx block sized to \\textwidth exactly (no
\\resizebox hack), matching a typical paper-ready results table: Y-columns
sharing the page width evenly, small+small-caps body text, and a single
vertical rule marking each top-level group. The host document's preamble
must provide:
    \\usepackage{tabularx}
    \\usepackage{booktabs}
    \\usepackage[table]{xcolor}
    \\newcolumntype{Y}{>{\\centering\\arraybackslash}X}
and either an ICML/NeurIPS-style class defining the "sc" environment, or
swap \\begin{sc}...\\end{sc} for \\scshape in the generated .tex. table* spans
both columns of a twocolumn document; use --split-by-metric so each table is
narrow enough to actually fit (the combined, all-metrics table is very wide
regardless of styling — it's meant for reference/appendix use, not print).

Usage
-----
    python latex_evidence_table.py
    python latex_evidence_table.py --metrics f1 accuracy auc
    python latex_evidence_table.py --comparison-methods clean baseline gate saga
    python latex_evidence_table.py --split-by-metric
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from frogdq.comparison_methods import METHOD_CHOICES, resolve_comparison_methods

sys.path.insert(0, str(Path(__file__).resolve().parent))
from elo_ratings import broadcast_catboost_clean, build_matches, load_seed_data

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
REPORTS_DIR = HERE / "reports"

METRIC_CHOICES = ["f1", "accuracy", "precision", "recall", "auc"]
METRIC_LABELS = {
    "f1": "F1", "accuracy": "Accuracy", "precision": "Precision",
    "recall": "Recall", "auc": "AUC-ROC",
}

# canonical comparison_methods key (frogdq.comparison_methods.METHOD_CHOICES)
# -> study-name "config" token used in the Optuna DB. "clean" has no config
# token of its own (it's curr0_gate0 at data_mode == "clean"); it is given a
# synthetic token here so it can sit as its own column alongside "baseline"
# (curr0_gate0 at data_mode in {ar, nar}) without being confused for it —
# see broadcast_clean_reference() below.
CANONICAL_TO_TOKEN: Dict[str, str] = {
    "clean":           "clean_ref",
    "baseline":        "curr0_gate0",
    "curriculum":      "curr1_gate0",
    "gate":            "curr0_gate1",
    "gate_curriculum": "curr1_gate1",
    "autogluon":       "ag",
    "saga":            "saga",
    "cp":              "cp",
    "baseline_zero":   "baseline_zero",
    "knn":             "knn",
    "catboost_clean":  "catboost_clean",
    "catboost_dirty":  "catboost_dirty",
}

CANONICAL_LABELS: Dict[str, str] = {
    "clean":           "Clean",
    "baseline":        "Baseline",
    "curriculum":      "Curriculum",
    "gate":            "QuAIL",
    "gate_curriculum": "+ Gate + Curr",
    "autogluon":       "AutoGluon",
    "saga":            "Saga++",
    "cp":              "CP prep",
    "baseline_zero":   "Baseline (zero)",
    "knn":             "KNN",
    "catboost_clean":  "CatBoost Clean",
    "catboost_dirty":  "CatBoost Dirty",
}

# The noise-free "clean" reference is an upper-bound sanity check, not a
# competitor: it is excluded from the pairwise-match pool entirely (so it has
# no win/defeat cell of its own, and nobody else's win/defeat rate is diluted
# by matches against it), exactly like elo_ratings.py's own default rating
# pool excludes it. It is also excluded from the "best of the actual methods"
# mean comparison used for green-cell highlighting in build_latex().
NON_COMPETING_METHODS = {"clean"}


def broadcast_clean_reference(seed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Duplicate the noise-free "clean" reference (config == curr0_gate0,
    data_mode == "clean") into synthetic "ar"/"nar" rows tagged with the
    config token "clean_ref", so it can sit as its own column in the AR/NAR
    match pool without colliding with "baseline" (curr0_gate0 at data_mode
    in {ar, nar}). Mirrors broadcast_catboost_clean() in elo_ratings.py.
    """
    clean_rows = seed_df[(seed_df["config"] == "curr0_gate0") & (seed_df["data_mode"] == "clean")]
    if clean_rows.empty:
        return seed_df
    dupes = []
    for noise_mode in ["ar", "nar"]:
        dup = clean_rows.copy()
        dup["data_mode"] = noise_mode
        dup["config"] = "clean_ref"
        dupes.append(dup)
    return pd.concat([seed_df] + dupes, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_cells(metrics: List[str], methods: List[str], noise_mode: str,
                   tie_eps: float = 1e-4):
    """
    Returns (datasets, cells): `datasets` is the sorted list of datasets with
    any data under this noise mode; `cells[(dataset, metric, method)]` is a
    dict with mean/ci/win_pct/win_n/loss_pct/loss_n/n_matches, or None if
    that method has no data for that (dataset, metric) under this noise mode.

    Methods in NON_COMPETING_METHODS (i.e. "clean") still get a mean/CI, but
    are left out of the match pool entirely: their win_pct/loss_pct come back
    NaN, and — since they never appear in any match — every other method's
    win/defeat counts are unaffected by them, matching elo_ratings.py's own
    default rating pool (which excludes the clean-data reference the same
    way).
    """
    tokens = [CANONICAL_TO_TOKEN[m] for m in methods]
    match_methods = [m for m in methods if m not in NON_COMPETING_METHODS]
    match_tokens = [CANONICAL_TO_TOKEN[m] for m in match_methods]
    cells = {}
    all_datasets = set()

    for metric in metrics:
        seed_df = load_seed_data(metric=metric)
        seed_df = broadcast_catboost_clean(seed_df)
        seed_df = broadcast_clean_reference(seed_df)
        sub = seed_df[seed_df["data_mode"] == noise_mode]

        present_tokens = [t for t in tokens if t in sub["config"].unique()]
        sub_f = sub[sub["config"].isin(present_tokens)]
        if sub_f.empty:
            continue
        all_datasets.update(sub_f["dataset"].unique())

        agg = (
            sub_f.groupby(["dataset", "config"])["metric"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        present_match_tokens = [t for t in match_tokens if t in present_tokens]
        if len(present_match_tokens) >= 2:
            match_sub = sub_f[sub_f["config"].isin(present_match_tokens)]
            pivot = match_sub.pivot_table(index=["dataset", "seed"], columns="config",
                                           values="metric", aggfunc="mean")
            pivot = pivot[present_match_tokens]
            matches = build_matches(pivot, present_match_tokens, tie_eps=tie_eps)
        else:
            matches = pd.DataFrame(columns=["dataset", "seed", "i", "j", "outcome"])

        for ds in sub_f["dataset"].unique():
            ds_matches = matches[matches["dataset"] == ds] if not matches.empty else matches
            for method, token in zip(methods, tokens):
                row = agg[(agg["dataset"] == ds) & (agg["config"] == token)]
                if row.empty:
                    cells[(ds, metric, method)] = None
                    continue
                mean = float(row["mean"].iloc[0])
                std_val = row["std"].iloc[0]
                n = int(row["count"].iloc[0])
                std = float(std_val) if pd.notna(std_val) else 0.0
                ci = 1.96 * std / np.sqrt(n) if n > 1 else float("nan")

                as_i = ds_matches[ds_matches["i"] == token]["outcome"]
                as_j = ds_matches[ds_matches["j"] == token]["outcome"]
                wins = float(as_i.sum() + (1.0 - as_j).sum())
                losses = float((1.0 - as_i).sum() + as_j.sum())
                n_matches = len(as_i) + len(as_j)
                win_pct = wins / n_matches if n_matches else float("nan")
                loss_pct = losses / n_matches if n_matches else float("nan")

                cells[(ds, metric, method)] = {
                    "mean": mean, "ci": ci,
                    "win_pct": win_pct, "win_n": wins,
                    "loss_pct": loss_pct, "loss_n": losses,
                    "n_matches": n_matches,
                }

    return sorted(all_datasets), cells


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX rendering
# ─────────────────────────────────────────────────────────────────────────────

# Tie tolerance (absolute, in metric units) when deciding which method(s)
# have the "best" mean for a (dataset, metric) row.
BEST_TIE_ATOL = 1e-9

BEST_CELL_COLOR = "green!20"
CLEAN_CELL_COLOR = "gray!15"


def _fmt_cell(c: Optional[dict], color: Optional[str] = None) -> str:
    """
    One method's result as a single compact two-line cell: the mean +/- 95%
    CI on top, and a small win/defeat line underneath. Halves the column
    count vs. two separate Mean / Win-Defeat columns, which is what actually
    made the table too wide to be readable.

    The line break between them MUST be \\newline, not \\\\: inside a
    tabularx "Y" (paragraph-mode) column, a bare \\\\ is not scoped to the
    cell — it's tabular's row terminator, so it would end the whole table
    row early rather than just wrapping within this one cell.
    """
    if c is None or np.isnan(c["mean"]):
        return "--"
    num = f"{c['mean']:.3f}" if np.isnan(c["ci"]) else f"{c['mean']:.3f}\\pm{c['ci']:.3f}"
    mean_line = f"$\\mathbf{{{num}}}$" if color else f"${num}$"
    if np.isnan(c["win_pct"]):
        wd_line = ""
    else:
        wd_line = (f"{{\\scriptsize {c['win_pct'] * 100:.0f}/{c['loss_pct'] * 100:.0f}\\% "
                   f"({c['win_n']:g}/{c['loss_n']:g})}}")
    text = f"{mean_line}\\newline {wd_line}" if wd_line else mean_line
    return f"\\cellcolor{{{color}}}{text}" if color else text


def _best_methods(cells: dict, ds: str, metric: str, methods: List[str]) -> set:
    """Method(s) with the highest mean for this (dataset, metric) row, within
    BEST_TIE_ATOL of each other. Empty set if no method has data here."""
    means = {
        m: cells[(ds, metric, m)]["mean"]
        for m in methods
        if cells.get((ds, metric, m)) is not None and not np.isnan(cells[(ds, metric, m)]["mean"])
    }
    if not means:
        return set()
    best = max(means.values())
    return {m for m, v in means.items() if abs(v - best) <= BEST_TIE_ATOL}


def build_latex(datasets: List[str], cells: dict, metrics: List[str],
                 methods: List[str], noise_mode: str) -> str:
    """
    Renders a table*/tabularx table sized to fit \\textwidth exactly (no
    \\resizebox), following the same design as a paper-ready results table:
    Y-columns that share the page width evenly, small+sc body text, and a
    single vertical rule marking each top-level group instead of cmidrules
    under every header row.

    The top-level grouping — the thing that gets the "|" column separators —
    is whichever dimension actually has more than one value here: metrics
    when this table spans several (the combined, non-split table), methods
    when it's already been split down to a single metric (--split-by-metric).
    Showing a metric-spanning header row when there's only one metric would
    just be visual clutter, so it's dropped in that case.
    """
    n_methods = len(methods)
    group_by_metric = len(metrics) > 1

    # One column per method (mean and win/defeat are stacked into a single
    # two-line cell by _fmt_cell — see its docstring for why).
    if group_by_metric:
        col_groups = [" ".join(["Y"] * n_methods) for _ in metrics]
    else:
        col_groups = ["Y" for _ in methods]
    col_spec = "@{} l " + " | ".join(col_groups)

    lines = [
        "% Requires \\usepackage{tabularx}, \\usepackage{booktabs}, and \\usepackage[table]{xcolor}",
        "% in the document preamble, plus a centered X column type:",
        "%   \\newcolumntype{Y}{>{\\centering\\arraybackslash}X}",
        "% \\begin{sc} assumes an ICML/NeurIPS-style class providing that environment;",
        "% otherwise replace \\begin{sc}...\\end{sc} with \\scshape.",
        "\\begin{table*}[t]",
        "\\caption{Per-dataset evidence summary --- " + noise_mode.upper() + " noise. "
        "Each cell reports the seed mean $\\pm$ 95\\% CI on top and, below it, the win/defeat "
        "rate (effective win/defeat count) over the pairwise per-seed matches among the "
        "compared methods, as defined in elo\\_ratings.py. Clean (noise-free reference) is "
        "excluded from the match pool (no win/defeat line) and is bolded and shaded "
        "\\colorbox{" + CLEAN_CELL_COLOR + "}{grey}; the best mean per row among the remaining "
        "(actual) methods is bolded and shaded \\colorbox{" + BEST_CELL_COLOR + "}{green}.}",
        f"\\label{{tab:evidence_{noise_mode}}}",
        "\\begin{center}",
        "\\begin{small}",
        "\\begin{sc}",
        f"\\begin{{tabularx}}{{\\textwidth}}{{{col_spec}}}",
        "\\toprule",
    ]

    if group_by_metric:
        # Header row 1: metric groups (one "|"-separated block per metric).
        header1 = ["Dataset"]
        for metric in metrics:
            header1.append(f"\\multicolumn{{{n_methods}}}{{c}}{{{METRIC_LABELS[metric]}}}")
        lines.append(" & ".join(header1) + " \\\\")
        lines.append("\\midrule")

        # Header row 2: method names (one per column, unseparated within a
        # metric block — the metric-level "|" above already groups them).
        header2 = [""]
        for _ in metrics:
            for method in methods:
                header2.append(CANONICAL_LABELS[method])
        lines.append(" & ".join(header2) + " \\\\")
    else:
        # Only one metric: method names are the top (and only) header row.
        header1 = ["Dataset"] + [CANONICAL_LABELS[method] for method in methods]
        lines.append(" & ".join(header1) + " \\\\")
    lines.append("\\midrule")

    for ds in datasets:
        row = [ds.replace("_", "\\_")]
        for metric in metrics:
            comparable = [m for m in methods if m not in NON_COMPETING_METHODS]
            best = _best_methods(cells, ds, metric, comparable)
            for method in methods:
                c = cells.get((ds, metric, method))
                if method in NON_COMPETING_METHODS:
                    color = CLEAN_CELL_COLOR
                elif method in best:
                    color = BEST_CELL_COLOR
                else:
                    color = None
                row.append(_fmt_cell(c, color=color))
        lines.append(" & ".join(row) + " \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabularx}",
        "\\end{sc}",
        "\\end{small}",
        "\\end{center}",
        "\\end{table*}",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--metrics", nargs="+", default=METRIC_CHOICES, choices=METRIC_CHOICES,
        help="Performance metrics to report as column groups (default: all five).",
    )
    parser.add_argument(
        "--config", default="../config.yaml",
        help="Path to config.yaml (default: ../config.yaml), used to read "
             "comparison_methods when --comparison-methods is omitted.",
    )
    parser.add_argument(
        "--comparison-methods", nargs="+", choices=METHOD_CHOICES, default=None,
        help="Restrict the table's subcolumns to these methods (overrides "
             "config.yaml's comparison_methods).",
    )
    parser.add_argument(
        "--tie-eps", type=float, default=1e-4,
        help="Metric difference below which a seed-level match counts as a draw "
             "(default: 1e-4, matching elo_ratings.py).",
    )
    parser.add_argument(
        "--split-by-metric", action="store_true",
        help="Emit one table per (noise mode, metric) instead of one wide table per "
             "noise mode spanning all --metrics. Produces narrower, more printable tables.",
    )
    args = parser.parse_args()

    selected = resolve_comparison_methods(args.comparison_methods, args.config)
    methods = selected if selected is not None else list(CANONICAL_LABELS.keys())
    methods = [m for m in METHOD_CHOICES if m in methods and m in CANONICAL_TO_TOKEN]
    if not methods:
        print("No known comparison_methods selected; nothing to do.")
        return
    print(f"Comparison methods: {methods}")

    REPORTS_DIR.mkdir(exist_ok=True)

    for noise_mode in ["ar", "nar"]:
        datasets, cells = collect_cells(args.metrics, methods, noise_mode, tie_eps=args.tie_eps)
        if not datasets:
            print(f"\n  {noise_mode.upper()}: no data, skipping.")
            continue

        metric_groups = [[m] for m in args.metrics] if args.split_by_metric else [args.metrics]
        for group in metric_groups:
            group_datasets = [ds for ds in datasets
                               if any(cells.get((ds, m, meth)) is not None
                                      for m in group for meth in methods)]
            if not group_datasets:
                continue
            latex = build_latex(group_datasets, cells, group, methods, noise_mode)
            suffix = f"_{group[0]}" if args.split_by_metric else ""
            label = "+".join(m.upper() for m in group) if args.split_by_metric else "all metrics"
            print(f"\n% ── {noise_mode.upper()} evidence table [{label}] ({len(group_datasets)} datasets) ──")
            print(latex)
            out_path = REPORTS_DIR / f"evidence_table_{noise_mode}{suffix}.tex"
            out_path.write_text(latex)
            print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
