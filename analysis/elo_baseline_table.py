#!/usr/bin/env python3
"""
elo_baseline_table.py — Metric x Baseline Elo LaTeX table (CCAR / CNAR).

Builds two compact LaTeX tables — one for CCAR (data_mode == "ar", Corruption
Completely At Random) and one for CNAR (data_mode == "nar", Corruption Not At
Random) — with one row per performance metric (F1, Accuracy, Recall,
Precision, ROC-AUC) and one column per baseline (Baseline, Saga++, CP,
Curriculum, QuAIL). Each cell is that method's Bradley-Terry Elo rating (fit
per (metric, noise mode) exactly like elo_ratings.py's own default rating
pool) together with its dataset-cluster-bootstrap 95% CI. The best (highest
point-estimate Elo) method per row is bolded and shaded green.

Usage
-----
    python elo_baseline_table.py
    python elo_baseline_table.py --n-boot 1000
    python elo_baseline_table.py --comparison-methods baseline gate saga
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from elo_ratings import (
    ELO_BASE, ELO_SCALE, DB_PATH,
    load_seed_data, broadcast_catboost_clean,
    build_matches, fit_bradley_terry, bootstrap_ratings,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from frogdq.comparison_methods import METHOD_CHOICES, resolve_comparison_methods, resolve_from_config_token

HERE = Path(__file__).resolve().parent
REPORTS_DIR = HERE / "reports"

# canonical study "config" token -> column label, in the display order
# requested: baseline, saga++, cp, curriculum, quail.
METHODS_ORDER = ["curr0_gate0", "saga", "cp", "curr1_gate0", "curr0_gate1"]
METHOD_LABELS = {
    "curr0_gate0": "Baseline",
    "saga":        "Saga++",
    "cp":          "CP",
    "curr1_gate0": "Curriculum",
    "curr0_gate1": "QuAIL",
}

METRICS_ORDER = ["f1", "accuracy", "recall", "precision", "auc"]
METRIC_LABELS = {
    "f1": "F1", "accuracy": "Accuracy", "recall": "Recall",
    "precision": "Precision", "auc": "ROC-AUC",
}

NOISE_MODE_LABELS = {"ar": "CCAR", "nar": "CNAR"}

BEST_CELL_COLOR = "green!20"


def _filter_methods(methods_order, selected):
    """Restrict METHODS_ORDER to a comparison_methods selection, exactly like
    elo_ratings.py's _filter_tokens (config tokens resolved at data_mode="ar",
    which is safe here since none of these tokens are the clean-only token)."""
    if selected is None:
        return methods_order
    return [t for t in methods_order if resolve_from_config_token(t, data_mode="ar") in selected]


def compute_ratings(metrics, methods_order, tie_eps, reg, n_boot, seed):
    """
    Returns ratings[(metric, noise_mode, method)] = (point, lo, hi), fitting
    one independent Bradley-Terry model per (metric, noise_mode) pair, restricted
    to `methods_order`, exactly like elo_ratings.py's compute_and_report does
    for its own (larger) default method pool.
    """
    ratings = {}
    rng = np.random.default_rng(seed)

    for metric in metrics:
        seed_df = load_seed_data(metric=metric)
        seed_df = broadcast_catboost_clean(seed_df)

        for noise_mode in ["ar", "nar"]:
            sub = seed_df[seed_df["data_mode"] == noise_mode]
            pivot = sub.pivot_table(index=["dataset", "seed"], columns="config",
                                     values="metric", aggfunc="mean")
            available = [c for c in methods_order if c in pivot.columns]
            pivot = pivot[available]
            if pivot.empty or len(available) < 2:
                continue

            matches = build_matches(pivot, available, tie_eps=tie_eps)
            beta = fit_bradley_terry(matches, available, reg=reg)
            point = ELO_BASE + beta * ELO_SCALE

            boot = bootstrap_ratings(pivot, available, tie_eps, reg, n_boot, rng)
            lo = np.nanpercentile(boot, 2.5, axis=0)
            hi = np.nanpercentile(boot, 97.5, axis=0)

            for m, p, l, h in zip(available, point, lo, hi):
                ratings[(metric, noise_mode, m)] = (float(p), float(l), float(h))

    return ratings


def _fmt_cell(cell, color=None):
    if cell is None:
        return "--"
    p, lo, hi = cell
    num = f"{p:.0f} \\; [{lo:.0f}, {hi:.0f}]"
    text = f"$\\mathbf{{{num}}}$" if color else f"${num}$"
    return f"\\cellcolor{{{color}}}{text}" if color else text


def build_latex(ratings, noise_mode, metrics_order, methods_order) -> str:
    n_methods = len(methods_order)
    col_spec = "@{} l " + " ".join(["Y"] * n_methods)

    lines = [
        "% Requires \\usepackage{tabularx}, \\usepackage{booktabs}, and \\usepackage[table]{xcolor}",
        "% in the document preamble, plus a centered X column type:",
        "%   \\newcolumntype{Y}{>{\\centering\\arraybackslash}X}",
        "% \\begin{sc} assumes an ICML/NeurIPS-style class providing that environment;",
        "% otherwise replace \\begin{sc}...\\end{sc} with \\scshape.",
        "\\begin{table}[t]",
        "\\caption{Bradley-Terry / Elo rating per metric --- "
        + NOISE_MODE_LABELS[noise_mode] + " ("
        + ("Corruption Completely At Random" if noise_mode == "ar" else "Corruption Not At Random")
        + "). Each cell reports the point-estimate Elo rating and its dataset-cluster "
          "bootstrap 95\\% CI, $\\text{Elo}\\;[\\text{CI}_{lo}, \\text{CI}_{hi}]$, fit "
          "independently per (metric, method pool) exactly as in elo\\_ratings.py. The best "
          "(highest) rating per row is bolded and shaded \\colorbox{" + BEST_CELL_COLOR + "}{green}.}",
        f"\\label{{tab:elo_{noise_mode}}}",
        "\\begin{center}",
        "\\begin{small}",
        "\\begin{sc}",
        f"\\begin{{tabularx}}{{\\textwidth}}{{{col_spec}}}",
        "\\toprule",
    ]

    header = ["Metric"] + [METHOD_LABELS[m] for m in methods_order]
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")

    for metric in metrics_order:
        row_cells = {m: ratings.get((metric, noise_mode, m)) for m in methods_order}
        present = {m: c for m, c in row_cells.items() if c is not None}
        best_method = max(present, key=lambda m: present[m][0]) if present else None

        row = [METRIC_LABELS[metric]]
        for m in methods_order:
            color = BEST_CELL_COLOR if m == best_method else None
            row.append(_fmt_cell(row_cells[m], color=color))
        lines.append(" & ".join(row) + " \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabularx}",
        "\\end{sc}",
        "\\end{small}",
        "\\end{center}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="../config.yaml",
                         help="Path to config.yaml (default: ../config.yaml), used to read "
                              "comparison_methods when --comparison-methods is omitted.")
    parser.add_argument("--comparison-methods", nargs="+", choices=METHOD_CHOICES, default=None,
                         help="Restrict the table's columns to these methods (overrides "
                              "config.yaml's comparison_methods). Only baseline/curriculum/"
                              "gate/saga/cp are meaningful here; others are ignored.")
    parser.add_argument("--tie-eps", type=float, default=1e-4)
    parser.add_argument("--reg", type=float, default=0.5)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    selected = resolve_comparison_methods(args.comparison_methods, args.config)
    methods_order = _filter_methods(METHODS_ORDER, selected)
    if len(methods_order) < 2:
        print("Fewer than 2 of the requested baselines are in comparison_methods; nothing to do.")
        return
    if selected is not None:
        print(f"Restricting to comparison_methods: {selected}")

    print(f"DB: {DB_PATH.resolve()}  |  exists: {DB_PATH.exists()}")

    ratings = compute_ratings(
        METRICS_ORDER, methods_order,
        tie_eps=args.tie_eps, reg=args.reg, n_boot=args.n_boot, seed=args.seed,
    )

    REPORTS_DIR.mkdir(exist_ok=True)
    for noise_mode in ["ar", "nar"]:
        latex = build_latex(ratings, noise_mode, METRICS_ORDER, methods_order)
        print(f"\n% ── {NOISE_MODE_LABELS[noise_mode]} ({noise_mode.upper()}) Elo-by-metric table ──")
        print(latex)
        out_path = REPORTS_DIR / f"elo_metric_table_{noise_mode}.tex"
        out_path.write_text(latex)
        print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
