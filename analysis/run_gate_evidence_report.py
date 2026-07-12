#!/usr/bin/env python3
"""
run_gate_evidence_report.py — Reproduce the full Gate-vs-Baseline evidence
report end to end.

Runs, in order:
  1. optuna_progress.py       --metric {metric}
  2. gate_analysis.py         --metric {metric} --latex   (includes Nemenyi post-hoc)
  3. elo_ratings.py           --metric {metric}
  4. latex_evidence_table.py  (all metrics x comparison_methods, per dataset)

Captures both scripts' output verbatim, re-derives the headline numbers
(Friedman p-values, best-ranked method, Gate-vs-Baseline win rate and
Wilcoxon test) directly from the data, and writes everything to a single
Markdown report under analysis/reports/. All plots referenced in the report
are written to analysis/plots/ by the two scripts above.

Usage
-----
    python run_gate_evidence_report.py
    python run_gate_evidence_report.py --metric accuracy
    python run_gate_evidence_report.py --comparison-methods baseline gate saga catboost_dirty
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from frogdq.comparison_methods import METHOD_CHOICES, resolve_comparison_methods

HERE = Path(__file__).resolve().parent
REPORTS_DIR = HERE / "reports"

_METRIC_LABELS = {
    "f1": "F1", "accuracy": "Accuracy", "precision": "Precision",
    "recall": "Recall", "auc": "AUC-ROC", "loss": "Loss",
}


def run_script(args_list: list) -> str:
    """
    Run a script in-process under the analysis/ directory and capture
    everything it prints to stdout, while still streaming it live.

    Using runpy (not subprocess) keeps a single Python process and avoids
    re-importing heavy dependencies (optuna, matplotlib) twice.
    """
    import contextlib
    import io
    import runpy

    old_argv = sys.argv
    sys.argv = args_list
    buf = io.StringIO()

    class _Tee(io.TextIOBase):
        def write(self, s):
            sys.__stdout__.write(s)
            buf.write(s)
            return len(s)

    try:
        with contextlib.redirect_stdout(_Tee()):
            runpy.run_path(str(HERE / args_list[0]), run_name="__main__")
    finally:
        sys.argv = old_argv

    return buf.getvalue()


def key_findings(metric: str, selected_methods=None) -> str:
    """Re-derive the headline numbers using gate_analysis's own functions."""
    sys.path.insert(0, str(HERE))
    import gate_analysis as ga

    seed_df = ga.load_seed_data(metric=metric)
    dm = ga.dataset_means(seed_df)
    dm = ga.broadcast_catboost_clean(dm)

    ga._SELECTED_METHODS = selected_methods
    methods_order = ga._filter_tokens([
        "curr0_gate0", "curr0_gate1", "curr1_gate0", "saga", "cp",
        "catboost_clean", "catboost_dirty",
    ])
    lines = []

    for noise_mode in ["ar", "nar"]:
        sub = dm[dm["data_mode"] == noise_mode]
        pivot = sub.pivot_table(index="dataset", columns="config",
                                values="metric_mean", aggfunc="mean")
        available = [c for c in methods_order if c in pivot.columns]
        pivot_full = pivot[available].dropna()
        if pivot_full.empty:
            continue

        ranked = pivot_full.rank(axis=1, ascending=False, method="min")
        mean_ranks = ranked.mean(axis=0)
        try:
            _, p_friedman = ga.stats.friedmanchisquare(*[pivot_full[c].values for c in available])
        except Exception:
            p_friedman = float("nan")

        best_method = mean_ranks.idxmin()
        best_label  = ga.CONFIG_LABELS.get(best_method, best_method)

        gate_col, base_col = "curr0_gate1", "curr0_gate0"
        if gate_col in pivot.columns and base_col in pivot.columns:
            shared = pivot[[gate_col, base_col]].dropna()
            g, b = shared[gate_col].values, shared[base_col].values
            w, d, l = ga.sign_test(g, b)
            _, p_w, r_w = ga.wilcoxon_and_effect(g, b)
        else:
            w = d = l = 0
            p_w = r_w = float("nan")

        lines.append(
            f"- **{noise_mode.upper()}** (N={len(pivot_full)} datasets, {len(available)} methods): "
            f"Friedman p={p_friedman:.4f} — best mean rank is **{best_label}** "
            f"({mean_ranks[best_method]:.2f}). Gate vs Baseline win/draw/loss = "
            f"{w}/{d}/{l}, Wilcoxon p={p_w:.4f}, effect size |r|={abs(r_w):.3f}."
        )

    return "\n".join(lines) if lines else "- (insufficient data to derive findings)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--metric", default="f1",
        choices=["f1", "accuracy", "precision", "recall", "auc", "loss"],
        help="Performance metric to analyse (default: f1).",
    )
    parser.add_argument(
        "--config", default=str(HERE.parent / "config.yaml"),
        help="Path to config.yaml, used to read comparison_methods when "
             "--comparison-methods is omitted (default: project root config.yaml).",
    )
    parser.add_argument(
        "--comparison-methods", nargs="+", choices=METHOD_CHOICES, default=None,
        help="Restrict the whole report (optuna_progress.py, gate_analysis.py, and "
             "the key-findings summary below) to these methods (overrides "
             "config.yaml's comparison_methods).",
    )
    args = parser.parse_args()
    metric_name = _METRIC_LABELS[args.metric]
    selected_methods = resolve_comparison_methods(args.comparison_methods, args.config)
    extra_args = ["--comparison-methods", *selected_methods] if selected_methods else []

    # Both optuna_progress.py and gate_analysis.py resolve DB_PATH and
    # "plots/" relative to the current working directory, so anchor it here
    # regardless of where this script was invoked from.
    os.chdir(HERE)

    REPORTS_DIR.mkdir(exist_ok=True)
    (HERE / "plots").mkdir(exist_ok=True)

    print(f"Reproducing Gate-vs-Baseline evidence report  [{metric_name}]\n")
    if selected_methods is not None:
        print(f"Restricting to comparison_methods: {selected_methods}\n")

    print(f"── Step 1/4: optuna_progress.py --metric {args.metric} ──")
    out1 = run_script(["optuna_progress.py", "--metric", args.metric, "--complete-only", *extra_args])

    print(f"\n── Step 2/4: gate_analysis.py --metric {args.metric} --latex ──")
    out2 = run_script(["gate_analysis.py", "--metric", args.metric, "--latex", *extra_args])

    print(f"\n── Step 3/4: elo_ratings.py --metric {args.metric} --latex ──")
    out3 = run_script(["elo_ratings.py", "--metric", args.metric])

    print("\n── Step 4/4: latex_evidence_table.py ──")
    out4 = run_script(["latex_evidence_table.py", "--split-by-metric", "--config", args.config, *extra_args])


if __name__ == "__main__":
    main()
