#!/usr/bin/env python3
"""
Step 9 — Results Evaluation: Friedman test + critical-difference diagrams.

Reads the combined results CSV produced by main.py (step 8) and, for each
(model_type × corruption_mode) combination, runs a Friedman test across methods
(blocked by (dataset, seed)) and renders a Nemenyi critical-difference diagram.

Two plots are produced per combination:
  - with_clean   : all methods including the clean upper-bound reference
  - competitors  : all methods except clean (spreads ranks, increases sensitivity)

Plots are saved as PNG files under --output-dir.

Usage:
    python scripts/evaluate.py --results-dir results --output-dir evaluation
    python scripts/evaluate.py --config config.yaml
    python scripts/evaluate.py --comparison-methods clean baseline gate saga catboost_dirty
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import friedmanchisquare

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from frogdq.comparison_methods import METHOD_CHOICES, resolve_comparison_methods, resolve_from_label

try:
    import scikit_posthocs as sp
except ImportError:
    print("ERROR: scikit-posthocs is required.  Run: pip install scikit-posthocs>=0.8")
    sys.exit(1)

if not hasattr(sp, "critical_difference_diagram"):
    print(
        "ERROR: scikit-posthocs >= 0.8 is required for critical_difference_diagram. "
        "Run: pip install -U scikit-posthocs"
    )
    sys.exit(1)


# ── helpers ───────────────────────────────────────────────────────────────────

def method_label(data_mode, use_curriculum, use_gate, preparation):
    # CatBoost is a standalone model baseline (not tied to model_types), run
    # once on clean data and once on AR/NAR data. Checked before the
    # data_mode == "clean" shortcut below so its clean-mode row doesn't get
    # merged into the "clean" reference method.
    if preparation == "catboost":
        return "catboost_clean" if data_mode == "clean" else "catboost_dirty"
    if data_mode == "clean":
        return "clean"
    if preparation != "standard":
        return preparation
    if use_curriculum and use_gate:
        return "gate+curriculum"
    if use_gate:
        return "gate"
    if use_curriculum:
        return "curriculum"
    return "baseline"


def best_trial_seed_scores(trial_rows):
    """Per-seed test metric of the best (rank == 1) trial."""
    best = trial_rows[trial_rows["rank"] == trial_rows["rank"].min()]
    metric_col = (
        "test_f1"
        if "test_f1" in best.columns and best["test_f1"].notna().any()
        else "test_r2"
    )
    return dict(zip(best["seed"].astype(int), best[metric_col]))


def build_results(combined_results, selected_methods=None):
    """
    Build results[model_type][data_mode][method][(dataset, seed)] -> test metric.

    Parameters
    ----------
    selected_methods : list of str, optional
        Canonical method keys (see frogdq.comparison_methods.METHOD_CHOICES)
        to restrict to. None (default) includes every method found.
    """
    results = {}
    group_cols = [
        "model_type", "data_mode", "use_curriculum", "use_gate", "preparation", "dataset",
    ]
    for (model_type, data_mode, use_curriculum, use_gate, preparation, dataset), rows in (
        combined_results.groupby(group_cols)
    ):
        method = method_label(data_mode, use_curriculum, use_gate, preparation)
        if selected_methods is not None and resolve_from_label(method) not in selected_methods:
            continue
        seed_scores = best_trial_seed_scores(rows)
        method_dict = (
            results.setdefault(model_type, {})
                   .setdefault(data_mode, {})
                   .setdefault(method, {})
        )
        for seed, score in seed_scores.items():
            method_dict[(dataset, seed)] = score

    # CatBoost is a standalone model baseline (model_type == "catboost"), not
    # tied to linear/mlp architecture, so it never shares a model_type bucket
    # with them. Broadcast its two methods ("catboost_clean"/"catboost_dirty")
    # into every other model_type present, so they show up as two extra
    # competitor baselines in every comparison below — exactly like Saga/CP.
    catboost_by_mode = results.pop("catboost", None)
    if catboost_by_mode:
        for model_type, by_mode in results.items():
            for data_mode, methods in catboost_by_mode.items():
                for method, scores in methods.items():
                    by_mode.setdefault(data_mode, {})[method] = dict(scores)

    return results


def score_matrix(results, model_type, corruption_mode):
    """(dataset, seed) × methods matrix for the Friedman test."""
    # Every method under the "clean" data_mode bucket ("clean" itself, plus
    # "catboost_clean" — a second fixed reference of the same kind, since
    # CatBoost trained on clean data has no AR/NAR variant) is a fixed
    # reference reused across both corruption modes. Built with .get() /
    # dict unpacking (not indexing) since comparison_methods filtering may
    # have dropped either or both of them.
    clean_methods = results[model_type].get("clean", {})
    methods_scores = {
        **clean_methods,
        **results[model_type][corruption_mode],
    }
    common_blocks = sorted(
        set.intersection(*(set(s) for s in methods_scores.values()))
    )
    return pd.DataFrame(
        {method: [scores[blk] for blk in common_blocks] for method, scores in methods_scores.items()},
        index=pd.MultiIndex.from_tuples(common_blocks, names=["dataset", "seed"]),
    )


def friedman_and_cd_plot(matrix, title, output_path):
    """Friedman test + Nemenyi CD diagram; saves the figure to output_path."""
    stat, p_value = friedmanchisquare(*[matrix[c] for c in matrix.columns])
    avg_ranks = matrix.rank(axis=1, ascending=False).mean()
    nemenyi = sp.posthoc_nemenyi_friedman(matrix.to_numpy())
    nemenyi.index = nemenyi.columns = matrix.columns

    n_datasets = matrix.index.get_level_values("dataset").nunique()
    n_seeds = matrix.index.get_level_values("seed").nunique()
    print(
        f"  {title}\n"
        f"    blocks={len(matrix)} ({n_datasets} datasets × {n_seeds} seeds), "
        f"methods={matrix.shape[1]}, Friedman chi²={stat:.3f}, p={p_value:.3e}"
    )

    fig, ax = plt.subplots(figsize=(9, 0.6 * len(matrix.columns) + 1.5))
    sp.critical_difference_diagram(avg_ranks, nemenyi, ax=ax)
    ax.set_title(f"{title} — Friedman p = {p_value:.2e}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    → saved {output_path}")

    return stat, p_value, avg_ranks


# ── main ──────────────────────────────────────────────────────────────────────

def run_evaluation(results_csv: Path, output_dir: Path, selected_methods=None):
    if not results_csv.exists():
        print(f"ERROR: results CSV not found: {results_csv}")
        sys.exit(1)

    print(f"Loading results from: {results_csv}")
    combined_results = pd.read_csv(results_csv)
    print(f"  {len(combined_results)} rows loaded.")

    if selected_methods is not None:
        print(f"  Restricting to comparison_methods: {selected_methods}")

    results = build_results(combined_results, selected_methods=selected_methods)

    output_dir.mkdir(parents=True, exist_ok=True)

    cd_summary = {}
    for model_type, by_mode in results.items():
        if "clean" not in by_mode:
            print(f"Skipping {model_type}: no clean baseline results.")
            continue

        for corruption_mode in ("ar", "nar"):
            if corruption_mode not in by_mode:
                continue

            matrix = score_matrix(results, model_type, corruption_mode)
            if matrix.shape[1] < 3 or len(matrix) < 2:
                print(f"Skipping {model_type}/{corruption_mode}: not enough data.")
                continue

            base_title = f"{model_type.upper()} — {corruption_mode.upper()}"
            slug = f"{model_type}_{corruption_mode}"

            # Plot 1: all methods including clean
            stat, p, avg_ranks = friedman_and_cd_plot(
                matrix,
                title=f"{base_title} | with clean reference",
                output_path=output_dir / f"{slug}_with_clean.png",
            )
            cd_summary[(model_type, corruption_mode, "with_clean")] = (stat, p, avg_ranks)

            # Plot 2: competitors only (clean removed, if present)
            competitors = matrix.drop(columns="clean", errors="ignore")
            if competitors.shape[1] < 2:
                print(f"  Skipping competitors-only plot for {base_title}: only one method left.")
                continue
            stat2, p2, avg_ranks2 = friedman_and_cd_plot(
                competitors,
                title=f"{base_title} | competitors only (clean excluded)",
                output_path=output_dir / f"{slug}_competitors.png",
            )
            cd_summary[(model_type, corruption_mode, "competitors")] = (stat2, p2, avg_ranks2)

    # Summary table
    if cd_summary:
        rows = [
            {
                "model_type": k[0],
                "corruption_mode": k[1],
                "subset": k[2],
                "friedman_stat": round(v[0], 4),
                "p_value": round(v[1], 6),
                **{f"rank_{m}": round(r, 3) for m, r in v[2].items()},
            }
            for k, v in cd_summary.items()
        ]
        summary_df = pd.DataFrame(rows)
        summary_path = output_dir / "friedman_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to: {summary_path}")

    print(f"\nEvaluation complete. Plots written to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FrogDQ experiment results: Friedman test + CD diagrams",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing all_experiments_results.csv (overrides config output_dir)",
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        default=None,
        help="Direct path to all_experiments_results.csv (overrides --results-dir)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation",
        help="Directory to write CD diagram PNGs and summary CSV (default: evaluation)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml — used to read output_dir (when --results-dir is omitted) "
             "and comparison_methods (when --comparison-methods is omitted)",
    )
    parser.add_argument(
        "--comparison-methods",
        type=str,
        nargs="+",
        choices=METHOD_CHOICES,
        default=None,
        help="Restrict the comparison to these methods (overrides config.yaml's "
             "comparison_methods). Default: use config.yaml, or every method found if unset there.",
    )
    args = parser.parse_args()

    # Resolve the CSV path
    if args.results_csv:
        results_csv = Path(args.results_csv)
    elif args.results_dir:
        results_csv = Path(args.results_dir) / "all_experiments_results.csv"
    else:
        # Fall back to config output_dir
        config_path = Path(args.config)
        if config_path.exists():
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
            output_dir_cfg = cfg.get("output_dir", "results")
        else:
            output_dir_cfg = "results"
        results_csv = Path(output_dir_cfg) / "all_experiments_results.csv"

    selected_methods = resolve_comparison_methods(args.comparison_methods, args.config)

    run_evaluation(results_csv=results_csv, output_dir=Path(args.output_dir), selected_methods=selected_methods)


if __name__ == "__main__":
    main()
