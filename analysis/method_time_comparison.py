#!/usr/bin/env python3
"""
method_time_comparison.py — Single overall training-cost bar chart (results_old).

A collapsed version of poisoning_pct_comparison.py's cost plots: instead of
splitting by poisoning percentage (10/20/30) and noise mode (AR/NAR), this
pools every pct folder and both noise modes together and reports one mean
total {cpu,wall}_time per method — five bars: Baseline, Saga++, CP,
Curriculum, QuAIL (Clean is dropped, same rationale as the per-severity cost
plots: it isn't a poisoning-defense competitor).

Total time uses the exact same definition as poisoning_pct_comparison.py /
the earlier mlp_{noise}_time_pct.png plots — external Saga++/CP preprocessing
cost (data_cleaned_saga|cp/metrics/*_perf_metrics.csv) + in-process
preproc_{cpu,wall}_time + training {cpu,wall}_time — reusing that module's
loader so the two plots can't drift apart. The (%) under each bar is its
share of the slowest method's time; the fastest method is highlighted green.

Usage
-----
    python method_time_comparison.py
    python method_time_comparison.py --metric wall_time
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from poisoning_pct_comparison import (
    PCTS, PLOTS_DIR, CATEGORY_ORDER, CATEGORY_LABELS, CATEGORY_COLORS,
    BEST_COLOR, INK_PRIMARY, INK_MUTED, METRIC_SPECS,
    load_pct, _load_external_preproc_times,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metric", default="cpu_time", choices=["cpu_time", "wall_time"],
                     help="Which cost quantity to average (default: cpu_time; "
                          "cpu_time and wall_time are equally valid picks here).")
    args = ap.parse_args()
    metric = args.metric
    spec = METRIC_SPECS[metric]
    dec = spec["dec"]

    df = pd.concat([load_pct(pct, spec) for pct in PCTS], ignore_index=True)
    df = df[df["category"] != "clean"].copy()
    print(f"Loaded {len(df)} rows (rank==1, mlp only, Clean dropped) "
          f"across pct={PCTS} and both noise modes | metric: {metric}")

    ext_lookup = _load_external_preproc_times()
    ext_key = spec["ext_key"]

    def _external(row):
        ext = ext_lookup.get((row["dataset"], row["data_mode"], row["category"]))
        return ext[ext_key] if ext else 0.0

    df["value"] = df["value"] + df.apply(_external, axis=1)

    # Pool every pct and both noise modes into one mean per method.
    means = df.groupby("category")["value"].mean()

    cats_to_plot = [c for c in CATEGORY_ORDER if c != "clean" and c in means.index]
    values = means.loc[cats_to_plot]

    slowest = values.max()
    best_cat = values.idxmin()

    x = np.arange(len(cats_to_plot))
    span = values.max() - values.min()

    fig, ax = plt.subplots(figsize=(8, 6))

    for xi, cat in zip(x, cats_to_plot):
        h = values[cat]
        is_best = (cat == best_cat)
        ax.bar(xi, h, width=0.6, color=CATEGORY_COLORS[cat],
               edgecolor=BEST_COLOR if is_best else "none",
               linewidth=2.4 if is_best else 0, zorder=2)

        label_color = BEST_COLOR if is_best else INK_PRIMARY
        ax.text(xi, h + span * 0.02, f"{h:.{dec}f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold" if is_best else "normal",
                color=label_color, zorder=4)
        pct_of_slowest = h / slowest * 100.0
        ax.text(xi, h + span * 0.075, f"({pct_of_slowest:.0f}%)", ha="center", va="bottom",
                fontsize=8.5, color=label_color if is_best else INK_MUTED, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_LABELS[c] for c in cats_to_plot], fontsize=10)
    ax.set_ylabel(f"Mean {spec['label']}")
    ax.set_ylim(0, values.max() + span * 0.22)
    ax.grid(axis="y", linewidth=0.3, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handle = plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=BEST_COLOR, linewidth=2.4)
    ax.legend([handle], ["Fastest method"], loc="upper left", frameon=False, fontsize=9)

    ax.set_title(
        f"Mean {spec['label']} per method\n"
        f"pooled across poisoning percentages {PCTS} and both noise modes (AR + NAR)\n"
        f"(%) is that bar's share of the slowest method's time",
        fontsize=11, fontweight="bold",
    )

    plt.tight_layout()
    PLOTS_DIR.mkdir(exist_ok=True)
    out = PLOTS_DIR / f"method_time_comparison_{metric}.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
