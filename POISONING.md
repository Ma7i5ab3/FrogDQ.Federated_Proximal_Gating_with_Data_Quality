# Poisoning Configuration

Source: `poisoning:` block in [config.yaml](config.yaml), applied by `scripts/poison_data.py`.
Only feature columns (`num_*`, `cat_*`) are ever poisoned — targets (`cls_*`, `reg_*`) are untouched.

## Corruption model

Each poisonable column's tier rate is a **fixed corruption budget** for that column: the
rows selected for it are split **exactly 50/50, with no overlap**, between a value-level
mechanism (noise / flip) and a missingness mechanism (MCAR for AR; value-weighted for NAR).
Because the split is disjoint by construction, a column's total corrupted-cell share is
always *exactly* its configured tier rate — never more, regardless of how high that rate is
set (earlier revisions added missingness as an extra 0.75× on top of the noise rate, which
could overlap with already-noised cells at high rates and silently cap out below the
intended total; this is no longer possible).

Given the tier column-shares (`poisoning.noise`) and rates (`poisoning.presets.rates`), the
dataset-wide expected total (over **all** feature cells, including clean-reserved ones) is:

```
total = (Σ tier_column_share × tier_rate) × (1 − clean_feature_frac)
```

`scripts/poison_data.py` (`compute_ar_expected_rate`) computes this value and uses it as the
default corruption-budget cap for **both** AR and NAR (see below), so the two modes always
target the same overall corrupted-cell share unless explicitly overridden.

## Column setup

- **`clean_feature_frac: 0.40`** — 40% of feature columns are reserved as fully clean (never touched by any mechanism); the remaining 60% are "poisonable."
- **Noise tiers**, assigned across the poisonable 60% of columns (column shares are fixed across all presets below; only the per-tier rates change):

  | Tier | Column share |
  |---|---|
  | Mild | 33% (`1 - moderate - heavy - severe`) |
  | Moderate | 35% |
  | Heavy | 22% |
  | Severe | 10% |

## AR — At Random poisoning

Corruption is applied uniformly at random within each column, independent of the data's own
values. Each column's tier-rate budget is split 50/50 between:

- **`enable_numerical_noise`** — additive Gaussian noise on numerical columns (magnitude scaled to each column's own std, `k ≈ 1.2–1.5×`).
- **`enable_categorical_flips`** — categorical values replaced by a uniformly random *other* category.
- **`enable_missing`** — MCAR (Missing Completely At Random): `NaN`.

(If only one of the two mechanisms for a column's type is enabled, it receives the *entire*
budget instead of half.)

Because corruption doesn't depend on the value itself, AR poisoning is "easier" to correct statistically (e.g. imputation with column mean/mode is unbiased).

- **Corruption budget cap (`ar_max_corruption`)**: after the mechanisms run, total corrupted cells (over *all* feature cells) are clamped to at most this value — cells are randomly cleared if exceeded. Since the 50/50 split already makes AR's total deterministic by construction, this is a no-op safety net in normal use; it only bites if tier settings are overridden inconsistently (e.g. mismatched CLI flags). Left unset in `config.yaml`, it defaults to AR's own computed total.

## NAR — Not At Random poisoning

Corruption probability/magnitude depends on the value being corrupted, mimicking real-world
data-quality failures where the noise/missingness process is entangled with the data itself.
As in AR, each column's tier-rate budget is split 50/50 between a paired noise mechanism and
a missing mechanism — but here the *selection* of which rows land in which half is itself
value-dependent rather than uniform:

- **Numerical** — `enable_nnar` (NNAR: heteroscedastic noise, magnitude up to 4× larger for values near the column max vs. min) and `enable_mnar` (MNAR: missingness probability weighted by distance from the column median, so extreme/outlier values are disproportionately erased) share the budget; MNAR's candidate rows are exactly the ones NNAR didn't already take.
- **Categorical** — `enable_systematic_flips` (values corrupted via a cyclic confusion map: 70% chance of swapping to a fixed "adjacent" category, 30% chance random) and `enable_rare_missing` (missingness probability weighted inversely by category frequency, so rare categories vanish more often) share the budget the same way.
- **`enable_correlated_noise`** — noise chains: a few "anchor" numerical columns propagate correlated noise to other numerical columns on the same already-poisoned rows, simulating cascading corruption. This runs *on top of* the per-column budgets above (it targets *other* columns' cells) and is reined in by the corruption budget cap below.

Because NAR corruption is value-dependent, it is systematically biased and harder to detect/correct — plain imputation reintroduces bias.

- **Corruption budget**: after all mechanisms run, total corrupted cells (over *all* feature cells) are clamped to `[nar_min_corruption, nar_max_corruption]` — cells are randomly cleared or added as needed to hit this range. `config.yaml` leaves both **unset on purpose**: `scripts/poison_data.py` (`compute_ar_expected_rate`) then computes them dynamically from the AR noise-tier settings above, pinning `min_corruption == max_corruption` to AR's own total so NAR's total always matches AR's — no matter how the tier fracs/rates change. Set explicit values in `poisoning.nar` to override this and decouple the two.

## Presets

Column setup (`clean_feature_frac: 0.40`, tier column-shares 33/35/22/10%) is identical
across all four presets; only the per-tier rates change, scaled to hit each target total.
Per-column rates below are exact (50/50 split, no overlap), so **AR's total always lands
exactly on target by construction** — the corruption-budget cap exists only as a safety net,
and NAR's cap pins it to the same target.

| Preset | mild | moderate | heavy | severe | AR / NAR total (all cells) |
|---|---|---|---|---|---|
| 10% | 12.9% | 16.4% | 19.8% | 23.3% | ≈ 10.0% |
| 20% | 25.7% | 32.7% | 39.7% | 46.7% | ≈ 20.0% |
| 30% | 38.6% | 49.1% | 59.5% | 70.0% | ≈ 30.0% |
| 40% (current `config.yaml`) | 51.5% | 65.4% | 79.4% | 93.3% | ≈ 40.0% |

These presets live in `config.yaml` under `poisoning.presets`: `run` lists which presets the
pipeline iterates (and in what order), `rates` holds each preset's per-tier rates. Column
shares are *not* repeated there — they come from `poisoning.noise`
(`moderate_frac`/`heavy_frac`/`severe_frac` = `0.35`/`0.22`/`0.10`) and are identical for
every preset.

`./run_pipeline.sh` reads that block and runs each preset end-to-end (poison → CP → Saga++ →
experiments → evaluation) into `results/<preset>pct/`, wiping `data_poisoned/`,
`data_cleaned_cp/` and `data_cleaned_saga/` between rounds. Restrict a run with
`--presets "10,30"`.

A standalone `scripts/poison_data.py` invocation (outside the pipeline) uses the preset named
by `poisoning.presets.default`. Override any tier per-run with `--mild-rate`,
`--moderate-rate`, `--heavy-rate`, `--severe-rate`.

Note on the 40% preset: hitting a 40%-of-all-cells target with only 60% of columns
poisonable requires a *weighted-average* per-column rate of 66.7%. Keeping meaningful
tier separation (severe noticeably worse than mild) while staying under a 100% per-column
rate leaves little headroom — the tier spread above (mild=51.5% → severe=93.3%, a margin of
~6.7 points below the 100% ceiling) is close to the maximum achievable at this target without
either flattening the tiers further or reducing `clean_feature_frac`.
