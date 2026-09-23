# Poisoning Configuration

Source: `poisoning:` block in [config.yaml](config.yaml), applied by `scripts/poison_data.py`.
Only feature columns (`num_*`, `cat_*`) are ever poisoned — targets (`cls_*`, `reg_*`) are untouched.

## Corruption model

Each poisonable column's tier rate is a **fixed corruption budget** for that column: exactly
`round(rows × tier_rate)` cells are corrupted, drawn among the cells that **hold a value in
the source** (a cell that is already missing cannot be poisoned). They are split **exactly
50/50, with no overlap**, between a value-level mechanism (noise / flip) and a missingness
mechanism (MCAR for AR; value-weighted for NAR). A column's corrupted-cell share is
therefore always *exactly* its tier rate. (If a column has fewer observed values than its
budget, all of them are corrupted and a warning is logged; if the value-level mechanism
cannot act — a flip on a single-category column — the missingness mechanism takes the
whole budget.)

**The mask is derived from the data.** `<dataset>_mask.csv` is `True` exactly on the cells
the poisoning changed or made missing, and `False` on every cell it did not touch —
including the cells that were already missing in the source. Mask and data cannot
disagree.

Given the tier column-shares (`poisoning.noise`) and rates (`poisoning.presets.rates`), the
**nominal** dataset-wide total (over **all** feature cells, including clean-reserved ones) is:

```
total = (Σ tier_column_share × tier_rate) × (1 − clean_feature_frac)
```

(`compute_ar_expected_rate` in `scripts/poison_data.py`). It is exact only with infinitely
many columns: a real dataset has an integer number of clean and per-tier columns. Each count
is therefore its proportional quota rounded down *or* up (the shares below are kept to
within one column), and among those roundings `allocate_columns` picks the one whose total
`Σ n_columns(tier) × tier_rate / n_features` is closest to the nominal value. Both AR and NAR
then land exactly on that **realized** total — see the Presets table for how close it gets.

## Column setup

- **`clean_feature_frac: 0.40`** — 40% of feature columns are reserved as fully clean (never touched by any mechanism); the remaining 60% are "poisonable." Columns with no observed value in the source cannot be corrupted and are always among the clean ones.
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

- **Corruption budget cap (`ar_max_corruption`)**: unset by default, and then no clamp runs — the per-column budgets already fix the total. If set explicitly, randomly chosen corrupted cells are **restored to their clean value** until the total (over *all* feature cells) fits, at the cost of per-column exactness.

## NAR — Not At Random poisoning

Corruption probability/magnitude depends on the value being corrupted, mimicking real-world
data-quality failures where the noise/missingness process is entangled with the data itself.
As in AR, each column's tier-rate budget is split 50/50 between a paired noise mechanism and
a missing mechanism — but here the *selection* of which rows land in which half is itself
value-dependent rather than uniform:

- **Numerical** — `enable_nnar` (NNAR: heteroscedastic noise, magnitude up to 4× larger for values near the column max vs. min) and `enable_mnar` (MNAR: missingness probability weighted by distance from the column median, so extreme/outlier values are disproportionately erased) share the budget; MNAR's candidate rows are exactly the ones NNAR didn't already take.
- **Categorical** — `enable_systematic_flips` (values corrupted via a cyclic confusion map: 70% chance of swapping to a fixed "adjacent" category, 30% chance random) and `enable_rare_missing` (missingness probability weighted inversely by category frequency, so rare categories vanish more often) share the budget the same way.
- **`enable_correlated_noise`** — noise chains: a few "anchor" numerical columns propagate correlated noise to other numerical columns on the same already-poisoned rows, simulating cascading corruption. It runs *on top of* the per-column budgets above, so each column's **noise half** is then brought back to its size by restoring randomly chosen noisy cells to their clean value: correlated noise and the column's own noise compete for the same half, the missing half is untouched. Every column thus ends at exactly its tier rate with an exact 50/50 split.

Because NAR corruption is value-dependent, it is systematically biased and harder to detect/correct — plain imputation reintroduces bias.

- **Corruption budget**: since every column carries exactly the same budget as in AR, NAR's total equals AR's on every dataset by construction, and `config.yaml` leaves `nar_min_corruption` / `nar_max_corruption` **unset** — no clamp runs. Set them in `poisoning.nar` to force explicit bounds: the clamp then acts on the data, restoring corrupted cells to their clean value (cap) or corrupting untouched ones with their column's value-level mechanism (floor), at the cost of per-column exactness.

## Presets

Column setup (`clean_feature_frac: 0.40`, tier column-shares 33/35/22/10%) is identical
across all four presets; only the per-tier rates change, scaled to hit each target total.
Per-column rates below are exact (50/50 split, no overlap), and AR and NAR land on the same
realized total on every dataset. The last column is that realized total on the 15 datasets
of `config.yaml` — it departs from the nominal value only through the integer rounding of
the column counts, most on the datasets with few features (wilt: 5, car: 6, tic_tac_toe: 9).

| Preset | mild | moderate | heavy | severe | nominal total | realized AR = NAR (15 datasets) |
|---|---|---|---|---|---|---|
| 10% | 12.9% | 16.4% | 19.8% | 23.3% | 10.0% | 9.81 – 10.22% |
| 20% | 25.7% | 32.7% | 39.7% | 46.7% | 20.0% | 19.62 – 20.45% |
| 30% | 38.6% | 49.1% | 59.5% | 70.0% | 30.0% | 29.44 – 30.66% |
| 40% (current `config.yaml`) | 51.5% | 65.4% | 79.4% | 93.3% | 40.0% | 39.24 – 40.88% |

These presets live in `config.yaml` under `poisoning.presets`: `run` lists which presets the
pipeline iterates (and in what order), `rates` holds each preset's per-tier rates. Column
shares are *not* repeated there — they come from `poisoning.noise`
(`moderate_frac`/`heavy_frac`/`severe_frac` = `0.35`/`0.22`/`0.10`) and are identical for
every preset.

`./run_pipeline.sh` reads that block and runs each preset end-to-end (poison → CP → Saga++ →
Learn2Clean → DiffPrep → CtxPipe → experiments → evaluation) into `results/<preset>pct/`, wiping
`data_poisoned/` and every `data_cleaned_*/` directory between rounds. Restrict a run with
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
