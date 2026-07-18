# Poisoning Configuration 10%

Source: `poisoning:` block in [config.yaml](config.yaml), applied by `scripts/poison_data.py`.
Only feature columns (`num_*`, `cat_*`) are ever poisoned — targets (`cls_*`, `reg_*`) are untouched.

## Column setup

- **`clean_feature_frac: 0.40`** — 40% of feature columns are reserved as fully clean (never touched by any mechanism); the remaining 60% are "poisonable."
- **Noise tiers**, assigned across the poisonable 60% of columns, each with its own per-column corruption rate:

  | Tier | Column share | Per-column rate |
  |---|---|---|
  | Mild | 33% (`1 - moderate - heavy - severe`) | 4.1% |
  | Moderate | 35% | 7.7% |
  | Heavy | 22% | 14.4% |
  | Severe | 10% | 23.1% |

## AR — At Random poisoning

Corruption is applied uniformly at random within each column at its tier rate, independent of the data's own values. All three mechanisms are enabled:

- **`enable_numerical_noise`** — additive Gaussian noise on numerical columns (magnitude scaled to each column's own std, `k ≈ 1.2–1.5×`), added to a random subset of rows at the column's tier rate.
- **`enable_categorical_flips`** — a random subset of rows (tier rate) get their categorical value replaced by a uniformly random *other* category.
- **`enable_missing`** — MCAR (Missing Completely At Random): an extra `0.75×` the tier rate of rows (preferring rows not already corrupted) are set to `NaN`.

Because corruption doesn't depend on the value itself, AR poisoning is "easier" to correct statistically (e.g. imputation with column mean/mode is unbiased).

## NAR — Not At Random poisoning

Corruption probability/magnitude depends on the value being corrupted, mimicking real-world data-quality failures where the noise/missingness process is entangled with the data itself. All five mechanisms are enabled:

- **`enable_nnar`** (NNAR) — heteroscedastic noise: larger values get proportionally more Gaussian noise (up to 4× the base noise at the column max vs. its min).
- **`enable_mnar`** (MNAR) — missingness probability grows with distance from the column median, so extreme/outlier values are disproportionately erased.
- **`enable_systematic_flips`** — categorical values are corrupted via a cyclic confusion map (70% chance of swapping to a fixed "adjacent" category, 30% chance random), instead of a uniform random flip.
- **`enable_correlated_noise`** — noise chains: a few "anchor" numerical columns propagate correlated noise to other numerical columns on the same already-poisoned rows, simulating cascading corruption.
- **`enable_rare_missing`** — rare categorical values are more likely to go missing than frequent ones (frequency-inverse missingness).

Because NAR corruption is value-dependent, it is systematically biased and harder to detect/correct — plain imputation reintroduces bias.

- **Corruption budget**: after all five mechanisms run, total corrupted cells (over the poisonable columns only) are clamped to `[min_corruption, max_corruption]` — cells are randomly cleared or added as needed to hit this range. `config.yaml` leaves both **unset on purpose**: `scripts/poison_data.py` (`compute_ar_expected_rate`) then computes them dynamically from the AR noise-tier settings above, pinning `min_corruption == max_corruption ≈ 16.7%` so NAR's total always matches AR's total — no matter how the tier fracs/rates change. Set explicit values in `poisoning.nar` to override this and decouple the two.

## Total percentage of cells poisoned

Counting over *all* feature cells (i.e., including the 40% clean-reserved columns):

- **AR**: ≈ **10.0%** of all cells (expected value; no hard clamp). Derived from the weighted average tier rate (≈9.53%) × 1.75 (value-corruption + 0.75× MCAR) × 0.60 (poisonable column share).
- **NAR**: pinned to the **same ≈10.0%** of all cells, since `min_corruption`/`max_corruption` are auto-derived from that same AR calculation (≈16.7% of the poisonable 60% → ≈10.0% of all cells). NAR still concentrates that corruption on extreme/rare values rather than spreading it uniformly like AR.

# Poisoning Configuration 20%

Source: `poisoning:` block in [config.yaml](config.yaml), applied by `scripts/poison_data.py`.
Only feature columns (`num_*`, `cat_*`) are ever poisoned — targets (`cls_*`, `reg_*`) are untouched.

## Column setup

- **`clean_feature_frac: 0.40`** — 40% of feature columns are reserved as fully clean (never touched by any mechanism); the remaining 60% are "poisonable."
- **Noise tiers**, assigned across the poisonable 60% of columns, each with its own per-column corruption rate:

  | Tier | Column share | Per-column rate |
  |---|---|---|
  | Mild | 33% (`1 - moderate - heavy - severe`) | 8% |
  | Moderate | 35% | 15% |
  | Heavy | 22% | 28% |
  | Severe | 10% | 45% |

## AR — At Random poisoning

Corruption is applied uniformly at random within each column at its tier rate, independent of the data's own values. All three mechanisms are enabled:

- **`enable_numerical_noise`** — additive Gaussian noise on numerical columns (magnitude scaled to each column's own std, `k ≈ 1.2–1.5×`), added to a random subset of rows at the column's tier rate.
- **`enable_categorical_flips`** — a random subset of rows (tier rate) get their categorical value replaced by a uniformly random *other* category.
- **`enable_missing`** — MCAR (Missing Completely At Random): an extra `0.75×` the tier rate of rows (preferring rows not already corrupted) are set to `NaN`.

Because corruption doesn't depend on the value itself, AR poisoning is "easier" to correct statistically (e.g. imputation with column mean/mode is unbiased).

## NAR — Not At Random poisoning

Corruption probability/magnitude depends on the value being corrupted, mimicking real-world data-quality failures where the noise/missingness process is entangled with the data itself. All five mechanisms are enabled:

- **`enable_nnar`** (NNAR) — heteroscedastic noise: larger values get proportionally more Gaussian noise (up to 4× the base noise at the column max vs. its min).
- **`enable_mnar`** (MNAR) — missingness probability grows with distance from the column median, so extreme/outlier values are disproportionately erased.
- **`enable_systematic_flips`** — categorical values are corrupted via a cyclic confusion map (70% chance of swapping to a fixed "adjacent" category, 30% chance random), instead of a uniform random flip.
- **`enable_correlated_noise`** — noise chains: a few "anchor" numerical columns propagate correlated noise to other numerical columns on the same already-poisoned rows, simulating cascading corruption.
- **`enable_rare_missing`** — rare categorical values are more likely to go missing than frequent ones (frequency-inverse missingness).

Because NAR corruption is value-dependent, it is systematically biased and harder to detect/correct — plain imputation reintroduces bias.

- **Corruption budget**: after all five mechanisms run, total corrupted cells (over the poisonable columns only) are clamped to `[min_corruption, max_corruption]` — cells are randomly cleared or added as needed to hit this range. `config.yaml` leaves both **unset on purpose**: `scripts/poison_data.py` (`compute_ar_expected_rate`) then computes them dynamically from the AR noise-tier settings above, pinning `min_corruption == max_corruption ≈ 32.5%` so NAR's total always matches AR's total — no matter how the tier fracs/rates change. Set explicit values in `poisoning.nar` to override this and decouple the two.

## Total percentage of cells poisoned

Counting over *all* feature cells (i.e., including the 40% clean-reserved columns):

- **AR**: ≈ **19.5%** of all cells (expected value; no hard clamp). Derived from the weighted average tier rate (≈18.55%) × 1.75 (value-corruption + 0.75× MCAR) × 0.60 (poisonable column share).
- **NAR**: pinned to the **same ≈19.5%** of all cells, since `min_corruption`/`max_corruption` are auto-derived from that same AR calculation (≈32.5% of the poisonable 60% → ≈19.5% of all cells). NAR still concentrates that corruption on extreme/rare values rather than spreading it uniformly like AR.

# Poisoning Configuration 30%

Source: `poisoning:` block in [config.yaml](config.yaml), applied by `scripts/poison_data.py`.
Only feature columns (`num_*`, `cat_*`) are ever poisoned — targets (`cls_*`, `reg_*`) are untouched.

## Column setup

- **`clean_feature_frac: 0.40`** — 40% of feature columns are reserved as fully clean (never touched by any mechanism); the remaining 60% are "poisonable."
- **Noise tiers**, assigned across the poisonable 60% of columns, each with its own per-column corruption rate:

  | Tier | Column share | Per-column rate |
  |---|---|---|
  | Mild | 33% (`1 - moderate - heavy - severe`) | 12.3% |
  | Moderate | 35% | 23.1% |
  | Heavy | 22% | 43.1% |
  | Severe | 10% | 69.3% |

## AR — At Random poisoning

Corruption is applied uniformly at random within each column at its tier rate, independent of the data's own values. All three mechanisms are enabled:

- **`enable_numerical_noise`** — additive Gaussian noise on numerical columns (magnitude scaled to each column's own std, `k ≈ 1.2–1.5×`), added to a random subset of rows at the column's tier rate.
- **`enable_categorical_flips`** — a random subset of rows (tier rate) get their categorical value replaced by a uniformly random *other* category.
- **`enable_missing`** — MCAR (Missing Completely At Random): an extra `0.75×` the tier rate of rows (preferring rows not already corrupted) are set to `NaN`.

Because corruption doesn't depend on the value itself, AR poisoning is "easier" to correct statistically (e.g. imputation with column mean/mode is unbiased).

## NAR — Not At Random poisoning

Corruption probability/magnitude depends on the value being corrupted, mimicking real-world data-quality failures where the noise/missingness process is entangled with the data itself. All five mechanisms are enabled:

- **`enable_nnar`** (NNAR) — heteroscedastic noise: larger values get proportionally more Gaussian noise (up to 4× the base noise at the column max vs. its min).
- **`enable_mnar`** (MNAR) — missingness probability grows with distance from the column median, so extreme/outlier values are disproportionately erased.
- **`enable_systematic_flips`** — categorical values are corrupted via a cyclic confusion map (70% chance of swapping to a fixed "adjacent" category, 30% chance random), instead of a uniform random flip.
- **`enable_correlated_noise`** — noise chains: a few "anchor" numerical columns propagate correlated noise to other numerical columns on the same already-poisoned rows, simulating cascading corruption.
- **`enable_rare_missing`** — rare categorical values are more likely to go missing than frequent ones (frequency-inverse missingness).

Because NAR corruption is value-dependent, it is systematically biased and harder to detect/correct — plain imputation reintroduces bias.

- **Corruption budget**: after all five mechanisms run, total corrupted cells (over the poisonable columns only) are clamped to `[min_corruption, max_corruption]` — cells are randomly cleared or added as needed to hit this range. `config.yaml` leaves both **unset on purpose**: `scripts/poison_data.py` (`compute_ar_expected_rate`) then computes them dynamically from the AR noise-tier settings above, pinning `min_corruption == max_corruption ≈ 50.0%` so NAR's total always matches AR's total — no matter how the tier fracs/rates change. Set explicit values in `poisoning.nar` to override this and decouple the two.

## Total percentage of cells poisoned

Counting over *all* feature cells (i.e., including the 40% clean-reserved columns):

- **AR**: ≈ **30.0%** of all cells (expected value; no hard clamp). Derived from the weighted average tier rate (≈28.56%) × 1.75 (value-corruption + 0.75× MCAR) × 0.60 (poisonable column share).
- **NAR**: pinned to the **same ≈30.0%** of all cells, since `min_corruption`/`max_corruption` are auto-derived from that same AR calculation (≈50.0% of the poisonable 60% → ≈30.0% of all cells). NAR still concentrates that corruption on extreme/rare values rather than spreading it uniformly like AR.

