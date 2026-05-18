# Data Poisoning Profile

Snapshot of the poisoning configuration used in the baseline experiments.
Source: `scripts/poison_data.py` (`DataPoisoner`) + `config.yaml`.
Seed: 42. All mechanisms enabled (no CLI overrides applied).

---

## Common Settings

| Parameter | Value | Source |
|---|---|---|
| Random seed | 42 | `DataPoisoner.__init__` |
| Test split fraction (clean holdout) | 0.30 | `config.yaml: test_size` |
| Train+val fraction | 0.70 | derived |

### Stratified Column Noise Distribution

Applied identically to both CAR and CNAR before any mechanism runs.
Columns are shuffled, then assigned a tier; remaining columns default to mild.

| Tier | Column fraction | Poison rate per column |
|---|---|---|
| Mild | 56.25 % | 6 % |
| Moderate | 25.00 % | 12 % |
| Heavy | 12.50 % | 24 % |
| Severe | 6.25 % | 48 % |

Calibrated to produce approximately **21 % total cell corruption in CAR**
(~12 % value noise + ~9 % MCAR).

---

## CAR — Corrupted At Random (`mode='ar'`)

Three independent mechanisms applied per column according to the tier rate above.
Labels (`cls_*`, `reg_*`) are never poisoned.

### Mechanism 1 — Additive Gaussian Noise (numerical features)

- **SNR**: 20 dB → `noise_std = signal_std / 10`
- `signal_std` computed on non-missing values of the clean column
- If `signal_std == 0`: fallback to `0.1 × |mean|` or `1.0`
- Number of poisoned rows per column: `floor(n_rows × poison_rate)`
- Rows sampled **without replacement** (uniform random)
- Noise drawn from `N(0, noise_std)` and added to the original value

### Mechanism 2 — Random Categorical Label Flips (categorical features)

- Number of poisoned rows: `floor(n_rows × poison_rate)`
- For each selected row: current value is replaced by a **uniformly random** choice
  from all other valid categories (skipped if only one category exists)

### Mechanism 3 — MCAR Missing Values (all feature columns)

- Number of missing rows per column: `floor(n_rows × poison_rate × 0.75)`
- Rows sampled **without replacement** (uniform random)
- Selected cells set to `NaN`
- Integer columns cast to `float64` before insertion

---

## CNAR — Corrupted Not At Random (`mode='nar'`)

Five mechanisms applied sequentially. Total cell corruption is clamped to
**[25 %, 50 %]** by a post-hoc budget step.

### Mechanism 1 — NNAR: Heteroscedastic Gaussian Noise (numerical features)

- **SNR**: 15 dB → `base_noise_std = signal_std / 10^(15/20) ≈ signal_std / 5.62`
- Noise magnitude scales with the **normalised value** of each sample:
  `noise_std_i = base_noise_std × (1 + 3 × (val_i − min) / (max − min))`
  → higher values receive up to 4× more noise than the minimum
- Number of poisoned rows: `floor(n_rows × poison_rate)`
- Only non-missing rows are eligible

### Mechanism 2 — MNAR: Extreme-Value Missingness (numerical features)

- Missingness probability per row proportional to distance from median:
  `prob_i = clip(|val_i − median| / (2 × IQR) × poison_rate, max=0.5)`
- Applied to all currently non-missing rows; selected cells set to `NaN`
- Columns with `IQR == 0` are skipped

### Mechanism 3 — Systematic Categorical Confusion (categorical features)

- **Confusion pairs**: values sorted into a cyclic map `v_i → v_{i+1 mod K}`
- For each selected row (n = `floor(n_rows × poison_rate)`):
  - 70 % probability: replace with the designated confusion partner
  - 30 % probability: replace with a uniformly random other category

### Mechanism 4 — Correlated Noise Propagation (numerical features)

- Number of anchor columns: `max(1, min(n_num_cols // 3, 3))`
- Anchor columns chosen uniformly at random (without replacement)
- For each anchor: rows already poisoned by Mechanism 1 are identified;
  each other numerical column has a **60 % probability** of also receiving
  `N(0, signal_std / 5.62)` noise on those same rows

### Mechanism 5 — Rare-Category Missingness (categorical features)

- Missingness probability per row inversely proportional to category frequency:
  `prob_i = poison_rate × (1 − freq(category_i))`
- Rare categories are more likely to be set to `NaN`

### NAR Corruption Budget

After all five mechanisms, total cell corruption across feature columns is
clamped:

| Bound | Value |
|---|---|
| Floor (minimum) | 25 % |
| Cap (maximum) | 50 % |

If above the cap: True entries are randomly cleared until the cap is met.
If below the floor: False entries are randomly set to True until the floor is met.

---

## Post-Poisoning: TabularPreprocessor (applied at training time)

The preprocessor is fitted on the poisoned **training** data and transforms all splits.
This partially mitigates corruption before the model sees the data.

| Step | Numerical | Categorical |
|---|---|---|
| Imputation | Median of poisoned train | Most-frequent of poisoned train |
| Outlier clipping | p1–p99 of poisoned train | — |
| Encoding | StandardScaler | OneHotEncoder (`handle_unknown='ignore'`) |
| Rare grouping | — | Categories < 1 % frequency → `RARE_CATEGORY` |

**Effect**: NaN values injected by MCAR/MNAR are always imputed away before training.
Gaussian noise within the clipping range passes through. The preprocessor's own
statistics (median, mean, std) are computed from the corrupted training distribution
and are therefore slightly biased relative to the clean distribution.
