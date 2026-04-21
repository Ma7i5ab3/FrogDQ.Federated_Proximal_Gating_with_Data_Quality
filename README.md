# frog-dq

## Data Corruption (Poisoning)

To reproduce the corrupted datasets used in the paper, run `poison_data.py` with the corruption distribution from Section IV.A:

| Severity | Feature fraction | Cell corruption rate |
|----------|-----------------|----------------------|
| Mild     | 56.25%          | 5%                   |
| Moderate | 25%             | 10%                  |
| Heavy    | 12.5%           | 20%                  |
| Severe   | 6.25%           | 40%                  |

```bash
python scripts/poison_data.py \
  --input_dir data \
  --output_dir data_poisoned \
  --mild-rate 0.05 \
  --moderate-frac 0.25 \
  --moderate-rate 0.10 \
  --heavy-frac 0.125 \
  --heavy-rate 0.20 \
  --severe-frac 0.0625 \
  --severe-rate 0.40
```

This generates two corruption modes under `data_poisoned/`:
- `ar/` — CCAR (Corruption Completely At Random): Gaussian noise, random categorical flips, MCAR missing values
- `nar/` — CNAR (Corruption Not At Random): heteroscedastic NNAR noise, MNAR missing values, systematic categorical confusion, correlated noise propagation, rare-value missingness