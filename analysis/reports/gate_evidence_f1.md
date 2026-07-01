# Gate Evidence Report — F1

Generated: 2026-07-01 07:45
Reproduce with: `python run_gate_evidence_report.py --metric f1`
(equivalent to running `optuna_progress.py --metric f1` then
`gate_analysis.py --metric f1 --latex` from this directory)

## Key findings (auto-derived)

- **AR** (N=27 datasets, 5 methods): Friedman p=0.0000 — best mean rank is **+ Gate** (1.96). Gate vs Baseline win/draw/loss = 18/0/9, Wilcoxon p=0.1286, effect size |r|=0.339.
- **NAR** (N=26 datasets, 5 methods): Friedman p=0.0000 — best mean rank is **+ Gate** (2.27). Gate vs Baseline win/draw/loss = 18/0/8, Wilcoxon p=0.0594, effect size |r|=0.425.

Full statistical detail (per-competitor Wilcoxon tests, per-dataset delta
table, Nemenyi post-hoc pairwise p-values, gate-behaviour correlations) is in
section 2 below. All referenced figures are in `analysis/plots/`.

## 1. Performance comparison — `optuna_progress.py --metric f1`

```
DB: /home/cappiello/mattia_sabella/quAIL/FrogDQ.Federated_Proximal_Gating_with_Data_Quality/results/optuna_studies.db  |  exists: True

────────────────────────────────────────────────────────────
 Studies loaded :  295  matched  |    0 skipped
 Metric key     : 'test_f1'
 Datasets       : 27
 Configs        : 5
 With CPU time  : 295 / 295
 With conv.stats: 295 / 295
────────────────────────────────────────────────────────────

[complete-only] Filtering datasets…
  [complete-only] mlp/nar: dropping 1 incomplete dataset(s)
[complete-only] 27 dataset(s) retained.


── MLP / AR ──
  Saved: plots/mlp_ar_metric_mean.png
  Saved: plots/mlp_ar_metric_mean_heatmap.png
  Saved: plots/mlp_ar_metric_mean_efficiency.png

── MLP / NAR ──
  Saved: plots/mlp_nar_metric_mean.png
  Saved: plots/mlp_nar_metric_mean_heatmap.png
  Saved: plots/mlp_nar_metric_mean_efficiency.png

══════════════════════════════════════════════════════════════════════
  Model: MLP   [F1]
══════════════════════════════════════════════════════════════════════

  Rank  Config                       CLEAN          AR         NAR    Wins
  ----- ----------------------  ----------  ----------  ----------  ------
  1     Baseline                    0.8595         N/A         N/A      36
  1     + Gate                         N/A      0.7900      0.7816      23
  2     + Curriculum                   N/A      0.7878         N/A       8
  2     Baseline                       N/A         N/A      0.7772      36
  3     + Curriculum                   N/A         N/A      0.7720       8
  3     Baseline                       N/A      0.7863         N/A      36
  4     Saga++ prep                    N/A      0.7605      0.7695      10
  5     CP prep                        N/A      0.7332      0.6701       3

  Macro-median F1:
  Config                            CLEAN          AR         NAR
  ---------------------------  ----------  ----------  ----------
  CP prep                             N/A      0.7612      0.7440
  Baseline                         0.9244      0.8025      0.7991
  + Gate                              N/A      0.8079      0.7952
  + Curriculum                        N/A      0.7946      0.7982
  Saga++ prep                         N/A      0.7660      0.8098

══════════════════════════════════════════════════════════════════════
```

## 2. Statistical evidence for Gate — `gate_analysis.py --metric f1 --latex`

```
DB: /home/cappiello/mattia_sabella/quAIL/FrogDQ.Federated_Proximal_Gating_with_Data_Quality/results/optuna_studies.db  |  exists: True
Loaded 2950 seed rows | 27 datasets | 5 configs | metric: test_f1

Running analyses…

════════════════════════════════════════════════════════════════════════
  STATISTICAL COMPARISON — Gate vs competitors  [F1]
════════════════════════════════════════════════════════════════════════

  Noise mode: AR
  Competitor             N    Δ mean   Δ median      W/D/L    p-value     |r|   sig
  -------------------- ---  --------  ---------  ---------  ---------  ------  ----
  Baseline              27   +0.0037    +0.0015  18/0/ 9     0.1286   0.339    ns
  + Curriculum          27   +0.0022    +0.0024  19/0/ 8     0.2021   0.286    ns
  Saga++                27   +0.0295    +0.0144  23/0/ 4     0.0002   0.767   ***
  CP prep               27   +0.0568    +0.0062  22/0/ 5     0.0002   0.772   ***

  Noise mode: NAR
  Competitor             N    Δ mean   Δ median      W/D/L    p-value     |r|   sig
  -------------------- ---  --------  ---------  ---------  ---------  ------  ----
  Baseline              26   +0.0044    +0.0035  18/0/ 8     0.0594   0.425    ns
  + Curriculum          26   +0.0095    +0.0046  17/1/ 8     0.0630   0.419    ns
  Saga++                26   +0.0120    +0.0009  16/0/10     0.8809   0.037    ns
  CP prep               26   +0.1114    +0.0653  20/1/ 5     0.0002   0.783   ***


════════════════════════════════════════════════════════════════════════
  PER-DATASET Δ (Gate − Baseline)  [F1]
════════════════════════════════════════════════════════════════════════
\begin{tabular}{llllll}
\toprule
Dataset & Mode & Baseline & Gate & Δ & Direction \\
\midrule
banknote_authentication & AR & 0.9949 & 0.9892 & -0.0057 & ↓ \\
car & AR & 0.5254 & 0.5394 & +0.0140 & ↑ \\
climate_model_simulation_crashes & AR & 0.9545 & 0.9558 & +0.0013 & ↑ \\
cmc & AR & 0.4406 & 0.4604 & +0.0199 & ↑ \\
cylinder_bands & AR & 0.7718 & 0.7733 & +0.0015 & ↑ \\
diabetes & AR & 0.6398 & 0.6265 & -0.0133 & ↓ \\
dna & AR & 0.7266 & 0.7181 & -0.0085 & ↓ \\
first_order_theorem_proving & AR & 0.3909 & 0.3902 & -0.0006 & ↓ \\
kr_vs_kp & AR & 0.8025 & 0.8079 & +0.0054 & ↑ \\
letter & AR & 0.9243 & 0.9247 & +0.0004 & ↑ \\
miceprotein & AR & 0.9702 & 0.9749 & +0.0047 & ↑ \\
pc4 & AR & 0.5449 & 0.5820 & +0.0371 & ↑ \\
pendigits & AR & 0.9857 & 0.9837 & -0.0020 & ↓ \\
phishingwebsites & AR & 0.8993 & 0.9052 & +0.0060 & ↑ \\
phoneme & AR & 0.7206 & 0.7187 & -0.0019 & ↓ \\
qsar_biodeg & AR & 0.7915 & 0.7719 & -0.0197 & ↓ \\
satimage & AR & 0.8786 & 0.8769 & -0.0017 & ↓ \\
segment & AR & 0.8972 & 0.9035 & +0.0063 & ↑ \\
sick & AR & 0.7327 & 0.7336 & +0.0009 & ↑ \\
spambase & AR & 0.8936 & 0.8938 & +0.0002 & ↑ \\
steel_plates_fault & AR & 0.7285 & 0.7358 & +0.0073 & ↑ \\
texture & AR & 0.9914 & 0.9930 & +0.0016 & ↑ \\
tic_tac_toe & AR & 0.8298 & 0.8366 & +0.0068 & ↑ \\
vehicle & AR & 0.7332 & 0.7263 & -0.0069 & ↓ \\
vowel & AR & 0.8123 & 0.8364 & +0.0241 & ↑ \\
wall_robot_navigation & AR & 0.8955 & 0.9072 & +0.0116 & ↑ \\
wilt & AR & 0.7530 & 0.7648 & +0.0117 & ↑ \\
banknote_authentication & NAR & 0.9911 & 0.9925 & +0.0014 & ↑ \\
car & NAR & 0.6455 & 0.6539 & +0.0085 & ↑ \\
climate_model_simulation_crashes & NAR & 0.9638 & 0.9697 & +0.0059 & ↑ \\
cmc & NAR & 0.3650 & 0.3513 & -0.0137 & ↓ \\
cylinder_bands & NAR & 0.7369 & 0.7612 & +0.0243 & ↑ \\
diabetes & NAR & 0.6227 & 0.6148 & -0.0079 & ↓ \\
dna & NAR & 0.6297 & 0.6457 & +0.0160 & ↑ \\
first_order_theorem_proving & NAR & 0.3759 & 0.3910 & +0.0152 & ↑ \\
kr_vs_kp & NAR & 0.8271 & 0.8333 & +0.0062 & ↑ \\
miceprotein & NAR & 0.9623 & 0.9709 & +0.0087 & ↑ \\
pc4 & NAR & 0.4878 & 0.4913 & +0.0035 & ↑ \\
pendigits & NAR & 0.9841 & 0.9817 & -0.0024 & ↓ \\
phishingwebsites & NAR & 0.8899 & 0.8966 & +0.0067 & ↑ \\
phoneme & NAR & 0.7169 & 0.7067 & -0.0102 & ↓ \\
qsar_biodeg & NAR & 0.8056 & 0.7987 & -0.0069 & ↓ \\
satimage & NAR & 0.8730 & 0.8766 & +0.0036 & ↑ \\
segment & NAR & 0.8762 & 0.8877 & +0.0114 & ↑ \\
sick & NAR & 0.7433 & 0.7799 & +0.0367 & ↑ \\
spambase & NAR & 0.9000 & 0.9027 & +0.0028 & ↑ \\
steel_plates_fault & NAR & 0.7329 & 0.7273 & -0.0056 & ↓ \\
texture & NAR & 0.9908 & 0.9914 & +0.0006 & ↑ \\
tic_tac_toe & NAR & 0.8462 & 0.8556 & +0.0094 & ↑ \\
vehicle & NAR & 0.7588 & 0.7530 & -0.0058 & ↓ \\
vowel & NAR & 0.7899 & 0.7917 & +0.0018 & ↑ \\
wall_robot_navigation & NAR & 0.8994 & 0.9037 & +0.0042 & ↑ \\
wilt & NAR & 0.7925 & 0.7918 & -0.0007 & ↓ \\
\bottomrule
\end{tabular}



════════════════════════════════════════════════════════════════════════
  EVIDENCE SUMMARY  [F1]
════════════════════════════════════════════════════════════════════════
\begin{tabular}{lllllll}
\toprule
Mode & Method & Macro mean & Macro med. & W/D/L & p (Wilcoxon) & |r| \\
\midrule
AR & Baseline & 0.7863 & 0.8025 & — & — & — \\
AR & + Gate & 0.7900 & 0.8079 & 18/0/9 & 0.1286 & 0.339 \\
AR & + Curriculum & 0.7878 & 0.7946 & 11/0/16 & 0.8779 & 0.037 \\
AR & Saga++ & 0.7605 & 0.7660 & 7/0/20 & 0.0065 & 0.587 \\
AR & CP prep & 0.7332 & 0.7612 & 7/0/20 & 0.0014 & 0.677 \\
NAR & Baseline & 0.7819 & 0.8056 & — & — & — \\
NAR & + Gate & 0.7816 & 0.7952 & 18/0/8 & 0.0594 & 0.425 \\
NAR & + Curriculum & 0.7720 & 0.7982 & 10/0/16 & 0.1291 & 0.345 \\
NAR & Saga++ & 0.7746 & 0.8100 & 16/0/11 & 0.3012 & 0.233 \\
NAR & CP prep & 0.6693 & 0.7428 & 4/0/23 & 0.0000 & 0.825 \\
\bottomrule
\end{tabular}


Generating plots…
  Saved: plots/gate_delta_boxplot.png
  Saved: plots/gate_noise_robustness.png
  Saved: plots/gate_behaviour.png
  Saved: plots/gate_training_efficiency.png
  Saved: plots/gate_rank_distribution.png

════════════════════════════════════════════════════════════
  RANK TABLE  [F1]
════════════════════════════════════════════════════════════

  AR (N=27, Friedman p=0.0000)
  Method                  Mean rank      Std  Best count
  ---------------------- ----------  -------  ----------
  Baseline                    2.593    1.309           7
  + Gate                      1.963    0.980          11
  + Curriculum                2.667    1.074           5
  Saga++                      3.889    1.340           2
  CP prep                     3.889    1.311           2

  NAR (N=26, Friedman p=0.0000)
  Method                  Mean rank      Std  Best count
  ---------------------- ----------  -------  ----------
  Baseline                    2.808    1.021           2
  + Gate                      2.269    1.538          12
  + Curriculum                2.962    1.076           3
  Saga++                      2.615    1.416           8
  CP prep                     4.346    1.093           1


════════════════════════════════════════════════════════════════════════
  POST-HOC NEMENYI TEST (after Friedman)  [F1]
════════════════════════════════════════════════════════════════════════

  AR  (N=27 datasets, 5 methods)
  Friedman chi2=31.674, p=0.0000

  Pairwise Nemenyi p-values:
                Baseline  + Gate  + Curriculum  Saga++  CP prep
  Baseline        1.0000  0.5866        0.9998  0.0218   0.0218
  + Gate          0.5866  1.0000        0.4747  0.0001   0.0001
  + Curriculum    0.9998  0.4747        1.0000  0.0364   0.0364
  Saga++          0.0218  0.0001        0.0364  1.0000   1.0000
  CP prep         0.0218  0.0001        0.0364  1.0000   1.0000

  Significant pairs (p < 0.05):
    Baseline       vs Saga++          p=0.0218
    Baseline       vs CP prep         p=0.0218
    + Gate         vs Saga++          p=0.0001
    + Gate         vs CP prep         p=0.0001
    + Curriculum   vs Saga++          p=0.0364
    + Curriculum   vs CP prep         p=0.0364

  Saved: plots/gate_nemenyi_ar.png

  NAR  (N=26 datasets, 5 methods)
  Friedman chi2=26.338, p=0.0000

  Pairwise Nemenyi p-values:
                Baseline  + Gate  + Curriculum  Saga++  CP prep
  Baseline        1.0000  0.7351        0.9968  0.9923   0.0041
  + Gate          0.7351  1.0000        0.5111  0.9338   0.0000
  + Curriculum    0.9968  0.5111        1.0000  0.9338   0.0138
  Saga++          0.9923  0.9338        0.9338  1.0000   0.0008
  CP prep         0.0041  0.0000        0.0138  0.0008   1.0000

  Significant pairs (p < 0.05):
    Baseline       vs CP prep         p=0.0041
    + Gate         vs CP prep         p=0.0000
    + Curriculum   vs CP prep         p=0.0138
    Saga++         vs CP prep         p=0.0008

  Saved: plots/gate_nemenyi_nar.png


Done. All plots saved to plots/
```
