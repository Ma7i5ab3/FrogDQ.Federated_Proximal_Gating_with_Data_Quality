# CtxPipe agent weights

The six `ctx_32000_*.pkl` files are the DQN agents released with CtxPipe
(Gao et al., SIGMOD '25, https://doi.org/10.1145/3698831), copied unchanged from
the official repository:

- repository: https://github.com/ctxpipe/ctxpipe
- commit: `79caaa17f17ebdeeac6ba549abe150c5b3f1381d`
- path: `models/ctxpipe-3linear/`

They are the agents with the context plug-in and the Open-Closed-Gated experience
replay enabled (upstream's `-3linear` setup), trained for 32,000 steps on the
HAIPipe corpus. That is the configuration the paper evaluates (Section 6,
"Agent Optimization"). `scripts/ctxpipe.py` loads them with `strict=True` and
never updates them.

| File | Agent | Actions |
|------|-------|---------|
| `ctx_32000_logical_pipeline.pkl` | logical pipeline (DQN) | 6 component-type orders |
| `ctx_32000_imputernum_model.pkl` | ImputerNum (RnnDQN) | 3 |
| `ctx_32000_encoder_model.pkl` | Encoder (RnnDQN) | 3 |
| `ctx_32000_fpreprocessing_model.pkl` | FeaturePreprocessing (RnnDQN) | 9 |
| `ctx_32000_fengine_model.pkl` | FeatureEngine (RnnDQN) | 8 |
| `ctx_32000_fselection_model.pkl` | FeatureSelection (RnnDQN) | 2 |

SHA-256:

```
e40cd97c3dd85787bfe898959cb2ff854870d4c690213fca0be7cf3d9058d26d  ctx_32000_logical_pipeline.pkl
5543afd7b45c073dda4278b70a0e1730757a10ee31b4349295ce5145a326eaea  ctx_32000_imputernum_model.pkl
6bc5265ab2751cdd99767d821f43cd942e96e30ec4691d6420ecb239f5218cc5  ctx_32000_encoder_model.pkl
4c4d445a003b05ee22ef299e633e06a1a807a4f5ab4f496fff36f177080cb58b  ctx_32000_fpreprocessing_model.pkl
c4423961ad46de8fd03d2d79615028b8c529d415536b10eb222de050fb6c892e  ctx_32000_fengine_model.pkl
80c9b9eaa232099f99a9cc3a160c16a1c190087cd2766fac4120202982a210cf  ctx_32000_fselection_model.pkl
```

The context gate matrices (`context_gate`, `context_gate_bias`) are not in these
files: upstream stores them as plain tensors rather than `nn.Parameter`s, so they
are neither trained nor saved (see the CTX-PORT notes in `scripts/ctxpipe.py`).

The weights and the code they come from are licensed under the Apache License
2.0 — see `LICENSE` and `NOTICE` in this directory.
