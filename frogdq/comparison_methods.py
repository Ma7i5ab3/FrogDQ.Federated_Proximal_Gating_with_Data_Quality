"""
Canonical vocabulary for the competitor "methods"/baselines compared across
scripts/evaluate.py, scripts/generate_latex_tables.py, and the analysis/*.py
scripts (gate_analysis.py, optuna_progress.py, run_gate_evidence_report.py).

Each of those scripts represents "which method a row belongs to" differently
(a method_label() string, a study-name "config" token, or a
(data_mode, model_type, use_curriculum, use_gate) column tuple), so this
module defines one canonical set of method keys and a `resolve_*` function
per representation, all mapping into that same canonical vocabulary. A single
`comparison_methods` list (read from config.yaml, or overridden with
--comparison-methods on the CLI) can then restrict every script to the same
chosen subset.

"clean" is just another canonical key here — it is NOT protected/forced-on.
If a user's comparison_methods list omits "clean", the un-poisoned reference
is dropped from evaluate.py/optuna_progress.py/generate_latex_tables.py too.
gate_analysis.py is the one exception: its noise-robustness analysis is
built around measuring degradation *from* clean, so its "Clean" reference
lookup stays structurally always-on regardless of this setting (see the
module docstring in gate_analysis.py).
"""

from pathlib import Path
from typing import Dict, List, Optional

import yaml

# Canonical method keys, in the order they should be displayed/considered.
METHOD_CHOICES: List[str] = [
    "clean",
    "baseline",
    "curriculum",
    "gate",
    "gate_curriculum",
    "autogluon",
    "saga",
    "cp",
    "baseline_zero",
    "knn",
    "catboost_clean",
    "catboost_dirty",
]

# study-name "config" token (gate_analysis.py / optuna_progress.py /
# run_gate_evidence_report.py) -> canonical key.
# "curr0_gate0" is ambiguous on its own — it means "clean" at data_mode ==
# "clean" and "baseline" everywhere else — so it is deliberately absent here;
# callers resolve it via resolve_from_config_token(), which takes data_mode.
_CONFIG_TOKEN_TO_METHOD: Dict[str, str] = {
    "curr1_gate0": "curriculum",
    "curr0_gate1": "gate",
    "curr1_gate1": "gate_curriculum",
    "ag": "autogluon",
    "saga": "saga",
    "cp": "cp",
    "baseline_zero": "baseline_zero",
    "knn": "knn",
    "catboost_clean": "catboost_clean",
    "catboost_dirty": "catboost_dirty",
}

# evaluate.py's method_label() string -> canonical key.
_LABEL_TO_METHOD: Dict[str, str] = {
    "clean": "clean",
    "baseline": "baseline",
    "curriculum": "curriculum",
    "gate": "gate",
    "gate+curriculum": "gate_curriculum",
    "autogluon": "autogluon",
    "saga": "saga",
    "cp": "cp",
    "baseline_zero": "baseline_zero",
    "knn": "knn",
    "catboost_clean": "catboost_clean",
    "catboost_dirty": "catboost_dirty",
}


def resolve_from_config_token(config_token: str, data_mode: str) -> str:
    """Map a gate_analysis.py/optuna_progress.py 'config' token to a canonical key."""
    if config_token == "curr0_gate0":
        return "clean" if data_mode == "clean" else "baseline"
    return _CONFIG_TOKEN_TO_METHOD.get(config_token, config_token)


def resolve_from_label(label: str) -> str:
    """Map an evaluate.py method_label() string to a canonical key."""
    return _LABEL_TO_METHOD.get(label, label)


def resolve_from_column(data_mode: str, model_type: str, use_curriculum, use_gate) -> str:
    """Map a generate_latex_tables.py (data_mode, model_type, use_curr, use_gate) column to a canonical key."""
    if model_type == "catboost":
        return "catboost_clean" if data_mode == "clean" else "catboost_dirty"
    if use_curriculum == "aggregated" or (use_curriculum and use_gate):
        return "gate_curriculum"
    if use_gate:
        return "gate"
    if use_curriculum:
        return "curriculum"
    return "clean" if data_mode == "clean" else "baseline"


def is_selected(canonical: str, selected: Optional[List[str]]) -> bool:
    """Whether a canonical method key should be kept, given a comparison_methods selection."""
    return selected is None or canonical in selected


def load_comparison_methods(config_path: str = "config.yaml") -> Optional[List[str]]:
    """
    Read the `comparison_methods` key from a FrogDQ config.yaml.

    Returns
    -------
    None
        If the key is absent, null, or the file doesn't exist — meaning
        "include every method" (the backward-compatible default).
    List[str]
        The requested canonical method keys (see METHOD_CHOICES).

    Raises
    ------
    ValueError
        If the list contains a key not in METHOD_CHOICES.
    """
    path = Path(config_path)
    if not path.exists():
        return None
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    methods = cfg.get("comparison_methods")
    if not methods:
        return None
    unknown = sorted(set(methods) - set(METHOD_CHOICES))
    if unknown:
        raise ValueError(
            f"Unknown comparison_methods entries: {unknown}. Valid choices: {METHOD_CHOICES}"
        )
    return list(methods)


def resolve_comparison_methods(
    cli_methods: Optional[List[str]], config_path: str = "config.yaml"
) -> Optional[List[str]]:
    """CLI --comparison-methods override takes precedence over config.yaml; both default to 'all'."""
    if cli_methods:
        return list(cli_methods)
    return load_comparison_methods(config_path)
