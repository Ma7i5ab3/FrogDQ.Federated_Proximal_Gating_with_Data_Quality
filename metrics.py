import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from loguru import logger

def auc_conv(
    metric_values: Sequence[float],
    window_size: int = 50, 
    delta_threshold: float = 0.001,
    normalization: bool = True
) -> float:
    """
    Compute the discrete area under a validation metric curve until convergence.

    Convergence is considered reached when the absolute difference between
    successive values within the most recent `window_size` epochs
    falls below or equals `delta_threshold`.

    Parameters
    ----------
    metric_values : sequence of float
        Per-epoch validation scores for the selected metric.
    window_size : int
        Number of trailing values considered for the convergence check.
    delta_threshold : float
        Maximum allowed absolute change between consecutive metric values
        to mark convergence.
    normalization : bool, optional
        If True, normalize the AUC by the number of epochs considered
        (default is True).

    Returns
    -------
    float
        Area under the validation curve up to convergence (inclusive).
    """

    if window_size <= 1:
        raise ValueError(f"'window_size' must be > 1, got {window_size}.")
    if delta_threshold < 0:
        raise ValueError(f"'delta_threshold' must be non-negative, got {delta_threshold}.")
    if not metric_values:
        return 0.0

    values = np.asarray(metric_values, dtype=float).flatten()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0

    stop_idx = values.size
    if window_size <= values.size:
        for idx in range(window_size, values.size + 1):
            window = values[idx - window_size : idx]
            diffs = np.abs(np.diff(window))
            if np.all(diffs <= delta_threshold):
                stop_idx = idx
                break

    print(f"Convergence reached at index {stop_idx} out of {values.size}.")

    segment = values[:stop_idx]
    if segment.size == 1:
        return float(segment[0])

    auc_value = np.trapezoid(segment, dx=1.0)
    if normalization:
        auc_value /= segment.size

    return float(auc_value)


def epochs_to_impr_acc(
    metric_values: Sequence[float],
    improvement_ratio: float = 0.9
) -> float:
    """

    Compute the fraction of epochs required to reach a target improvement.
    The target improvement is defined as a percentage (`improvement_ratio`)
    of the total gain between the first recorded metric value and the best
    value observed. For example, if the metric starts at 0.50 and peaks at
    1.00, an `improvement_ratio` of 0.9 will set a target value of 0.95. The
    function then returns the fraction of epochs elapsed up to (and including)
    the first epoch in which the metric meets or exceeds that target.

    Parameters
    ----------
    metric_values : sequence of float
        Per-epoch validation scores for the selected metric.
    improvement_ratio : float
        Desired fraction of the achievable improvement in the metric. Must
        lie in the [0, 1] interval, where 0 corresponds to no improvement and
        1 to the best metric value observed.

    Returns
    -------
    float
        Fraction of epochs needed to reach the requested improvement. If the
        target is never achieved, the function returns 1.0.
    """
    if not 0 <= improvement_ratio <= 1:
        raise ValueError(
            f"'improvement_ratio' must lie in [0, 1], got {improvement_ratio}."
        )
    if not metric_values:
        raise ValueError("'metric_values' must contain at least one element.")
    
    values = np.asarray(metric_values, dtype=float)
    if values.ndim != 1:
        values = values.reshape(-1)
    
    finite_mask = np.isfinite(values)
    if not np.all(finite_mask):
        values = values[finite_mask]
    if values.size == 0:
        raise ValueError("All metric values are non-finite.")
    
    start_value = values[0]
    best_value = float(np.max(values))
    total_improvement = best_value - start_value
    if total_improvement <= 0 or improvement_ratio == 0:
        return 0.0
    target_value = start_value + total_improvement * improvement_ratio

    meet_target = values >= target_value
    if not np.any(meet_target):
        return 1.0
    target_index = int(np.argmax(meet_target)) # First index meeting the target
    epoch_fraction = (target_index + 1) / values.size

    return float(epoch_fraction)


def epochs_to_acc(
    metric_values: Sequence[float],
    acc_value: float = 0.8
) -> float:
    """

    Compute the fraction of epochs required to reach a accuracy value.
    The function returns the fraction of epochs elapsed up to (and including)
    the first epoch in which the metric meets or exceeds the specified
    accuracy value.

    Parameters
    ----------
    metric_values : sequence of float
        Per-epoch validation scores for the selected metric.
    acc_value : float
        Desired accuracy value to be reached. Must lie in the [0, 1] interval

    Returns
    -------
    float
        Fraction of epochs needed to reach the requested accuracy value. If the
        target is never achieved, the function returns 1.0.
    """
    if not 0 <= acc_value <= 1:
        raise ValueError(
            f"'acc_value' must lie in [0, 1], got {acc_value}."
        )
    if not metric_values:
        raise ValueError("'metric_values' must contain at least one element.")
    
    values = np.asarray(metric_values, dtype=float)
    if values.ndim != 1:
        values = values.reshape(-1)
    
    finite_mask = np.isfinite(values)
    if not np.all(finite_mask):
        values = values[finite_mask]
    if values.size == 0:
        raise ValueError("All metric values are non-finite.") 

    meet_target = values >= acc_value
    if not np.any(meet_target):
        return 1.0
    target_index = int(np.argmax(meet_target)) # First index meeting the target
    epoch_fraction = (target_index + 1) / values.size

    return float(epoch_fraction)


def max_value(
    metric_values: Sequence[float]
) -> float:
    """
    Compute the maximum value of a validation metric series.

    Parameters
    ----------
    metric_values : sequence of float
        Per-epoch validation scores for the selected metric.

    Returns
    -------
    float
        Maximum value observed in the metric series.
    """
    if not metric_values:
        raise ValueError("'metric_values' must contain at least one element.")
    
    values = np.asarray(metric_values, dtype=float)
    if values.ndim != 1:
        values = values.reshape(-1)
    
    finite_mask = np.isfinite(values)
    if not np.all(finite_mask):
        values = values[finite_mask]
    if values.size == 0:
        raise ValueError("All metric values are non-finite.") 

    return float(np.max(values))