import math
from dataclasses import dataclass
from typing import Dict, List

from core.eeg_adapter import compute_score


@dataclass
class Thresholds:
    baseline_mean: float
    baseline_std: float
    low: float   # above this -> ambiguous transition
    high: float  # above this -> confirmed intent


def calibrate_thresholds(
    windows: List[Dict[str, float]], n_std: float = 1.0
) -> Thresholds:
    """
    Compute adaptive thresholds from a list of resting-state EEG feature windows.

    low  = baseline_mean + 1 * n_std * baseline_std  (transition boundary)
    high = baseline_mean + 2 * n_std * baseline_std  (confirmation boundary)
    """
    if not windows:
        raise ValueError("windows must not be empty")

    scores = [compute_score(w) for w in windows]
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    std = math.sqrt(variance)

    return Thresholds(
        baseline_mean=mean,
        baseline_std=std,
        low=mean + n_std * std,
        high=mean + 2.0 * n_std * std,
    )


def state_from_score(score: float, low: float, high: float) -> int:
    """
    Map a scalar score to a discrete intent state.

    Returns:
        1   confirmed intent   (score >= high)
        0   ambiguous          (low <= score < high)
       -1   disconnection/idle (score < low)
    """
    if score >= high:
        return 1
    if score >= low:
        return 0
    return -1
