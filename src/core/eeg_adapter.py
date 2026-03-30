from typing import Dict


def compute_score(features: Dict[str, float]) -> float:
    """
    Compute a scalar activation score from EEG-derived features.

    Weights:
      focus  -> 0.5  (primary intent signal)
      gamma  -> 0.3  (cognitive engagement)
      calm   -> 0.2  (inverse: low calm = high arousal)

    Returns a value in [0, 1].
    """
    focus = float(features.get("focus", 0.0))
    gamma = float(features.get("gamma", 0.0))
    calm = float(features.get("calm", 0.0))

    score = focus * 0.5 + gamma * 0.3 + (1.0 - calm) * 0.2
    return max(0.0, min(1.0, score))
