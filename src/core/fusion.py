from typing import List


def fusion_vector(
    states: List[float], quantum_states: List[float]
) -> List[float]:
    """
    Fuse classical and quantum-inspired node states into a single intent vector.

    Each output element is the arithmetic mean of the corresponding classical
    and quantum state, keeping values in [0, 1].
    """
    return [(s + q) / 2.0 for s, q in zip(states, quantum_states)]
