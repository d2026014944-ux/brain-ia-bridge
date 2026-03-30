import math
import random
from typing import List


class HyperBitnet:
    """
    Minimal HyperBitnet: a network of nodes with classical and quantum-inspired
    state vectors.

    Classical states are biased by an intent value and perturbed by small noise.
    Quantum states are derived via a cosine projection to model superposition.
    """

    def __init__(self, n_nodes: int = 8, seed: int | None = None):
        self.n_nodes = n_nodes
        self._rng = random.Random(seed)
        self.states: List[float] = [0.0] * n_nodes
        self.quantum_states: List[float] = [0.5] * n_nodes

    def inject_state(self, intent_state: int) -> None:
        """
        Propagate a discrete intent state (-1, 0, 1) into all nodes.

        intent_state is mapped to a [0, 1] bias:
          -1 -> 0.0  (idle / disconnected)
           0 -> 0.5  (ambiguous)
           1 -> 1.0  (confirmed intent)
        """
        bias = (intent_state + 1) / 2.0  # maps {-1, 0, 1} -> {0.0, 0.5, 1.0}
        for i in range(self.n_nodes):
            noise = self._rng.gauss(0.0, 0.05)
            classical = max(0.0, min(1.0, bias + noise))
            self.states[i] = classical
            # Quantum projection: high classical activation -> high quantum coherence
            # angle sweeps from π (low activation) to 0 (full activation)
            angle = math.pi * (1.0 - classical)
            self.quantum_states[i] = (1.0 + math.cos(angle)) / 2.0
