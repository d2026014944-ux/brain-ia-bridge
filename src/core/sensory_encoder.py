from typing import List, Tuple

from core.hyperbitnet import HyperBitnet


class SensoryEncoder:
    """
    Convert HyperBitnet activations into spike events using rate coding.

    Each event is represented as (node_id, spike_time_ms).
    """

    MAX_FREQ = 100.0

    def __init__(self, net: HyperBitnet | None = None):
        self._net = net

    def encode(
        self, duration_ms: int, net: HyperBitnet | None = None
    ) -> List[Tuple[int, float]]:
        source_net = net or self._net
        if source_net is None:
            raise ValueError("SensoryEncoder requires a HyperBitnet instance.")

        events: List[Tuple[int, float]] = []
        for node_id, quantum_state in enumerate(source_net.quantum_states):
            freq = max(0.0, float(quantum_state) * self.MAX_FREQ)
            if freq <= 0.0:
                continue

            interval_ms = 1000.0 / freq
            classical_state = float(source_net.states[node_id])
            jitter_ms = (classical_state - 0.5) * 0.2 * interval_ms

            t_ms = interval_ms + jitter_ms
            while t_ms < float(duration_ms):
                events.append((node_id, t_ms))
                t_ms += interval_ms

        events.sort(key=lambda event: event[1])
        return events
