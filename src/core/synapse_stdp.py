import math


class SynapseSTDP:
    tau = 20.0
    A_plus = 0.02
    A_minus = 0.025
    W_MAX = 1.0
    W_MIN = 0.0

    def __init__(self, weight: float = 0.5):
        self.weight = float(weight)
        self.last_pre_t_ms = None
        self.last_post_t_ms = None
        self.weight = max(self.W_MIN, min(self.W_MAX, self.weight))

    def _apply_delta(self, delta: float) -> None:
        self.weight = max(self.W_MIN, min(self.W_MAX, self.weight + delta))

    def register_pre_spike(self, time: float) -> None:
        self.last_pre_t_ms = float(time)
        if self.last_post_t_ms is None:
            return
        dt = self.last_post_t_ms - self.last_pre_t_ms
        if dt > 0:
            delta = self.A_plus * math.exp(-dt / self.tau)
        else:
            delta = -self.A_minus * math.exp(dt / self.tau)
        self._apply_delta(delta)

    def register_post_spike(self, time: float) -> None:
        self.last_post_t_ms = float(time)
        if self.last_pre_t_ms is None:
            return
        dt = self.last_post_t_ms - self.last_pre_t_ms
        if dt > 0:
            delta = self.A_plus * math.exp(-dt / self.tau)
        else:
            delta = -self.A_minus * math.exp(dt / self.tau)
        self._apply_delta(delta)

    def update(self, pre_t_ms: float, post_t_ms: float) -> None:
        self.register_pre_spike(pre_t_ms)
        self.register_post_spike(post_t_ms)
