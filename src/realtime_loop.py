import time
from collections import deque, Counter
from typing import Dict, Iterable, List

from core.eeg_adapter import compute_score
from core.calibration import calibrate_thresholds, state_from_score
from core.hyperbitnet import HyperBitnet
from core.fusion import fusion_vector
from integration.tribe_adapter import to_tribe_command


class SlidingWindowStateFilter:
    """
    Smooth instantaneous states (-1, 0, 1) with a sliding window to avoid
    firing on isolated spikes.
    """

    def __init__(self, window_size: int = 8, min_votes: int = 5):
        self._window_size = window_size
        self.window: deque = deque(maxlen=window_size)
        self.min_votes = min_votes

    def push(self, state: int) -> int:
        self.window.append(state)
        if len(self.window) < self._window_size:
            return 0  # still warming up

        counts = Counter(self.window)
        most_common_state, votes = counts.most_common(1)[0]

        if votes >= self.min_votes:
            return most_common_state
        return 0  # no consensus


def run_pipeline(
    eeg_features: Dict[str, float], low: float, high: float
) -> Dict[str, float]:
    score = compute_score(eeg_features)
    raw_state = state_from_score(score, low, high)
    return {"score": score, "raw_state": raw_state}


def execute_state(stable_state: int) -> Dict[str, object]:
    net = HyperBitnet(n_nodes=8)
    net.inject_state(stable_state)
    intent = fusion_vector(net.states, net.quantum_states)
    command = to_tribe_command(intent)
    return {
        "stable_state": stable_state,
        "intent_energy": round(sum(intent), 4),
        "command": command["command"],
    }


def simulated_eeg_stream() -> Iterable[Dict[str, float]]:
    """
    Simulate a stream with four phases:
    1) neutral
    2) rising intent
    3) strong intent
    4) return to neutral
    """
    neutral = [{"focus": 0.34, "gamma": 0.30, "calm": 0.60}] * 12
    ramp = [
        {"focus": 0.45, "gamma": 0.40, "calm": 0.50},
        {"focus": 0.52, "gamma": 0.48, "calm": 0.42},
        {"focus": 0.60, "gamma": 0.56, "calm": 0.36},
        {"focus": 0.68, "gamma": 0.63, "calm": 0.30},
    ] * 2
    strong = [{"focus": 0.82, "gamma": 0.78, "calm": 0.20}] * 10
    back = [{"focus": 0.36, "gamma": 0.32, "calm": 0.58}] * 12

    yield from neutral + ramp + strong + back


def collect_baseline() -> List[Dict[str, float]]:
    """
    Resting-state baseline used for calibration.
    In production: collect 5–20 seconds from the user.
    """
    return [
        {"focus": 0.30, "gamma": 0.28, "calm": 0.62},
        {"focus": 0.35, "gamma": 0.32, "calm": 0.60},
        {"focus": 0.33, "gamma": 0.30, "calm": 0.58},
        {"focus": 0.29, "gamma": 0.27, "calm": 0.65},
        {"focus": 0.31, "gamma": 0.29, "calm": 0.61},
        {"focus": 0.34, "gamma": 0.31, "calm": 0.59},
        {"focus": 0.32, "gamma": 0.30, "calm": 0.60},
        {"focus": 0.28, "gamma": 0.26, "calm": 0.66},
        {"focus": 0.36, "gamma": 0.33, "calm": 0.57},
        {"focus": 0.30, "gamma": 0.29, "calm": 0.63},
    ] * 3


def main():
    # 1) Calibration
    baseline = collect_baseline()
    th = calibrate_thresholds(baseline)

    print("=== Calibration ===")
    print(
        {
            "baseline_mean": round(th.baseline_mean, 4),
            "baseline_std": round(th.baseline_std, 4),
            "low": round(th.low, 4),
            "high": round(th.high, 4),
        }
    )

    # 2) Temporal stability filter
    smoother = SlidingWindowStateFilter(window_size=8, min_votes=5)

    print("\n=== Realtime Loop ===")
    for tick, eeg in enumerate(simulated_eeg_stream(), start=1):
        out = run_pipeline(eeg, th.low, th.high)
        stable_state = smoother.push(out["raw_state"])
        act = execute_state(stable_state)

        print(
            {
                "tick": tick,
                "score": round(out["score"], 4),
                "raw_state": out["raw_state"],
                "stable_state": act["stable_state"],
                "intent_energy": act["intent_energy"],
                "command": act["command"],
            }
        )

        # Simulate ~10 Hz update rate
        time.sleep(0.1)


if __name__ == "__main__":
    main()
