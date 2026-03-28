import functools
import threading
import time
from collections import deque, Counter
from typing import Dict, List

from core.eeg_adapter import compute_score
from core.calibration import calibrate_thresholds, state_from_score
from core.hyperbitnet import HyperBitnet
from core.fusion import fusion_vector
from integration.tribe_adapter import to_tribe_command
from integration.neurosity_adapter import NeurosityAdapter, NeurosityConfig


class SlidingWindowStateFilter:
    def __init__(self, window_size: int = 8, min_votes: int = 5):
        self._window_size = window_size
        self.window: deque = deque(maxlen=window_size)
        self.min_votes = min_votes

    def push(self, state: int) -> int:
        self.window.append(state)
        if len(self.window) < self._window_size:
            return 0
        counts = Counter(self.window)
        winner, votes = counts.most_common(1)[0]
        return winner if votes >= self.min_votes else 0


def execute_pipeline(
    sample: Dict[str, float], low: float, high: float
) -> Dict[str, float]:
    score = compute_score(sample)
    raw_state = state_from_score(score, low, high)
    return {
        "score": score,
        "raw_state": raw_state,
        "focus": sample.get("focus", 0.0),
        "calm": sample.get("calm", 0.0),
        "gamma": sample.get("gamma", 0.0),
        "timestamp": sample.get("timestamp", time.time()),
    }


def execute_tribe(stable_state: int) -> Dict[str, object]:
    net = HyperBitnet(n_nodes=8)
    net.inject_state(stable_state)
    intent = fusion_vector(net.states, net.quantum_states)
    cmd = to_tribe_command(intent)
    return {
        "stable_state": stable_state,
        "intent_energy": sum(intent),
        "command": cmd["command"],
    }


def collect_baseline_from_stream(
    adapter: NeurosityAdapter, seconds: int = 8, hz: int = 10
) -> List[Dict[str, float]]:
    """Collect a resting-state baseline by streaming for `seconds` seconds."""
    buffer: List[Dict[str, float]] = []
    target = seconds * hz
    done_flag = {"done": False}

    def on_sample(sample: Dict[str, float]) -> None:
        if len(buffer) < target:
            buffer.append(sample)
        else:
            done_flag["done"] = True

    t = threading.Thread(
        target=functools.partial(adapter.start_stream, on_sample=on_sample, hz=float(hz)),
        daemon=True,
    )
    t.start()

    while not done_flag["done"]:
        time.sleep(0.05)

    adapter.stop_stream()
    t.join(timeout=1.0)
    return buffer


def main():
    # 1) Adapter config (mock stream enabled by default)
    config = NeurosityConfig(
        device_id="YOUR_DEVICE_ID",
        email="YOUR_EMAIL",
        password="YOUR_PASSWORD",
    )
    adapter = NeurosityAdapter(config=config, use_mock_stream=True)

    # 2) Connect
    adapter.connect()

    # 3) Initial calibration
    print("Collecting baseline... please relax / maintain neutral state.")
    baseline = collect_baseline_from_stream(adapter, seconds=8, hz=10)
    th = calibrate_thresholds(baseline)

    print("=== Calibration ===")
    print(
        {
            "samples": len(baseline),
            "baseline_mean": round(th.baseline_mean, 4),
            "baseline_std": round(th.baseline_std, 4),
            "low": round(th.low, 4),
            "high": round(th.high, 4),
        }
    )

    # 4) Realtime loop
    smoother = SlidingWindowStateFilter(window_size=8, min_votes=5)

    def on_live_sample(sample: Dict[str, float]) -> None:
        out = execute_pipeline(sample, th.low, th.high)
        stable_state = smoother.push(out["raw_state"])
        act = execute_tribe(stable_state)

        print(
            {
                "score": round(out["score"], 4),
                "raw_state": out["raw_state"],
                "stable_state": act["stable_state"],
                "intent_energy": round(act["intent_energy"], 4),
                "command": act["command"],
                "focus": round(out["focus"], 3),
                "gamma": round(out["gamma"], 3),
                "calm": round(out["calm"], 3),
            }
        )

    print("\nStarting realtime loop (Ctrl+C to stop)...")
    try:
        adapter.start_stream(on_sample=on_live_sample, hz=10.0)
    except KeyboardInterrupt:
        adapter.stop_stream()
        adapter.disconnect()
        print("\nShutdown complete.")


if __name__ == "__main__":
    main()
