from core.eeg_adapter import compute_score
from core.calibration import calibrate_thresholds, state_from_score
from core.hyperbitnet import HyperBitnet
from core.fusion import fusion_vector
from integration.tribe_adapter import to_tribe_command


def collect_baseline_windows():
    """
    Simulate 30 resting-state windows for calibration.
    In production these would come from the Crown/SDK during a brief
    eyes-closed baseline period.
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
    ] * 3  # 30 windows


def run_once(eeg_features, low, high):
    score = compute_score(eeg_features)
    state = state_from_score(score, low, high)

    net = HyperBitnet(n_nodes=8)
    net.inject_state(state)

    intent = fusion_vector(net.states, net.quantum_states)
    command = to_tribe_command(intent)

    return {
        "score": round(score, 4),
        "state": state,
        "intent_energy": round(sum(intent), 4),
        "command": command["command"],
    }


if __name__ == "__main__":
    # 1) Initial calibration
    baseline = collect_baseline_windows()
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

    # 2) Sample real-time inference
    live_samples = [
        {"focus": 0.40, "gamma": 0.35, "calm": 0.50},  # likely transition
        {"focus": 0.82, "gamma": 0.78, "calm": 0.20},  # likely confirmed intent
        {"focus": 0.22, "gamma": 0.18, "calm": 0.70},  # likely disconnection
    ]

    print("\n=== Live Inference ===")
    for i, sample in enumerate(live_samples, start=1):
        out = run_once(sample, th.low, th.high)
        print(f"sample_{i}:", out)
