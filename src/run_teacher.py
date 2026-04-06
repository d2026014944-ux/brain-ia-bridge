from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from core.lif_neuron import LIFNeuron
from core.spiking_network import SpikingNetwork
from core.subliminal_learning import AITeacher


GRID_SIZE = 8
N_NEURONS = GRID_SIZE * GRID_SIZE
DEFAULT_STATE_FILE = Path(__file__).resolve().parent / "mind_panel_state.json"


def _node_xy(node_id: int) -> tuple[float, float]:
    row = node_id // GRID_SIZE
    col = node_id % GRID_SIZE
    return float(col), float(row)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload), encoding="utf-8")
    temp_path.replace(path)


def _build_network(initial_weight: float, fanout: int, seed: int) -> SpikingNetwork:
    rng = random.Random(seed)
    network = SpikingNetwork(learning_enabled=True)

    for neuron_id in range(N_NEURONS):
        network.add_neuron(
            node_id=neuron_id,
            neuron_instance=LIFNeuron(v_thresh=1.0, tau=20.0, refractory_period=5.0),
        )

    fanout = max(1, min(fanout, N_NEURONS - 1))
    for pre_id in range(N_NEURONS):
        candidates = [idx for idx in range(N_NEURONS) if idx != pre_id]
        targets = rng.sample(candidates, k=fanout)
        for post_id in targets:
            delay_ms = rng.choice((1.0, 2.0, 3.0))
            network.add_connection(
                pre_id=pre_id,
                post_id=post_id,
                weight=initial_weight,
                delay_ms=delay_ms,
            )

    return network


def _run_until_empty_with_learning(network: SpikingNetwork, learning_enabled: bool) -> None:
    # Backward-compatible: some bindings expose only run_until_empty() without kwargs.
    try:
        network.run_until_empty(learning_enabled=learning_enabled)
        return
    except TypeError:
        pass

    network.learning_enabled = learning_enabled
    network.run_until_empty()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AI Teacher driving a plastic 8x8 spiking grid")
    parser.add_argument("--teacher-hz", type=float, default=40.0)
    parser.add_argument("--tick-ms", type=float, default=100.0)
    parser.add_argument("--teacher-weight", type=float, default=1.05)
    parser.add_argument("--initial-weight", type=float, default=0.2)
    parser.add_argument("--fanout", type=int, default=4)
    parser.add_argument("--input-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional debug cap. Omit for infinite loop.",
    )
    args = parser.parse_args()

    if args.tick_ms <= 0.0:
        raise ValueError("--tick-ms must be > 0")
    if args.input_count <= 0 or args.input_count > N_NEURONS:
        raise ValueError(f"--input-count must be in [1, {N_NEURONS}]")

    teacher = AITeacher(
        teacher_hz=args.teacher_hz,
        target_id=0,
        spike_weight=args.teacher_weight,
    )
    network = _build_network(initial_weight=args.initial_weight, fanout=args.fanout, seed=args.seed)

    input_neurons = list(range(args.input_count))
    nodes = [
        {
            "id": node_id,
            "x": _node_xy(node_id)[0],
            "y": _node_xy(node_id)[1],
        }
        for node_id in range(N_NEURONS)
    ]

    sim_t_ms = 0.0
    step = 0

    print("AI Teacher online")
    print(f"- teacher_hz: {args.teacher_hz}")
    print(f"- tick_ms: {args.tick_ms}")
    print(f"- input_neurons: {input_neurons}")
    print(f"- state_file: {args.state_file}")

    try:
        while args.max_steps is None or step < args.max_steps:
            gamma_train = teacher.generate_gamma_train(duration_ms=args.tick_ms)
            for offset_ms, _, spike_weight in gamma_train:
                event_t = sim_t_ms + float(offset_ms)
                for target_id in input_neurons:
                    network.schedule_event(time_ms=event_t, target_id=target_id, weight=float(spike_weight))

            _run_until_empty_with_learning(network, learning_enabled=True)

            snapshot = {
                "timestamp": time.time(),
                "learning_enabled": bool(network.learning_enabled),
                "nodes": nodes,
                "synapses": network.get_synaptic_strengths(),
                "meta": {
                    "step": step,
                    "sim_t_ms": round(sim_t_ms, 3),
                    "teacher_hz": args.teacher_hz,
                    "tick_ms": args.tick_ms,
                    "teacher_events_per_tick": len(gamma_train) * len(input_neurons),
                },
            }
            _atomic_write_json(args.state_file, snapshot)

            if step % 10 == 0:
                mean_weight = 0.0
                synapses = snapshot["synapses"]
                if synapses:
                    mean_weight = sum(float(s["weight"]) for s in synapses) / len(synapses)
                print(
                    {
                        "step": step,
                        "mean_weight": round(mean_weight, 4),
                        "synapses": len(synapses),
                        "teacher_events": len(gamma_train) * len(input_neurons),
                    }
                )

            sim_t_ms += args.tick_ms
            step += 1
            time.sleep(args.tick_ms / 1000.0)
    except KeyboardInterrupt:
        print("Teacher loop interrupted by user.")


if __name__ == "__main__":
    main()
