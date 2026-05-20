"""
brain-ia-bridge — Pipeline principal unificado.

Fluxo completo:
  1. Calibração adaptativa com detecção automática de domínio.
  2. Inferência ao vivo (Neurosity focus/calm/gamma).
  3. Simulação avançada HyperBitnet com fusão matricial (se deps disponíveis).
  4. Execução em tempo real MNHI 4.0 via NomaCore.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from core.calibration import calibrate_thresholds, state_from_score
from core.eeg_adapter import compute_score
from core.fusion import fusion_vector
from core.hyperbitnet import HyperBitnet
from integration.tribe_adapter import to_tribe_command
from core.noma_core import DEFAULT_MEMORY_FILE, DEFAULT_STATE_FILE, NomaCore

# Advanced mode check
try:
    import numpy as np
    from core.fusion import full_fusion
    from core.hyperbitnet import bitnet_efficient_matrix

    _HAS_ADVANCED = True
except ImportError:
    _HAS_ADVANCED = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NUM_NODES = 8
NUM_EDGES = 12
SIM_STEPS = 5


def collect_baseline_windows():
    """Simulate 30 resting-state windows for calibration."""
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


def run_mvp_demo():
    """MVP demo: calibration + live inference + TRIBE translation."""
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
        score = compute_score(sample)
        state = state_from_score(score, th.low, th.high)

        net = HyperBitnet(n_nodes=NUM_NODES)
        net.inject_state(state)
        intent = fusion_vector(net.states, net.quantum_states)
        command = to_tribe_command(intent)

        print(
            f"sample_{i}:",
            {
                "score": round(score, 4),
                "state": state,
                "intent_energy": round(sum(intent), 4),
                "command": command["command"],
            },
        )

    return th


def run_advanced_simulation():
    """Advanced HyperBitnet simulation with quantum graph and matrix fusion."""
    if not _HAS_ADVANCED:
        print(
            "\n[Modo avançado indisponível — instale numpy, networkx, scipy]",
            file=sys.stderr,
        )
        return

    print("\n=== Advanced HyperBitnet Simulation ===")

    # 1) Build quantum graph
    print(f"\n[1] Inicializando HyperBitnet ({NUM_NODES} nós, {NUM_EDGES} arestas)...")
    hbn = HyperBitnet(num_nodes=NUM_NODES)
    hbn.connect_quantum_nodes(num_edges=NUM_EDGES)

    print(f"    -> Arestas efetivas: {hbn.graph.number_of_edges()}")
    print(f"    -> Estados iniciais:  {hbn.get_state_vector()}")
    print(f"    -> Q-states iniciais: {np.round(hbn.get_quantum_vector(), 3)}")

    # 2) Quantum simulation
    print(f"\n[2] Executando simulação quântica ({SIM_STEPS} passos)...")
    hbn.run_quantum_simulation(num_steps=SIM_STEPS)

    print(f"    -> Estados pós-sim:  {hbn.get_state_vector()}")
    print(f"    -> Q-states pós-sim: {np.round(hbn.get_quantum_vector(), 3)}")

    # 3) Matrix fusion
    print("\n[3] Fusão Matricial BitNet x HyperBitnet...")
    fusion_matrix = full_fusion(hbn)

    nonzero = int(np.count_nonzero(fusion_matrix))
    energy = float(np.sum(fusion_matrix))
    print(f"    -> Dimensão:      {fusion_matrix.shape}")
    print(f"    -> Elementos != 0: {nonzero}")
    print(f"    -> Energia total:  {energy:.6f}")

    # 4) TRIBE command from fused vector
    fused = fusion_vector(hbn.states, hbn.quantum_states)
    tribe_cmd = to_tribe_command(fused)
    print(f"\n[4] Comando TRIBE: {tribe_cmd}")

    print("\n✓ Simulação avançada completa.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MNHI 4.0 unified runtime")
    parser.add_argument("--tick-ms", type=float, default=100.0, help="Main loop interval in milliseconds.")
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE, help="Realtime state JSON for Mind Panel.")
    parser.add_argument("--memory-file", type=Path, default=DEFAULT_MEMORY_FILE, help="Persistent synaptic memory binary.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional debug cap. Omit for continuous runtime.")
    parser.add_argument("--demo", action="store_true", help="Executa a demo MVP (calibração e inferência ao vivo) e encerra.")
    parser.add_argument("--advanced", action="store_true", help="Executa a simulação avançada HyperBitnet e encerra.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.demo:
        run_mvp_demo()
        return

    if args.advanced:
        run_advanced_simulation()
        return

    core = NomaCore(
        state_file=args.state_file,
        memory_file=args.memory_file,
        tick_ms=args.tick_ms,
    )
    core.boot_log()

    steps_executed = 0
    try:
        while args.max_steps is None or steps_executed < args.max_steps:
            cycle_start = time.perf_counter()
            core.step()
            steps_executed += 1

            elapsed_ms = (time.perf_counter() - cycle_start) * 1000.0
            remaining_ms = args.tick_ms - elapsed_ms
            if remaining_ms > 0.0:
                time.sleep(remaining_ms / 1000.0)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt recebido. Encerrando MNHI 4.0...")
    finally:
        try:
            core.shutdown()
            print(f"Memoria salva em {args.memory_file}")
        except Exception as exc:
            print(f"Falha ao salvar memoria em {args.memory_file}: {exc}")


if __name__ == "__main__":
    main()
