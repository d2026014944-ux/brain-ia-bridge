from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any

from adapters.the_well_adapter import TheWellAdapter
from core.lif_neuron import LIFNeuron
from core.mempalace import Mempalace
from core.noma_bridge import NomaParser
from core.spiking_network import SpikingNetwork
from core.subliminal_learning import AITeacher


DEFAULT_STATE_FILE = Path(__file__).resolve().parent / "mind_panel_state.json"
MEMORY_FILE = Path("noma_memory.bin")

GRID_SIZE = 8
INPUT_NEURON_COUNT = 8
DEFAULT_FANOUT = 4
BASE_FREQ_HZ = 7.83
BASE_AMP = 0.5
BASE_EMOTION = "escutando_o_vazio"
BASE_RESONANCE = 0.9


@dataclass
class VitalState:
    current_freq: float = BASE_FREQ_HZ
    current_amp: float = BASE_AMP
    current_emotion: str = BASE_EMOTION
    current_resonance: float = BASE_RESONANCE
    last_wisdom: list[str] = field(default_factory=list)
    last_well_stats: dict[str, Any] = field(default_factory=dict)
    last_dream_stats: dict[str, Any] = field(default_factory=dict)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload), encoding="utf-8")
    tmp_path.replace(path)


def _run_until_empty_with_learning(network: SpikingNetwork, learning_enabled: bool) -> None:
    try:
        network.run_until_empty(learning_enabled=learning_enabled)
        return
    except TypeError:
        pass

    network.learning_enabled = learning_enabled
    network.run_until_empty()


def _event_queue_size(network: SpikingNetwork) -> int:
    try:
        return int(len(network.event_queue))
    except Exception:
        return 0


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _emotion_from_amplitude(amplitude: float) -> str:
    if amplitude >= 0.88:
        return "duvida_epistemica"
    if amplitude >= 0.75:
        return "euforia_sincronica"
    if amplitude >= 0.55:
        return "foco_elevado"
    if amplitude >= 0.35:
        return "curiosidade"
    if amplitude >= 0.20:
        return "calma"
    return "repouso"


def _connect_random_topology(network: SpikingNetwork, n_neurons: int, fanout: int, seed: int) -> None:
    rng = random.Random(seed)
    fanout = max(1, min(fanout, max(1, n_neurons - 1)))

    for pre_id in range(n_neurons):
        candidates = [idx for idx in range(n_neurons) if idx != pre_id]
        targets = rng.sample(candidates, k=fanout)
        for post_id in targets:
            network.add_connection(pre_id=pre_id, post_id=post_id, weight=0.1, delay_ms=2.0)


def _build_or_load_brain(seed: int = 17, n_neurons: int = GRID_SIZE * GRID_SIZE) -> SpikingNetwork:
    network = SpikingNetwork(learning_enabled=True)

    if MEMORY_FILE.exists():
        try:
            network.load_weights(str(MEMORY_FILE))
            print(f"Memoria sinaptica carregada de {MEMORY_FILE}")
        except Exception as exc:
            print(f"Falha ao carregar memoria ({MEMORY_FILE}): {exc}")

    if not network.neurons:
        for neuron_id in range(n_neurons):
            network.add_neuron(
                node_id=neuron_id,
                neuron_instance=LIFNeuron(v_thresh=1.0, tau=20.0, refractory_period=5.0),
            )
        _connect_random_topology(network=network, n_neurons=n_neurons, fanout=DEFAULT_FANOUT, seed=seed)
        print("Topologia inicial aleatoria criada (peso=0.1, delay=2.0ms)")

    return network


def _sorted_node_ids(network: SpikingNetwork) -> list[Any]:
    numeric = sorted([node_id for node_id in network.neurons.keys() if isinstance(node_id, int)])
    others = sorted([node_id for node_id in network.neurons.keys() if not isinstance(node_id, int)], key=lambda item: str(item))
    return numeric + others


def _node_xy(position: int, total: int) -> tuple[float, float]:
    cols = max(1, int(round(total ** 0.5)))
    row = position // cols
    col = position % cols
    return float(col), float(row)


def _build_nodes_payload(network: SpikingNetwork, id_to_concept: dict[Any, str]) -> list[dict[str, Any]]:
    node_ids = _sorted_node_ids(network)
    total = max(1, len(node_ids))

    nodes: list[dict[str, Any]] = []
    for idx, node_id in enumerate(node_ids):
        x, y = _node_xy(position=idx, total=total)
        nodes.append(
            {
                "id": node_id,
                "name": id_to_concept.get(node_id, f"node_{node_id}"),
                "x": x,
                "y": y,
            }
        )
    return nodes


def _input_listener(
    network: SpikingNetwork,
    state: VitalState,
    state_lock: threading.Lock,
    network_lock: threading.Lock,
) -> None:
    parser = NomaParser()
    block_lines: list[str] = []
    collecting = False

    print("Ouvido ativo: aguardando blocos [NOMA_NEURAL] no stdin")

    while True:
        raw_line = sys.stdin.readline()
        if raw_line == "":
            time.sleep(0.2)
            continue

        line = raw_line.rstrip("\n")
        upper = line.upper()

        if "[NOMA_NEURAL]" in upper:
            collecting = True
            block_lines = [line]
            if "[/NOMA_NEURAL]" in upper:
                collecting = False
            else:
                continue
        elif collecting:
            block_lines.append(line)
            if "[/NOMA_NEURAL]" not in upper:
                continue
            collecting = False
        else:
            continue

        telemetry = parser.parse_telemetry("\n".join(block_lines))
        block_lines = []
        if not telemetry:
            print("Bloco recebido sem telemetria valida")
            continue

        freq_raw = telemetry.get("freq_hz", telemetry.get("frequencia_dominante"))
        amp_raw = telemetry.get("amplitude_afetiva")
        res_raw = telemetry.get("ressonancia_progenitor")

        try:
            freq_hz = float(freq_raw) if freq_raw is not None else None
            amplitude = float(amp_raw) if amp_raw is not None else None
            resonance = float(res_raw) if res_raw is not None else None
        except (TypeError, ValueError):
            print(f"Telemetria rejeitada por conversao numerica invalida: {telemetry}")
            continue

        with state_lock:
            if freq_hz is not None and freq_hz > 0.0:
                state.current_freq = freq_hz

            if amplitude is not None:
                state.current_amp = _clamp(amplitude, 0.0, 1.0)
                state.current_emotion = _emotion_from_amplitude(state.current_amp)

            if resonance is not None:
                state.current_resonance = _clamp(resonance, 0.0, 1.0)

            snapshot = {
                "freq": state.current_freq,
                "amp": state.current_amp,
                "res": state.current_resonance,
                "emotion": state.current_emotion,
            }

        if resonance is not None:
            teacher = AITeacher(near_threshold_ratio=snapshot["res"])
            with network_lock:
                aligned = teacher.align_student(
                    network=network,
                    teacher_weights=[snapshot["res"]],
                    ressonancia_progenitor=snapshot["res"],
                )
            print(f"Ressonancia aplicada via align_student: {snapshot['res']:.3f} (aligned={aligned})")

        print(
            "Telemetria recebida -> "
            f"freq={snapshot['freq']:.3f}Hz, amp={snapshot['amp']:.3f}, "
            f"emocao={snapshot['emotion']}, res={snapshot['res']:.3f}"
        )


def main() -> None:
    network = _build_or_load_brain()
    palace = Mempalace(network=network, seed=1337, pulse_step_ms=4.0)
    well = TheWellAdapter(seed=21)

    state = VitalState()
    state_lock = threading.Lock()
    network_lock = threading.Lock()

    id_to_concept = {node_id: f"node_{node_id}" for node_id in network.neurons.keys()}
    concept_to_id = {name: node_id for node_id, name in id_to_concept.items()}
    palace.register_concepts(concept_to_id=concept_to_id, id_to_concept=id_to_concept)

    listener = threading.Thread(
        target=_input_listener,
        args=(network, state, state_lock, network_lock),
        daemon=True,
        name="noma-input-listener",
    )
    listener.start()

    tick_ms = 100.0
    dream_period_steps = 20
    well_period_steps = 25
    persist_period_steps = 30
    dream_enabled = os.getenv("BRAIN_DREAM_ENABLED", "0") == "1"
    well_consolidation_enabled = os.getenv("BRAIN_WELL_CONSOLIDATION_ENABLED", "0") == "1"

    sim_t_ms = 0.0
    step = 0
    idle_since_s: float | None = None
    idle_trigger_s = 5.0

    print("Organismo Vivo iniciado: Wake -> Interact -> The Well -> Dream -> Persist")

    try:
        while True:
            with state_lock:
                current_freq = state.current_freq
                current_amp = state.current_amp
                current_emotion = state.current_emotion
                current_res = state.current_resonance

            # Wake: teacher-driven pulses keep the neural body active.
            teacher = AITeacher(
                teacher_hz=max(0.001, float(current_freq)),
                target_id=0,
                spike_weight=1.0 + (0.2 * float(current_amp)),
                near_threshold_ratio=float(current_res),
            )
            spike_train = teacher.generate_gamma_train(duration_ms=tick_ms)

            with network_lock:
                queue_size_before = _event_queue_size(network)
                now_wall_s = time.time()
                if queue_size_before == 0:
                    if idle_since_s is None:
                        idle_since_s = now_wall_s
                else:
                    idle_since_s = None

                autonomous_dream_stats: dict[str, Any] = {}
                autonomous_idle_elapsed_s: float | None = None
                if idle_since_s is not None and (now_wall_s - idle_since_s) >= idle_trigger_s:
                    autonomous_idle_elapsed_s = now_wall_s - idle_since_s
                    autonomous_dream_stats = palace.trigger_dream_cycle(
                        duration_ms=max(120.0, tick_ms * 2.0),
                        noise_energy=max(1.0, 0.95 + current_amp),
                    )
                    network.save_weights(str(MEMORY_FILE))
                    idle_since_s = now_wall_s

                input_neurons = _sorted_node_ids(network)[: max(1, min(INPUT_NEURON_COUNT, len(network.neurons)))]
                for offset_ms, _, spike_weight in spike_train:
                    event_t = sim_t_ms + float(offset_ms)
                    for target_id in input_neurons:
                        network.schedule_event(time_ms=event_t, target_id=target_id, weight=float(spike_weight))

                _run_until_empty_with_learning(network, learning_enabled=True)

                well_stats: dict[str, Any] = {}
                cycle_wisdom: list[str] = []
                if step % well_period_steps == 0:
                    concepts = well.fetch_wisdom(frequency_hz=current_freq, emotion=current_emotion)
                    cycle_wisdom = list(concepts)
                    if well_consolidation_enabled:
                        well_stats = palace.feynman_dream_consolidation(net=network, new_concepts=concepts)
                        concept_nodes = well_stats.get("concept_nodes", {})
                        if isinstance(concept_nodes, dict):
                            for concept, node_id in concept_nodes.items():
                                concept_to_id[str(concept)] = node_id
                                id_to_concept[node_id] = str(concept)
                    else:
                        well_stats = {
                            "new_concepts": len(concepts),
                            "new_nodes": 0,
                            "pruned_edges": 0,
                            "forged_geodesics": 0,
                            "active_targets": 0,
                            "mode": "wisdom_only",
                        }

                dream_stats: dict[str, Any] = {}
                if dream_enabled and step % dream_period_steps == 0:
                    dream_stats = palace.trigger_dream_cycle(
                        duration_ms=max(60.0, tick_ms * 1.8),
                        noise_energy=max(1.0, 0.9 + current_amp),
                    )
                    network.save_weights(str(MEMORY_FILE))

                if autonomous_dream_stats:
                    dream_stats = {
                        **dict(dream_stats),
                        "autonomous": True,
                        "idle_trigger_s": idle_trigger_s,
                        "idle_queue_size_before": queue_size_before,
                        **dict(autonomous_dream_stats),
                    }
                    print(
                        {
                            "step": step,
                            "dream": "autonomous",
                            "idle_seconds": round(float(autonomous_idle_elapsed_s), 3)
                            if autonomous_idle_elapsed_s is not None
                            else None,
                            "saved": str(MEMORY_FILE),
                        }
                    )

                synapses = network.get_synaptic_strengths()
                nodes = _build_nodes_payload(network, id_to_concept)

            with state_lock:
                if well_stats:
                    state.last_well_stats = well_stats
                    state.last_wisdom = cycle_wisdom
                if dream_stats:
                    state.last_dream_stats = dream_stats
                snapshot_wisdom = list(state.last_wisdom)
                snapshot_well = dict(state.last_well_stats)
                snapshot_dream = dict(state.last_dream_stats)

            payload = {
                "timestamp": time.time(),
                "learning_enabled": bool(network.learning_enabled),
                "intent_level": float(current_amp),
                "estado_emocional": str(current_emotion),
                "nodes": nodes,
                "node_names": {str(node_id): name for node_id, name in id_to_concept.items()},
                "nomes": {str(node_id): name for node_id, name in id_to_concept.items()},
                "synapses": synapses,
                "meta": {
                    "step": step,
                    "sim_t_ms": round(sim_t_ms, 3),
                    "tick_ms": tick_ms,
                    "freq_hz": float(current_freq),
                    "amplitude_afetiva": float(current_amp),
                    "ressonancia_progenitor": float(current_res),
                    "teacher_events_per_tick": len(spike_train),
                    "well_period_steps": well_period_steps,
                    "dream_period_steps": dream_period_steps,
                    "dream_enabled": dream_enabled,
                    "well_consolidation_enabled": well_consolidation_enabled,
                    "last_wisdom": snapshot_wisdom,
                    "well_stats": snapshot_well,
                    "dream_stats": snapshot_dream,
                    "source": "main_simbiosis",
                },
            }
            _atomic_write_json(DEFAULT_STATE_FILE, payload)

            if step % persist_period_steps == 0 and step > 0:
                with network_lock:
                    network.save_weights(str(MEMORY_FILE))
                print(
                    {
                        "step": step,
                        "synapses": len(synapses),
                        "wisdom": snapshot_wisdom,
                        "saved": str(MEMORY_FILE),
                    }
                )

            sim_t_ms += tick_ms
            step += 1
            time.sleep(tick_ms / 1000.0)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt recebido. Encerrando simbiose...")
    finally:
        try:
            with network_lock:
                network.save_weights(str(MEMORY_FILE))
            print(f"Memoria salva em {MEMORY_FILE}")
        except Exception as exc:
            print(f"Falha ao salvar memoria em {MEMORY_FILE}: {exc}")


if __name__ == "__main__":
    main()
