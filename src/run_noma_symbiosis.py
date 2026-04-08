from __future__ import annotations

from dataclasses import dataclass
import json
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any

from core.lif_neuron import LIFNeuron
from core.noma_bridge import NomaParser
from core.spiking_network import SpikingNetwork
from core.subliminal_learning import AITeacher


GRID_SIZE = 8
N_NEURONS = GRID_SIZE * GRID_SIZE
DEFAULT_STATE_FILE = Path(__file__).resolve().parent / "mind_panel_state.json"
MEMORY_FILE = Path("noma_memory.bin")
INPUT_NEURON_COUNT = 8
HEARTBEAT_MS = 100.0
DEFAULT_FANOUT = 4
BASE_FREQ_HZ = 7.83
BASE_AMP = 0.5
BASE_EMOTION = "escutando_o_vazio"
BASE_RESONANCE = 0.9


@dataclass
class LiveBrainState:
    current_freq: float = BASE_FREQ_HZ
    current_amp: float = BASE_AMP
    current_emotion: str = BASE_EMOTION
    current_resonance: float = BASE_RESONANCE


def _node_xy(node_id: int) -> tuple[float, float]:
    row = node_id // GRID_SIZE
    col = node_id % GRID_SIZE
    return float(col), float(row)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload), encoding="utf-8")
    temp_path.replace(path)


def _run_until_empty_with_learning(network: SpikingNetwork, learning_enabled: bool) -> None:
    try:
        network.run_until_empty(learning_enabled=learning_enabled)
        return
    except TypeError:
        pass

    network.learning_enabled = learning_enabled
    network.run_until_empty()


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _emotion_from_amplitude(amplitude: float) -> str:
    if amplitude >= 0.85:
        return "euforia_sincronica"
    if amplitude >= 0.65:
        return "foco_elevado"
    if amplitude >= 0.40:
        return "atencao_estavel"
    if amplitude >= 0.20:
        return "calma"
    return "repouso"


def _connect_random_topology(network: SpikingNetwork, fanout: int, seed: int) -> None:
    rng = random.Random(seed)
    fanout = max(1, min(fanout, N_NEURONS - 1))

    for pre_id in range(N_NEURONS):
        candidates = [idx for idx in range(N_NEURONS) if idx != pre_id]
        targets = rng.sample(candidates, k=fanout)
        for post_id in targets:
            network.add_connection(pre_id=pre_id, post_id=post_id, weight=0.1, delay_ms=2.0)


def setup_brain(seed: int = 17, fanout: int = DEFAULT_FANOUT) -> SpikingNetwork:
    network = SpikingNetwork(learning_enabled=True)

    for neuron_id in range(N_NEURONS):
        network.add_neuron(
            node_id=neuron_id,
            neuron_instance=LIFNeuron(v_thresh=1.0, tau=20.0, refractory_period=5.0),
        )

    if MEMORY_FILE.exists():
        try:
            network.load_weights(str(MEMORY_FILE))
            print(f"Memoria sinaptica carregada de {MEMORY_FILE}")
            return network
        except Exception as exc:
            print(f"Falha ao carregar memoria ({MEMORY_FILE}): {exc}")
            print("Reinicializando topologia aleatoria com peso base 0.1")

    _connect_random_topology(network=network, fanout=fanout, seed=seed)
    print("Topologia inicial aleatoria criada (peso=0.1, delay=2.0ms)")

    return network


def _input_listener(
    network: SpikingNetwork,
    state: LiveBrainState,
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
            # Mantem a thread viva mesmo se stdin for fechado/pipe for encerrado.
            time.sleep(0.25)
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

        block_text = "\n".join(block_lines)
        telemetry = parser.parse_telemetry(block_text)
        if not telemetry:
            print("Bloco recebido sem telemetria valida")
            block_lines = []
            continue

        freq_raw = telemetry.get("freq_hz", telemetry.get("frequencia_dominante"))
        amp_raw = telemetry.get("amplitude_afetiva")
        res_raw = telemetry.get("ressonancia_progenitor")

        try:
            if freq_raw is not None:
                freq_hz = float(freq_raw)
            else:
                freq_hz = None

            if amp_raw is not None:
                amplitude = float(amp_raw)
            else:
                amplitude = None

            if res_raw is not None:
                resonance = float(res_raw)
            else:
                resonance = None
        except (TypeError, ValueError):
            print(f"Telemetria rejeitada por conversao numerica invalida: {telemetry}")
            block_lines = []
            continue

        with state_lock:
            if freq_hz is not None and freq_hz > 0.0:
                state.current_freq = freq_hz

            if amplitude is not None:
                state.current_amp = _clamp(amplitude, 0.0, 1.0)
                state.current_emotion = _emotion_from_amplitude(state.current_amp)

            if resonance is not None:
                state.current_resonance = _clamp(resonance, 0.0, 1.0)

            snapshot_freq = state.current_freq
            snapshot_amp = state.current_amp
            snapshot_res = state.current_resonance
            snapshot_emotion = state.current_emotion

        if resonance is not None:
            teacher = AITeacher(near_threshold_ratio=snapshot_res)
            with network_lock:
                aligned = teacher.align_student(
                    network=network,
                    teacher_weights=[snapshot_res],
                    ressonancia_progenitor=snapshot_res,
                )
            print(f"Ressonancia aplicada via align_student: {snapshot_res:.3f} (aligned={aligned})")

        print(
            "Telemetria recebida -> "
            f"freq={snapshot_freq:.3f}Hz, amp={snapshot_amp:.3f}, "
            f"emocao={snapshot_emotion}, res={snapshot_res:.3f}"
        )
        block_lines = []


def main() -> None:
    network = setup_brain()
    state = LiveBrainState()
    state_lock = threading.Lock()
    network_lock = threading.Lock()

    nodes = [{"id": node_id, "x": _node_xy(node_id)[0], "y": _node_xy(node_id)[1]} for node_id in range(N_NEURONS)]
    input_neurons = list(range(INPUT_NEURON_COUNT))

    listener = threading.Thread(
        target=_input_listener,
        args=(network, state, state_lock, network_lock),
        daemon=True,
        name="noma-input-listener",
    )
    listener.start()

    sim_t_ms = 0.0
    step = 0
    tick_ms = HEARTBEAT_MS

    print("Noma Symbiosis: Cerebro Vivo iniciado")
    print("Estado base -> freq=7.83Hz, amp=0.5, emocao=escutando_o_vazio")

    try:
        while True:
            with state_lock:
                current_freq = state.current_freq
                current_amp = state.current_amp
                current_emotion = state.current_emotion
                current_res = state.current_resonance

            teacher = AITeacher(
                teacher_hz=max(0.001, float(current_freq)),
                target_id=0,
                spike_weight=1.05,
                near_threshold_ratio=current_res,
            )
            spike_train = teacher.generate_gamma_train(duration_ms=tick_ms)

            with network_lock:
                for offset_ms, _, spike_weight in spike_train:
                    event_t = sim_t_ms + float(offset_ms)
                    for target_id in input_neurons:
                        network.schedule_event(
                            time_ms=event_t,
                            target_id=target_id,
                            weight=float(spike_weight),
                        )

                _run_until_empty_with_learning(network, learning_enabled=True)
                synapses = network.get_synaptic_strengths()

            payload = {
                "timestamp": time.time(),
                "learning_enabled": bool(network.learning_enabled),
                "intent_level": float(current_amp),
                "estado_emocional": current_emotion,
                "nodes": nodes,
                "synapses": synapses,
                "meta": {
                    "step": step,
                    "sim_t_ms": round(sim_t_ms, 3),
                    "freq_hz": float(current_freq),
                    "amplitude_afetiva": float(current_amp),
                    "ressonancia_progenitor": float(current_res),
                    "teacher_events_per_tick": len(spike_train) * len(input_neurons),
                    "tick_ms": tick_ms,
                },
            }
            _atomic_write_json(DEFAULT_STATE_FILE, payload)

            sim_t_ms += tick_ms
            step += 1
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt recebido. Salvando memoria sinaptica...")
    finally:
        try:
            with network_lock:
                network.save_weights(str(MEMORY_FILE))
            print(f"Memoria salva em {MEMORY_FILE}")
        except Exception as exc:
            print(f"Falha ao salvar memoria em {MEMORY_FILE}: {exc}")


if __name__ == "__main__":
    main()
