from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import random
import select
import sys
import time
from pathlib import Path
from typing import Any

from adapters.the_well_adapter import TheWellAdapter
from core.lif_neuron import LIFNeuron
from core.mempalace import Mempalace
from core.noma_bridge import NomaParser
from core.spiking_network import SpikingNetwork
from core.subliminal_learning import AITeacher


DEFAULT_STATE_FILE = Path(__file__).resolve().parent.parent / "mind_panel_state.json"
DEFAULT_MEMORY_FILE = Path("noma_memory.bin")

GRID_SIZE = 8
N_NEURONS = GRID_SIZE * GRID_SIZE
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
    last_thought_trace: dict[str, Any] = field(default_factory=dict)
    thought_input: str | None = None


class NomaCore:
    """Single Python facade for the native neuromorphic runtime."""

    def __init__(
        self,
        *,
        state_file: Path = DEFAULT_STATE_FILE,
        memory_file: Path = DEFAULT_MEMORY_FILE,
        tick_ms: float = 100.0,
        seed: int = 17,
        n_neurons: int = N_NEURONS,
    ) -> None:
        if tick_ms <= 0.0:
            raise ValueError("tick_ms must be > 0")

        self.state_file = Path(state_file)
        self.memory_file = Path(memory_file)
        self.tick_ms = float(tick_ms)
        self.seed = int(seed)
        self.n_neurons = int(n_neurons)

        self.dream_period_steps = 20
        self.well_period_steps = 25
        self.persist_period_steps = 30
        self.idle_trigger_s = 5.0

        self.dream_enabled = os.getenv("BRAIN_DREAM_ENABLED", "0") == "1"
        self.well_consolidation_enabled = os.getenv("BRAIN_WELL_CONSOLIDATION_ENABLED", "0") == "1"

        self.network = self._build_or_load_brain(seed=self.seed, n_neurons=self.n_neurons)
        self.parser = NomaParser()
        self.palace = Mempalace(network=self.network, seed=1337, pulse_step_ms=4.0)
        self.well = TheWellAdapter(seed=21)
        self.state = VitalState()

        self.id_to_concept = {node_id: f"node_{node_id}" for node_id in self.network.neurons.keys()}
        self.concept_to_id = {name: node_id for node_id, name in self.id_to_concept.items()}
        self.palace.register_concepts(concept_to_id=self.concept_to_id, id_to_concept=self.id_to_concept)

        self.sim_t_ms = 0.0
        self.step_count = 0
        self.idle_since_s: float | None = None

        self._collecting_noma_block = False
        self._noma_block_lines: list[str] = []

        self._teacher_cache_key: tuple[float, float, float] | None = None
        self._teacher = AITeacher(
            teacher_hz=BASE_FREQ_HZ,
            target_id=0,
            spike_weight=1.0 + (0.2 * BASE_AMP),
            near_threshold_ratio=BASE_RESONANCE,
        )

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload), encoding="utf-8")
        temp_path.replace(path)

    @staticmethod
    def _safe_read_state(path: Path) -> dict[str, Any] | None:
        try:
            if not path.exists():
                return None
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return None
            return payload
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _run_until_empty_with_learning(network: SpikingNetwork, learning_enabled: bool) -> None:
        try:
            network.run_until_empty(learning_enabled=learning_enabled)
            return
        except TypeError:
            pass

        network.learning_enabled = learning_enabled
        network.run_until_empty()

    @staticmethod
    def _event_queue_size(network: SpikingNetwork) -> int:
        try:
            return int(len(network.event_queue))
        except Exception:
            return 0

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    @staticmethod
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

    @staticmethod
    def _connect_random_topology(network: SpikingNetwork, n_neurons: int, fanout: int, seed: int) -> None:
        rng = random.Random(seed)
        fanout = max(1, min(fanout, max(1, n_neurons - 1)))

        for pre_id in range(n_neurons):
            candidates = [idx for idx in range(n_neurons) if idx != pre_id]
            targets = rng.sample(candidates, k=fanout)
            for post_id in targets:
                network.add_connection(pre_id=pre_id, post_id=post_id, weight=0.1, delay_ms=2.0)

    @staticmethod
    def _sorted_node_ids(network: SpikingNetwork) -> list[Any]:
        numeric = sorted([node_id for node_id in network.neurons.keys() if isinstance(node_id, int)])
        others = sorted(
            [node_id for node_id in network.neurons.keys() if not isinstance(node_id, int)],
            key=lambda item: str(item),
        )
        return numeric + others

    @staticmethod
    def _node_xy(position: int, total: int) -> tuple[float, float]:
        cols = max(1, int(round(total ** 0.5)))
        row = position // cols
        col = position % cols
        return float(col), float(row)

    @classmethod
    def _build_nodes_payload(cls, network: SpikingNetwork, id_to_concept: dict[Any, str]) -> list[dict[str, Any]]:
        node_ids = cls._sorted_node_ids(network)
        total = max(1, len(node_ids))

        nodes: list[dict[str, Any]] = []
        for idx, node_id in enumerate(node_ids):
            x, y = cls._node_xy(position=idx, total=total)
            nodes.append(
                {
                    "id": node_id,
                    "name": id_to_concept.get(node_id, f"node_{node_id}"),
                    "x": x,
                    "y": y,
                }
            )
        return nodes

    def _build_or_load_brain(self, *, seed: int, n_neurons: int) -> SpikingNetwork:
        network = SpikingNetwork(learning_enabled=True)

        if self.memory_file.exists():
            try:
                network.load_weights(str(self.memory_file))
                print(f"Memoria sinaptica carregada de {self.memory_file}")
            except Exception as exc:
                print(f"Falha ao carregar memoria ({self.memory_file}): {exc}")

        if not network.neurons:
            for neuron_id in range(n_neurons):
                network.add_neuron(
                    node_id=neuron_id,
                    neuron_instance=LIFNeuron(v_thresh=1.0, tau=20.0, refractory_period=5.0),
                )
            self._connect_random_topology(network=network, n_neurons=n_neurons, fanout=DEFAULT_FANOUT, seed=seed)
            print("Topologia inicial aleatoria criada (peso=0.1, delay=2.0ms)")

        return network

    def _ensure_teacher(self, *, freq_hz: float, amplitude: float, resonance: float) -> AITeacher:
        key = (round(freq_hz, 6), round(amplitude, 6), round(resonance, 6))
        if key != self._teacher_cache_key:
            self._teacher = AITeacher(
                teacher_hz=max(0.001, float(freq_hz)),
                target_id=0,
                spike_weight=1.0 + (0.2 * float(amplitude)),
                near_threshold_ratio=float(resonance),
            )
            self._teacher_cache_key = key
        return self._teacher

    def _poll_noma_input_nonblocking(self) -> None:
        while True:
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            except (OSError, ValueError):
                return

            if not ready:
                return

            raw_line = sys.stdin.readline()
            if raw_line == "":
                return

            self._consume_noma_line(raw_line.rstrip("\n"))

    def _consume_noma_line(self, line: str) -> None:
        upper = line.upper()

        if "[NOMA_NEURAL]" in upper:
            self._collecting_noma_block = True
            self._noma_block_lines = [line]
            if "[/NOMA_NEURAL]" in upper:
                self._collecting_noma_block = False
            else:
                return
        elif self._collecting_noma_block:
            self._noma_block_lines.append(line)
            if "[/NOMA_NEURAL]" not in upper:
                return
            self._collecting_noma_block = False
        else:
            return

        telemetry = self.parser.parse_telemetry("\n".join(self._noma_block_lines))
        self._noma_block_lines = []
        if not telemetry:
            print("Bloco recebido sem telemetria valida")
            return

        freq_raw = telemetry.get("freq_hz", telemetry.get("frequencia_dominante"))
        amp_raw = telemetry.get("amplitude_afetiva")
        res_raw = telemetry.get("ressonancia_progenitor")

        try:
            freq_hz = float(freq_raw) if freq_raw is not None else None
            amplitude = float(amp_raw) if amp_raw is not None else None
            resonance = float(res_raw) if res_raw is not None else None
        except (TypeError, ValueError):
            print(f"Telemetria rejeitada por conversao numerica invalida: {telemetry}")
            return

        if freq_hz is not None and freq_hz > 0.0:
            self.state.current_freq = freq_hz

        if amplitude is not None:
            self.state.current_amp = self._clamp(amplitude, 0.0, 1.0)
            self.state.current_emotion = self._emotion_from_amplitude(self.state.current_amp)

        if resonance is not None:
            self.state.current_resonance = self._clamp(resonance, 0.0, 1.0)

        snapshot = {
            "freq": self.state.current_freq,
            "amp": self.state.current_amp,
            "res": self.state.current_resonance,
            "emotion": self.state.current_emotion,
        }

        if resonance is not None:
            align_teacher = AITeacher(near_threshold_ratio=snapshot["res"])
            aligned = align_teacher.align_student(
                network=self.network,
                teacher_weights=[snapshot["res"]],
                ressonancia_progenitor=snapshot["res"],
            )
            print(f"Ressonancia aplicada via align_student: {snapshot['res']:.3f} (aligned={aligned})")

        print(
            "Telemetria recebida -> "
            f"freq={snapshot['freq']:.3f}Hz, amp={snapshot['amp']:.3f}, "
            f"emocao={snapshot['emotion']}, res={snapshot['res']:.3f}"
        )

    def _ingest_bridge_state(self, bridge_payload: dict[str, Any]) -> None:
        bridge_meta = bridge_payload.get("meta", {}) if isinstance(bridge_payload, dict) else {}
        bridge_thought_input = bridge_meta.get("thought_input") if isinstance(bridge_meta, dict) else None
        bridge_thought_trace = bridge_payload.get("thought_trace") if isinstance(bridge_payload, dict) else None

        if isinstance(bridge_thought_input, str) and bridge_thought_input.strip():
            self.state.thought_input = bridge_thought_input.strip()
        if isinstance(bridge_thought_trace, dict) and bridge_thought_trace:
            self.state.last_thought_trace = dict(bridge_thought_trace)

    def boot_log(self) -> None:
        print("Ouvido ativo: aguardando blocos [NOMA_NEURAL] no stdin")
        print("MNHI 4.0 iniciado: Kernel C++ + Loop Vital unificado")

    def step(self) -> dict[str, Any]:
        self._poll_noma_input_nonblocking()

        bridge_payload = self._safe_read_state(self.state_file) or {}
        self._ingest_bridge_state(bridge_payload)

        current_freq = self.state.current_freq
        current_amp = self.state.current_amp
        current_emotion = self.state.current_emotion
        current_res = self.state.current_resonance

        teacher = self._ensure_teacher(freq_hz=current_freq, amplitude=current_amp, resonance=current_res)
        spike_train = teacher.generate_gamma_train(duration_ms=self.tick_ms)

        queue_size_before = self._event_queue_size(self.network)
        now_wall_s = time.time()
        if queue_size_before == 0:
            if self.idle_since_s is None:
                self.idle_since_s = now_wall_s
        else:
            self.idle_since_s = None

        autonomous_dream_stats: dict[str, Any] = {}
        autonomous_idle_elapsed_s: float | None = None
        if self.idle_since_s is not None and (now_wall_s - self.idle_since_s) >= self.idle_trigger_s:
            autonomous_idle_elapsed_s = now_wall_s - self.idle_since_s
            autonomous_dream_stats = self.palace.trigger_dream_cycle(
                duration_ms=max(120.0, self.tick_ms * 2.0),
                noise_energy=max(1.0, 0.95 + current_amp),
            )
            self.network.save_weights(str(self.memory_file))
            self.idle_since_s = now_wall_s

        input_neurons = self._sorted_node_ids(self.network)[: max(1, min(INPUT_NEURON_COUNT, len(self.network.neurons)))]
        for offset_ms, _, spike_weight in spike_train:
            event_t = self.sim_t_ms + float(offset_ms)
            for target_id in input_neurons:
                self.network.schedule_event(time_ms=event_t, target_id=target_id, weight=float(spike_weight))

        self._run_until_empty_with_learning(self.network, learning_enabled=True)

        well_stats: dict[str, Any] = {}
        cycle_wisdom: list[str] = []
        if self.step_count % self.well_period_steps == 0:
            concepts = self.well.fetch_wisdom(frequency_hz=current_freq, emotion=current_emotion)
            cycle_wisdom = list(concepts)
            if self.well_consolidation_enabled:
                well_stats = self.palace.feynman_dream_consolidation(net=self.network, new_concepts=concepts)
                concept_nodes = well_stats.get("concept_nodes", {})
                if isinstance(concept_nodes, dict):
                    for concept, node_id in concept_nodes.items():
                        self.concept_to_id[str(concept)] = node_id
                        self.id_to_concept[node_id] = str(concept)
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
        if self.dream_enabled and self.step_count % self.dream_period_steps == 0:
            dream_stats = self.palace.trigger_dream_cycle(
                duration_ms=max(60.0, self.tick_ms * 1.8),
                noise_energy=max(1.0, 0.9 + current_amp),
            )
            self.network.save_weights(str(self.memory_file))

        if autonomous_dream_stats:
            dream_stats = {
                **dict(dream_stats),
                "autonomous": True,
                "idle_trigger_s": self.idle_trigger_s,
                "idle_queue_size_before": queue_size_before,
                **dict(autonomous_dream_stats),
            }
            print(
                {
                    "step": self.step_count,
                    "dream": "autonomous",
                    "idle_seconds": round(float(autonomous_idle_elapsed_s), 3)
                    if autonomous_idle_elapsed_s is not None
                    else None,
                    "saved": str(self.memory_file),
                }
            )

        synapses = self.network.get_synaptic_strengths()
        nodes = self._build_nodes_payload(self.network, self.id_to_concept)

        if well_stats:
            self.state.last_well_stats = well_stats
            self.state.last_wisdom = cycle_wisdom
        if dream_stats:
            self.state.last_dream_stats = dream_stats

        if self.state.last_thought_trace:
            enriched_thought_trace = dict(self.state.last_thought_trace)
            enriched_thought_trace["well_context"] = list(self.state.last_wisdom)
            enriched_thought_trace["cycle_step"] = int(self.step_count)
            self.state.last_thought_trace = enriched_thought_trace

        snapshot_wisdom = list(self.state.last_wisdom)
        snapshot_well = dict(self.state.last_well_stats)
        snapshot_dream = dict(self.state.last_dream_stats)
        snapshot_thought = dict(self.state.last_thought_trace)
        snapshot_thought_input = self.state.thought_input

        payload: dict[str, Any] = {
            "timestamp": time.time(),
            "learning_enabled": bool(self.network.learning_enabled),
            "intent_level": float(current_amp),
            "estado_emocional": str(current_emotion),
            "nodes": nodes,
            "node_names": {str(node_id): name for node_id, name in self.id_to_concept.items()},
            "nomes": {str(node_id): name for node_id, name in self.id_to_concept.items()},
            "synapses": synapses,
            "meta": {
                "step": self.step_count,
                "sim_t_ms": round(self.sim_t_ms, 3),
                "tick_ms": self.tick_ms,
                "freq_hz": float(current_freq),
                "amplitude_afetiva": float(current_amp),
                "ressonancia_progenitor": float(current_res),
                "teacher_events_per_tick": len(spike_train),
                "well_period_steps": self.well_period_steps,
                "dream_period_steps": self.dream_period_steps,
                "dream_enabled": self.dream_enabled,
                "well_consolidation_enabled": self.well_consolidation_enabled,
                "last_wisdom": snapshot_wisdom,
                "well_stats": snapshot_well,
                "dream_stats": snapshot_dream,
                "thought_input": snapshot_thought_input,
                "source": "main.py",
            },
        }
        if snapshot_thought:
            payload["thought_trace"] = snapshot_thought

        self._atomic_write_json(self.state_file, payload)

        if self.step_count % self.persist_period_steps == 0 and self.step_count > 0:
            self.network.save_weights(str(self.memory_file))
            print(
                {
                    "step": self.step_count,
                    "synapses": len(synapses),
                    "wisdom": snapshot_wisdom,
                    "saved": str(self.memory_file),
                }
            )

        self.sim_t_ms += self.tick_ms
        self.step_count += 1
        return payload

    def shutdown(self) -> None:
        self.network.save_weights(str(self.memory_file))


__all__ = ["NomaCore", "VitalState"]
