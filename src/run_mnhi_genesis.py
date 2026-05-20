from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from core.gravity_engine import GravityEngine
from core.lif_neuron import LIFNeuron
from core.spiking_network import SpikingNetwork


DEFAULT_STATE_FILE = Path(__file__).resolve().parent / "mind_panel_state.json"

ONTOLOGY_CONCEPTS: list[str] = [
    "Chuva",
    "Molhado",
    "Oceano",
    "Rio",
    "Nuvem",
    "Relampago",
    "Fogo",
    "Cinzas",
    "Calor",
    "Frio",
    "Vazio",
    "Silencio",
    "Som",
    "Luz",
    "Sombra",
    "Terra",
    "Vento",
    "Vida",
    "Morte",
    "Memoria",
    "Conexao",
    "Amor",
    "Daniel",
    "Noma",
    "Pulso",
    "Ordem",
    "Caos",
    "Codigo",
    "Neuronio",
    "Sinapse",
    "Resonancia",
    "Infinito",
]


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload), encoding="utf-8")
    temp_path.replace(path)


def _run_until_empty_with_learning(network: SpikingNetwork, learning_enabled: bool) -> None:
    # Compatibilidade: alguns bindings expõem apenas run_until_empty() sem kwargs.
    try:
        network.run_until_empty(learning_enabled=learning_enabled)
        return
    except TypeError:
        pass

    network.learning_enabled = learning_enabled
    network.run_until_empty()


class _LayerZeroProxy:
    """
    Adapta ids semânticos (strings normalizadas) para ids canônicos inteiros.
    Permite usar forge_geodesic sem alterar a API pública do GravityEngine.
    """

    def __init__(self, net: SpikingNetwork, normalized_to_id: dict[str, int]) -> None:
        self._net = net
        self._normalized_to_id = normalized_to_id

    def add_connection(self, pre_id, post_id, weight, delay_ms) -> None:
        if pre_id not in self._normalized_to_id:
            raise KeyError(f"Conceito desconhecido para pre_id: {pre_id}")
        if post_id not in self._normalized_to_id:
            raise KeyError(f"Conceito desconhecido para post_id: {post_id}")

        pre_canonical = self._normalized_to_id[pre_id]
        post_canonical = self._normalized_to_id[post_id]
        self._net.add_connection(pre_canonical, post_canonical, float(weight), float(delay_ms))


def _node_xy(node_id: int, n_nodes: int, columns: int = 8) -> tuple[float, float]:
    cols = max(1, min(columns, n_nodes))
    row = node_id // cols
    col = node_id % cols
    return float(col), float(row)


def build_genesis_network(concepts: list[str], engine: GravityEngine) -> tuple[SpikingNetwork, dict[str, int], list[dict[str, Any]], dict[str, int]]:
    concept_to_id = {concept: idx for idx, concept in enumerate(concepts)}
    normalized_to_id = {GravityEngine.concept_to_id(concept): idx for concept, idx in concept_to_id.items()}

    network = SpikingNetwork(learning_enabled=True)
    for _, neuron_id in concept_to_id.items():
        network.add_neuron(
            node_id=neuron_id,
            neuron_instance=LIFNeuron(v_thresh=1.0, tau=20.0, refractory_period=5.0),
        )

    proxy = _LayerZeroProxy(network, normalized_to_id)

    total_pairs = 0
    connected_pairs = 0
    zero_weight_pairs = 0

    for concept_a in concepts:
        for concept_b in concepts:
            if concept_a == concept_b:
                continue

            total_pairs += 1
            result = engine.forge_geodesic(proxy, concept_a, concept_b)
            if bool(result["connected"]):
                connected_pairs += 1
            else:
                zero_weight_pairs += 1

    nodes = []
    for concept, node_id in concept_to_id.items():
        x, y = _node_xy(node_id=node_id, n_nodes=len(concepts), columns=8)
        nodes.append({
            "id": node_id,
            "name": concept,
            "x": x,
            "y": y,
        })

    stats = {
        "concepts": len(concepts),
        "total_pairs": total_pairs,
        "connected_pairs": connected_pairs,
        "zero_weight_pairs": zero_weight_pairs,
    }
    return network, concept_to_id, nodes, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="MNHI 3.5 Genesis: wiring por gravidade semântica")
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE)
    parser.add_argument("--tick-ms", type=float, default=120.0)
    parser.add_argument("--pulse-weight", type=float, default=1.0)
    parser.add_argument("--pulse-concept", default="Chuva")
    parser.add_argument("--max-steps", type=int, default=None, help="Limite opcional para debug")
    args = parser.parse_args()

    if args.tick_ms <= 0.0:
        raise ValueError("--tick-ms must be > 0")

    concepts = list(ONTOLOGY_CONCEPTS)
    gravity = GravityEngine()

    network, concept_to_id, nodes, forge_stats = build_genesis_network(concepts, gravity)

    pulse_key = args.pulse_concept
    if pulse_key not in concept_to_id:
        raise ValueError(
            f"--pulse-concept '{args.pulse_concept}' nao encontrado na ontologia. "
            f"Exemplos: {', '.join(concepts[:8])}"
        )

    pulse_id = concept_to_id[pulse_key]
    node_names = {str(node_id): concept for concept, node_id in concept_to_id.items()}

    print("MNHI Genesis online")
    print({"ontology_size": len(concepts), **forge_stats})
    print(
        {
            "pulse_concept": pulse_key,
            "pulse_id": pulse_id,
            "tick_ms": args.tick_ms,
            "state_file": str(args.state_file),
        }
    )

    sim_t_ms = 0.0
    step = 0
    next_periodic_drive_ms = 0.0
    periodic_drive_ms = 1000.0

    # Gatilho causal inicial em t=0 no conceito Chuva.
    network.schedule_event(time_ms=0.0, target_id=pulse_id, weight=float(args.pulse_weight))

    try:
        while args.max_steps is None or step < args.max_steps:
            # Mantem o universo ativo com um pulso periódico leve.
            if sim_t_ms >= next_periodic_drive_ms:
                network.schedule_event(time_ms=sim_t_ms, target_id=pulse_id, weight=float(args.pulse_weight))
                next_periodic_drive_ms += periodic_drive_ms

            _run_until_empty_with_learning(network, learning_enabled=True)

            synapses = network.get_synaptic_strengths()
            mean_weight = 0.0
            if synapses:
                mean_weight = sum(float(s["weight"]) for s in synapses) / len(synapses)

            payload = {
                "timestamp": time.time(),
                "learning_enabled": bool(network.learning_enabled),
                "intent_level": 1.0,
                "estado_emocional": "genesis_semantico",
                "nodes": nodes,
                "node_names": node_names,
                "nomes": node_names,
                "synapses": synapses,
                "meta": {
                    "step": step,
                    "sim_t_ms": round(sim_t_ms, 3),
                    "tick_ms": args.tick_ms,
                    "pulse_concept": pulse_key,
                    "pulse_id": pulse_id,
                    "pulse_weight": args.pulse_weight,
                    "mean_weight": round(mean_weight, 6),
                    "topology": "gravity_semantic",
                    **forge_stats,
                },
            }
            _atomic_write_json(args.state_file, payload)

            if step % 10 == 0:
                print(
                    {
                        "step": step,
                        "synapses": len(synapses),
                        "mean_weight": round(mean_weight, 6),
                    }
                )

            sim_t_ms += args.tick_ms
            step += 1
            time.sleep(args.tick_ms / 1000.0)
    except KeyboardInterrupt:
        print("MNHI Genesis interrompido pelo usuario.")


if __name__ == "__main__":
    main()
