from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _normalize_text(text: str) -> str:
    return _strip_accents(text).strip().lower()


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass(frozen=True)
class ThoughtEvent:
    time_ms: float
    event_id: int
    node_id: Any
    concept: str
    input_weight: float


@dataclass(frozen=True)
class ThoughtResult:
    start_concept: str
    start_node_id: Any
    raw_sequence: str
    concept_chain: str
    natural_language: str
    confidence_score: float
    coherence_score: float
    thermodynamic_state: str
    sequence: list[dict[str, Any]]
    rejected_noise_nodes: list[str]


class ThoughtDecoder:
    """
    Layer 4: causal thought decoder.

    The decoder does not perform token prediction. It reads a causal firing path
    from the spiking network and translates the path back to semantic concepts.
    """

    WATER_CLUSTER = {
        "chuva",
        "molhado",
        "oceano",
        "rio",
        "nuvem",
        "vapor",
        "agua",
    }

    FIRE_CLUSTER = {
        "fogo",
        "calor",
        "cinzas",
        "relampago",
        "chama",
    }

    def __init__(
        self,
        concept_to_id: Mapping[str, Any] | None = None,
        id_to_concept: Mapping[Any, str] | None = None,
        max_hops: int = 20,
    ) -> None:
        self.max_hops = max_hops
        self._concept_to_id: dict[str, Any] = {}
        self._id_to_concept: dict[str, str] = {}

        if concept_to_id:
            for concept, node_id in concept_to_id.items():
                self._concept_to_id[_normalize_text(str(concept))] = node_id

        if id_to_concept:
            for node_id, concept in id_to_concept.items():
                self._id_to_concept[str(node_id)] = str(concept)

    @classmethod
    def from_node_names(cls, node_names: Mapping[str, str], max_hops: int = 20) -> "ThoughtDecoder":
        concept_to_id: dict[str, Any] = {}
        id_to_concept: dict[Any, str] = {}
        for node_id, concept in node_names.items():
            concept_to_id[str(concept)] = node_id
            id_to_concept[node_id] = str(concept)
        return cls(concept_to_id=concept_to_id, id_to_concept=id_to_concept, max_hops=max_hops)

    def register_node_names(self, node_names: Mapping[str, str]) -> None:
        for node_id, concept in node_names.items():
            self._concept_to_id[_normalize_text(str(concept))] = node_id
            self._id_to_concept[str(node_id)] = str(concept)

    def read_thought(
        self,
        network,
        start_concept: str,
        energy: float,
        start_time_ms: float = 0.0,
    ) -> ThoughtResult:
        if energy <= 0.0:
            raise ValueError("energy must be > 0")

        start_node_id = self._resolve_start_node(start_concept, network)
        fired_trace = self._execute_trace(
            network=network,
            start_node_id=start_node_id,
            energy=energy,
            start_time_ms=start_time_ms,
        )

        synapses = self._safe_get_synapses(network)
        adjacency = self._build_adjacency(synapses)
        events = self._materialize_events(fired_trace)

        chain_events = self._extract_causal_chain(events=events, start_node_id=start_node_id, adjacency=adjacency)
        rejected_noise_nodes = self._find_noise_nodes(events=events, chain=chain_events, adjacency=adjacency)

        sequence = [
            {
                "time_ms": round(event.time_ms, 6),
                "node_id": event.node_id,
                "concept": event.concept,
                "input_weight": round(event.input_weight, 6),
            }
            for event in chain_events
        ]

        ids = [str(event.node_id) for event in chain_events]
        concepts = [event.concept for event in chain_events]

        raw_sequence = " -> ".join(ids)
        concept_chain = " -> ".join(concepts)
        natural_language = self._render_natural_language(concepts)

        coherence_score = self._compute_coherence(chain_events, adjacency)
        confidence_score = self._compute_confidence(chain_events, events, coherence_score)
        thermodynamic_state = self._classify_thermodynamic_state(chain_events, coherence_score)

        return ThoughtResult(
            start_concept=start_concept,
            start_node_id=start_node_id,
            raw_sequence=raw_sequence,
            concept_chain=concept_chain,
            natural_language=natural_language,
            confidence_score=confidence_score,
            coherence_score=coherence_score,
            thermodynamic_state=thermodynamic_state,
            sequence=sequence,
            rejected_noise_nodes=rejected_noise_nodes,
        )

    def _resolve_start_node(self, start_concept: str, network) -> Any:
        concept_key = _normalize_text(start_concept)
        if concept_key in self._concept_to_id:
            return self._concept_to_id[concept_key]

        # Fallback 1: if the concept itself is a node id token.
        if start_concept in getattr(network, "neurons", {}):
            return start_concept

        # Fallback 2: check normalized id as used by gravity engine.
        if concept_key in getattr(network, "neurons", {}):
            return concept_key

        raise KeyError(f"Conceito inicial nao encontrado: {start_concept}")

    def _execute_trace(self, network, start_node_id: Any, energy: float, start_time_ms: float) -> list[dict[str, Any]]:
        network.schedule_event(float(start_time_ms), start_node_id, float(energy))

        if hasattr(network, "run_and_trace"):
            raw = list(network.run_and_trace())
            return [dict(item) for item in raw]

        # Compatibility fallback for environments not yet rebuilt with run_and_trace.
        prev_fire = {}
        for node_id, neuron in getattr(network, "neurons", {}).items():
            prev_fire[node_id] = float(getattr(neuron, "last_fire_t", float("-inf")))

        network.run_until_empty()

        out: list[dict[str, Any]] = []
        event_id = 0
        for node_id, neuron in getattr(network, "neurons", {}).items():
            fire_t = float(getattr(neuron, "last_fire_t", float("-inf")))
            if fire_t > prev_fire.get(node_id, float("-inf")):
                out.append(
                    {
                        "time_ms": fire_t,
                        "event_id": event_id,
                        "node_id": node_id,
                        "input_weight": 0.0,
                    }
                )
                event_id += 1
        out.sort(key=lambda item: (float(item.get("time_ms", 0.0)), int(item.get("event_id", 0))))
        return out

    def _safe_get_synapses(self, network) -> list[dict[str, Any]]:
        if hasattr(network, "get_synaptic_strengths"):
            raw = network.get_synaptic_strengths()
            return [dict(item) for item in raw]

        out = []
        synapses = getattr(network, "synapses", {})
        for pre_id, edges in synapses.items():
            for post_id, weight, delay_ms in edges:
                out.append(
                    {
                        "pre_id": pre_id,
                        "post_id": post_id,
                        "weight": float(weight),
                        "delay_ms": float(delay_ms),
                    }
                )
        return out

    def _build_adjacency(self, synapses: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        adjacency: dict[str, list[dict[str, Any]]] = {}
        for edge in synapses:
            pre_key = str(edge.get("pre_id"))
            adjacency.setdefault(pre_key, []).append(
                {
                    "post_id": edge.get("post_id"),
                    "post_key": str(edge.get("post_id")),
                    "weight": float(edge.get("weight", 0.0)),
                    "delay_ms": float(edge.get("delay_ms", 0.0)),
                }
            )
        return adjacency

    def _materialize_events(self, fired_trace: Iterable[dict[str, Any]]) -> list[ThoughtEvent]:
        events: list[ThoughtEvent] = []
        for row in fired_trace:
            node_id = row.get("node_id")
            concept = self._id_to_concept.get(str(node_id), str(node_id))
            events.append(
                ThoughtEvent(
                    time_ms=float(row.get("time_ms", 0.0)),
                    event_id=int(row.get("event_id", 0)),
                    node_id=node_id,
                    concept=concept,
                    input_weight=float(row.get("input_weight", 0.0)),
                )
            )

        events.sort(key=lambda event: (event.time_ms, event.event_id))
        return events

    def _extract_causal_chain(
        self,
        events: list[ThoughtEvent],
        start_node_id: Any,
        adjacency: dict[str, list[dict[str, Any]]],
    ) -> list[ThoughtEvent]:
        if not events:
            synthetic = ThoughtEvent(
                time_ms=0.0,
                event_id=0,
                node_id=start_node_id,
                concept=self._id_to_concept.get(str(start_node_id), str(start_node_id)),
                input_weight=0.0,
            )
            return [synthetic]

        start_key = str(start_node_id)
        start_idx = None
        for idx, event in enumerate(events):
            if str(event.node_id) == start_key:
                start_idx = idx
                break

        if start_idx is None:
            synthetic = ThoughtEvent(
                time_ms=0.0,
                event_id=-1,
                node_id=start_node_id,
                concept=self._id_to_concept.get(str(start_node_id), str(start_node_id)),
                input_weight=0.0,
            )
            chain = [synthetic]
            current_idx = -1
            current_event = synthetic
        else:
            chain = [events[start_idx]]
            current_idx = start_idx
            current_event = events[start_idx]

        used_ids = {(current_event.time_ms, current_event.event_id, str(current_event.node_id))}

        for _ in range(max(0, self.max_hops - 1)):
            outgoing = adjacency.get(str(current_event.node_id), [])
            if not outgoing:
                break

            best_idx = None
            best_cost = math.inf
            for idx in range(current_idx + 1, len(events)):
                candidate = events[idx]
                sig = (candidate.time_ms, candidate.event_id, str(candidate.node_id))
                if sig in used_ids:
                    continue

                edge = self._match_edge(outgoing, candidate.node_id)
                if edge is None:
                    continue

                min_time = current_event.time_ms + float(edge["delay_ms"])
                if candidate.time_ms + 1e-9 < min_time:
                    continue

                cost = self._edge_action_cost(edge, current_event, candidate)
                if cost < best_cost:
                    best_cost = cost
                    best_idx = idx

            if best_idx is None:
                break

            next_event = events[best_idx]
            chain.append(next_event)
            current_idx = best_idx
            current_event = next_event
            used_ids.add((next_event.time_ms, next_event.event_id, str(next_event.node_id)))

        if len(chain) == 1:
            projected = self._project_topological_chain(seed_event=chain[0], adjacency=adjacency)
            if len(projected) > 1:
                return projected

            # Fallback: if no valid edge projection is available, return a temporal projection.
            for event in events:
                if str(event.node_id) == str(chain[0].node_id):
                    continue
                chain.append(event)
                if len(chain) >= min(self.max_hops, 3):
                    break

        return chain

    def _project_topological_chain(
        self,
        seed_event: ThoughtEvent,
        adjacency: dict[str, list[dict[str, Any]]],
    ) -> list[ThoughtEvent]:
        chain = [seed_event]
        visited = {str(seed_event.node_id)}
        current = seed_event
        synthetic_event_id = max(1, seed_event.event_id + 1)

        for _ in range(max(0, self.max_hops - 1)):
            outgoing = adjacency.get(str(current.node_id), [])
            if not outgoing:
                break

            candidates = [edge for edge in outgoing if str(edge["post_id"]) not in visited and float(edge["weight"]) > 0.0]
            if not candidates:
                break

            candidates.sort(key=lambda edge: self._edge_action_cost(edge=edge, src=current, dst=ThoughtEvent(
                time_ms=current.time_ms + float(edge["delay_ms"]),
                event_id=synthetic_event_id,
                node_id=edge["post_id"],
                concept=self._id_to_concept.get(str(edge["post_id"]), str(edge["post_id"])),
                input_weight=float(edge["weight"]),
            )))
            edge = candidates[0]
            next_node = edge["post_id"]
            next_event = ThoughtEvent(
                time_ms=current.time_ms + float(edge["delay_ms"]),
                event_id=synthetic_event_id,
                node_id=next_node,
                concept=self._id_to_concept.get(str(next_node), str(next_node)),
                input_weight=float(edge["weight"]),
            )

            chain.append(next_event)
            visited.add(str(next_node))
            current = next_event
            synthetic_event_id += 1

        return chain

    @staticmethod
    def _match_edge(outgoing: list[dict[str, Any]], post_id: Any) -> dict[str, Any] | None:
        post_key = str(post_id)
        for edge in outgoing:
            if edge["post_key"] == post_key:
                return edge
        return None

    @staticmethod
    def _edge_action_cost(edge: dict[str, Any], src: ThoughtEvent, dst: ThoughtEvent) -> float:
        delay = max(0.0, float(edge.get("delay_ms", 0.0)))
        weight = max(1e-6, float(edge.get("weight", 0.0)))
        resistance = 1.0 / weight
        transit_dt = max(0.0, dst.time_ms - src.time_ms)
        return 0.55 * delay + 0.25 * resistance + 0.20 * transit_dt

    def _find_noise_nodes(
        self,
        events: list[ThoughtEvent],
        chain: list[ThoughtEvent],
        adjacency: dict[str, list[dict[str, Any]]],
    ) -> list[str]:
        chain_keys = {str(event.node_id) for event in chain}
        noise: set[str] = set()

        for idx, event in enumerate(events):
            key = str(event.node_id)
            if key in chain_keys:
                continue

            outgoing = adjacency.get(key, [])
            has_descendant = False
            for j in range(idx + 1, len(events)):
                candidate = events[j]
                edge = self._match_edge(outgoing, candidate.node_id)
                if edge is None:
                    continue
                if candidate.time_ms + 1e-9 >= event.time_ms + float(edge["delay_ms"]):
                    has_descendant = True
                    break

            if not has_descendant:
                noise.add(self._id_to_concept.get(key, key))

        return sorted(noise)

    def _compute_coherence(self, chain: list[ThoughtEvent], adjacency: dict[str, list[dict[str, Any]]]) -> float:
        if len(chain) <= 1:
            return 0.0

        costs: list[float] = []
        monotonic = 0
        for idx in range(len(chain) - 1):
            src = chain[idx]
            dst = chain[idx + 1]
            if dst.time_ms >= src.time_ms:
                monotonic += 1

            edge = self._match_edge(adjacency.get(str(src.node_id), []), dst.node_id)
            if edge is None:
                costs.append(8.0)
            else:
                costs.append(self._edge_action_cost(edge=edge, src=src, dst=dst))

        avg_cost = sum(costs) / len(costs)
        normalized_cost = _clamp(avg_cost / 8.0, 0.0, 1.0)
        temporal_consistency = monotonic / len(costs)

        coherence = (1.0 - normalized_cost) * 0.62 + temporal_consistency * 0.38
        return _clamp(coherence, 0.0, 1.0)

    @staticmethod
    def _compute_confidence(chain: list[ThoughtEvent], all_events: list[ThoughtEvent], coherence_score: float) -> float:
        if not all_events:
            return 0.0

        unique_chain_nodes = len({str(event.node_id) for event in chain})
        unique_all_nodes = max(1, len({str(event.node_id) for event in all_events}))
        continuity = unique_chain_nodes / unique_all_nodes

        confidence = coherence_score * 0.72 + continuity * 0.28
        return _clamp(confidence, 0.0, 1.0)

    @staticmethod
    def _classify_thermodynamic_state(chain: list[ThoughtEvent], coherence_score: float) -> str:
        if len(chain) <= 1:
            return "critical"
        if coherence_score >= 0.75:
            return "coherent"
        if coherence_score >= 0.45:
            return "critical"
        return "chaotic"

    def _render_natural_language(self, concepts: list[str]) -> str:
        if not concepts:
            return ""
        if len(concepts) == 1:
            return concepts[0]

        norm = [_normalize_text(concept) for concept in concepts]
        has_water = any(token in self.WATER_CLUSTER for token in norm)
        has_fire = any(token in self.FIRE_CLUSTER for token in norm)

        if has_water and len(concepts) >= 3:
            return f"{concepts[0]} causa {concepts[1]}, que flui para {concepts[2]}."
        if has_fire and len(concepts) >= 3:
            return f"{concepts[0]} aciona {concepts[1]}, propagando-se para {concepts[2]}."
        if len(concepts) == 2:
            return f"{concepts[0]} conduz a {concepts[1]}."
        return f"{concepts[0]} inicia uma cadeia causal: {' -> '.join(concepts[1:])}."


def read_thought(
    network,
    start_concept: str,
    energy: float,
    concept_to_id: Mapping[str, Any] | None = None,
    id_to_concept: Mapping[Any, str] | None = None,
    max_hops: int = 20,
) -> ThoughtResult:
    decoder = ThoughtDecoder(
        concept_to_id=concept_to_id,
        id_to_concept=id_to_concept,
        max_hops=max_hops,
    )
    return decoder.read_thought(network=network, start_concept=start_concept, energy=energy)


__all__ = [
    "ThoughtDecoder",
    "ThoughtEvent",
    "ThoughtResult",
    "read_thought",
]
