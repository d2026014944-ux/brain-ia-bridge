from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any

from core.gravity_engine import GravityEngine
from core.lif_neuron import LIFNeuron
from core.subliminal_learning import AITeacher
from core.synapse_stdp import SynapseSTDPBitNet


def _to_ternary(weight: float) -> float:
    if weight > 0.5:
        return 1.0
    if weight < -0.5:
        return -1.0
    return 0.0


class Mempalace:
    """
    Layer 5: dream-cycle memory sculpting.

    During idle windows the palace injects structured subliminal spikes using the
    existing event min-heap, then consolidates edge phases into ternary BitNet
    states {-1, 0, 1} based on spontaneous activity order.
    """

    def __init__(self, network, seed: int | None = None, pulse_step_ms: float = 5.0) -> None:
        self.network = network
        self._rng = random.Random(seed)
        self.pulse_step_ms = float(max(1.0, pulse_step_ms))
        self._gravity_engine = GravityEngine()
        self._daily_firing_counts: dict[Any, int] = defaultdict(int)
        self._concept_to_id: dict[str, Any] = {}
        self._id_to_concept: dict[Any, str] = {}

    def start_dream_mode(self, net=None, intensity: float = 1.0) -> dict[str, Any]:
        target_net = net if net is not None else self.network
        if intensity <= 0.0:
            raise ValueError("intensity must be > 0")

        weak_threshold = max(0.08, min(0.40, 0.22 + (0.06 * float(intensity))))
        weak_nodes = self._select_weak_zone_nodes(target_net=target_net, threshold=weak_threshold)
        if not weak_nodes:
            weak_nodes = list(target_net.neurons.keys())

        teacher = AITeacher(
            teacher_hz=40.0,
            spike_weight=max(1.0, 0.95 + (0.18 * float(intensity))),
        )
        duration_ms = max(120.0, min(1800.0, 320.0 + (200.0 * float(intensity))))
        teacher_spikes = teacher.generate_gamma_train(duration_ms=duration_ms)

        activation_log: dict[Any, list[float]] = defaultdict(list)
        selected_neurons: set[Any] = set()
        for event_t, _, spike_weight in teacher_spikes:
            node_id = self._rng.choice(weak_nodes)
            jitter = self._rng.uniform(0.0, 0.8)
            t_ms = float(event_t) + jitter
            target_net.schedule_event(time_ms=t_ms, target_id=node_id, weight=float(spike_weight))
            activation_log[node_id].append(t_ms)
            selected_neurons.add(node_id)

        previous_learning = bool(getattr(target_net, "learning_enabled", True))
        if hasattr(target_net, "learning_enabled"):
            target_net.learning_enabled = True

        trace: list[dict[str, Any]] = []
        try:
            if hasattr(target_net, "run_and_trace"):
                trace = list(target_net.run_and_trace())
            else:
                try:
                    target_net.run_until_empty(learning_enabled=True)
                except TypeError:
                    target_net.run_until_empty()
        finally:
            if hasattr(target_net, "learning_enabled"):
                target_net.learning_enabled = previous_learning

        self._register_activity(trace=trace, activation_log=activation_log)

        passes = max(1, min(4, int(round(float(intensity) * 2.0))))
        consolidated_edges = 0
        for _ in range(passes):
            consolidated_edges += self._consolidate_ternary_weights_from_activity(activation_log)

        pruned_edges = self._hard_prune_low_weights(target_net=target_net, threshold=max(0.10, weak_threshold))

        return {
            "teacher_spikes": int(len(teacher_spikes)),
            "selected_neurons": list(selected_neurons),
            "intensity": float(intensity),
            "duration_ms": float(duration_ms),
            "consolidated_edges": int(consolidated_edges),
            "pruned_edges": int(pruned_edges),
            "weak_threshold": float(weak_threshold),
        }

    def trigger_dream_cycle(self, duration_ms: float, noise_energy: float) -> dict[str, Any]:
        if duration_ms <= 0.0:
            raise ValueError("duration_ms must be > 0")
        if noise_energy <= 0.0:
            raise ValueError("noise_energy must be > 0")

        neuron_ids = list(self.network.neurons.keys())
        if not neuron_ids:
            return {
                "scheduled_events": 0,
                "selected_neurons": [],
                "duration_ms": float(duration_ms),
                "noise_energy": float(noise_energy),
                "consolidated_edges": 0,
            }

        n_select = max(1, math.ceil(0.10 * len(neuron_ids)))
        activation_log: dict[Any, list[float]] = defaultdict(list)
        selected_union: set[Any] = set()

        t_ms = 0.0
        scheduled_events = 0
        while t_ms <= duration_ms + 1e-9:
            selected = self._select_dream_neurons(neuron_ids, n_select=n_select)
            selected_union.update(selected)

            for node_id in selected:
                jitter = self._rng.uniform(0.0, 0.7)
                event_t = t_ms + jitter
                self.network.schedule_event(time_ms=event_t, target_id=node_id, weight=float(noise_energy))
                activation_log[node_id].append(event_t)
                scheduled_events += 1

            t_ms += self.pulse_step_ms

        previous_learning = bool(getattr(self.network, "learning_enabled", True))
        if hasattr(self.network, "learning_enabled"):
            self.network.learning_enabled = True

        try:
            trace = []
            if hasattr(self.network, "run_and_trace"):
                trace = list(self.network.run_and_trace())
            else:
                try:
                    # Some implementations expose this kwarg contract.
                    self.network.run_until_empty(learning_enabled=True)
                except TypeError:
                    self.network.run_until_empty()
        finally:
            if hasattr(self.network, "learning_enabled"):
                self.network.learning_enabled = previous_learning

        self._register_activity(trace=trace, activation_log=activation_log)
        consolidated = self._consolidate_ternary_weights_from_activity(activation_log)
        pruned = self._hard_prune_low_weights(target_net=self.network, threshold=0.12)

        return {
            "scheduled_events": int(scheduled_events),
            "selected_neurons": list(selected_union),
            "duration_ms": float(duration_ms),
            "noise_energy": float(noise_energy),
            "consolidated_edges": int(consolidated),
            "pruned_edges": int(pruned),
        }

    def register_concepts(
        self,
        concept_to_id: dict[str, Any] | None = None,
        id_to_concept: dict[Any, str] | None = None,
    ) -> None:
        if concept_to_id:
            for concept, node_id in concept_to_id.items():
                key = self._gravity_engine.concept_to_id(str(concept))
                self._concept_to_id[key] = node_id
                self._id_to_concept[node_id] = str(concept)

        if id_to_concept:
            for node_id, concept in id_to_concept.items():
                key = self._gravity_engine.concept_to_id(str(concept))
                self._concept_to_id[key] = node_id
                self._id_to_concept[node_id] = str(concept)

    def feynman_dream_consolidation(self, net=None, new_concepts: list[str] | None = None) -> dict[str, Any]:
        target_net = net if net is not None else self.network
        concepts = [str(concept).strip() for concept in (new_concepts or []) if str(concept).strip()]
        if not concepts:
            return {
                "new_concepts": 0,
                "new_nodes": 0,
                "pruned_edges": 0,
                "forged_geodesics": 0,
                "active_targets": 0,
            }

        active_targets = self._top_active_nodes(target_net, limit=6)
        pruned_edges = 0
        forged_geodesics = 0
        created_nodes = 0
        concept_nodes: dict[str, Any] = {}

        for concept in concepts:
            concept_id, created = self._resolve_or_create_concept_node(target_net, concept)
            concept_nodes[concept] = concept_id
            if created:
                created_nodes += 1

            pruned_edges += self._prune_weak_paths(target_net, anchor_node_id=concept_id)
            for target_id in active_targets:
                if target_id == concept_id:
                    continue

                concept_a = self._id_to_concept.get(concept_id, str(concept_id))
                concept_b = self._id_to_concept.get(target_id, str(target_id))
                distance = self._gravity_engine.semantic_distance(concept_a, concept_b)
                weight, delay_ms = self._gravity_engine.derive_weight_and_delay(distance)

                # BitNet path uses ternary {-1, 0, 1}; keep forged paths explicitly excitatory.
                target_weight = max(0.75, float(weight))
                delay = max(1.0, float(delay_ms))

                if self._upsert_connection(target_net, concept_id, target_id, target_weight, delay):
                    forged_geodesics += 1

        self._decay_daily_activity()
        return {
            "new_concepts": len(concepts),
            "new_nodes": int(created_nodes),
            "pruned_edges": int(pruned_edges),
            "forged_geodesics": int(forged_geodesics),
            "active_targets": int(len(active_targets)),
            "concept_nodes": concept_nodes,
        }

    def _select_dream_neurons(self, neuron_ids: list[Any], n_select: int) -> list[Any]:
        # If there is meaningful prior activity, bias toward recently active neurons.
        ranked: list[tuple[float, Any]] = []
        for node_id in neuron_ids:
            neuron = self.network.neurons[node_id]
            last_fire_t = float(getattr(neuron, "last_fire_t", float("-inf")))
            ranked.append((last_fire_t, node_id))

        ranked.sort(key=lambda item: item[0], reverse=True)
        finite = [node_id for t, node_id in ranked if math.isfinite(t)]

        if len(finite) >= n_select:
            top_pool = finite[: max(n_select * 2, n_select)]
            if len(top_pool) <= n_select:
                return top_pool
            return self._rng.sample(top_pool, n_select)

        return self._rng.sample(neuron_ids, n_select)

    def _select_weak_zone_nodes(self, target_net, threshold: float) -> list[Any]:
        weak_nodes: set[Any] = set()
        for pre_id, outgoing in target_net.synapses.items():
            for post_id, weight, _delay_ms in list(outgoing):
                if abs(float(weight)) <= float(threshold):
                    weak_nodes.add(pre_id)
                    weak_nodes.add(post_id)
        return list(weak_nodes)

    def _hard_prune_low_weights(self, target_net, threshold: float) -> int:
        pruned_edges = 0
        for pre_id in list(target_net.synapses.keys()):
            outgoing = list(target_net.synapses[pre_id])
            updated_edges: list[tuple[Any, float, float]] = []
            for post_id, weight, delay_ms in outgoing:
                current_weight = float(weight)
                if abs(current_weight) < float(threshold):
                    if current_weight != 0.0:
                        pruned_edges += 1
                    updated_edges.append((post_id, 0.0, float(delay_ms)))
                    continue
                updated_edges.append((post_id, current_weight, float(delay_ms)))
            target_net.synapses[pre_id] = updated_edges
        return pruned_edges

    def _register_activity(self, trace: list[dict[str, Any]], activation_log: dict[Any, list[float]]) -> None:
        for row in trace:
            node_id = row.get("node_id")
            if node_id is None:
                continue
            self._daily_firing_counts[node_id] += 1
            if node_id not in activation_log:
                activation_log[node_id] = [float(row.get("time_ms", 0.0))]

    def _top_active_nodes(self, target_net, limit: int) -> list[Any]:
        if self._daily_firing_counts:
            ranked = sorted(self._daily_firing_counts.items(), key=lambda item: item[1], reverse=True)
            return [node_id for node_id, _ in ranked[: max(1, limit)] if node_id in target_net.neurons]

        ranked_last_fire: list[tuple[float, Any]] = []
        for node_id, neuron in target_net.neurons.items():
            ranked_last_fire.append((float(getattr(neuron, "last_fire_t", float("-inf"))), node_id))
        ranked_last_fire.sort(key=lambda item: item[0], reverse=True)
        return [node_id for fire_t, node_id in ranked_last_fire[: max(1, limit)] if math.isfinite(fire_t)]

    def _resolve_or_create_concept_node(self, target_net, concept: str) -> tuple[Any, bool]:
        concept_key = self._gravity_engine.concept_to_id(concept)
        if concept_key in self._concept_to_id and self._concept_to_id[concept_key] in target_net.neurons:
            return self._concept_to_id[concept_key], False

        int_ids = [node_id for node_id in target_net.neurons.keys() if isinstance(node_id, int)]
        next_id = (max(int_ids) + 1) if int_ids else 0
        while next_id in target_net.neurons:
            next_id += 1

        target_net.add_neuron(
            node_id=next_id,
            neuron_instance=LIFNeuron(v_thresh=1.0, tau=20.0, refractory_period=5.0),
        )
        self._concept_to_id[concept_key] = next_id
        self._id_to_concept[next_id] = concept
        return next_id, True

    def _prune_weak_paths(self, target_net, anchor_node_id: Any) -> int:
        removed = 0
        for pre_id in list(target_net.synapses.keys()):
            outgoing = list(target_net.synapses[pre_id])
            filtered: list[tuple[Any, float, float]] = []
            for post_id, weight, delay_ms in outgoing:
                is_weak = abs(float(weight)) < 0.5
                touches_anchor = pre_id == anchor_node_id or post_id == anchor_node_id
                if is_weak and touches_anchor:
                    removed += 1
                    continue
                filtered.append((post_id, float(weight), float(delay_ms)))
            target_net.synapses[pre_id] = filtered
        return removed

    def _upsert_connection(self, target_net, pre_id: Any, post_id: Any, weight: float, delay_ms: float) -> bool:
        outgoing = list(target_net.synapses.get(pre_id, []))
        for idx, (edge_post_id, _, _) in enumerate(outgoing):
            if edge_post_id != post_id:
                continue
            outgoing[idx] = (post_id, float(weight), float(delay_ms))
            target_net.synapses[pre_id] = outgoing
            return True

        target_net.add_connection(pre_id=pre_id, post_id=post_id, weight=float(weight), delay_ms=float(delay_ms))
        return True

    def _decay_daily_activity(self) -> None:
        decayed: dict[Any, int] = {}
        for node_id, count in self._daily_firing_counts.items():
            reduced = int(count * 0.85)
            if reduced > 0:
                decayed[node_id] = reduced
        self._daily_firing_counts = defaultdict(int, decayed)

    def _consolidate_ternary_weights_from_activity(self, activation_log: dict[Any, list[float]]) -> int:
        consolidated = 0

        # Preserve insertion order from native dict/list and update in place.
        for pre_id in list(self.network.synapses.keys()):
            outgoing = list(self.network.synapses[pre_id])
            changed_any = False

            pre_times = activation_log.get(pre_id, [])
            if not pre_times:
                continue

            pre_mean = sum(pre_times) / float(len(pre_times))
            updated_edges: list[tuple[Any, float, float]] = []

            for post_id, weight, delay_ms in outgoing:
                post_times = activation_log.get(post_id, [])
                current_weight = float(weight)

                if not post_times:
                    updated_edges.append((post_id, _to_ternary(current_weight), float(delay_ms)))
                    continue

                post_mean = sum(post_times) / float(len(post_times))
                # Cross-pair intensity accelerates phase transition in dream mode.
                pair_count = max(1, (len(pre_times) * len(post_times)) // 2)

                # Causal direction determines STDP sign in dream consolidation.
                if abs(pre_mean - post_mean) <= 1.0:
                    parity_token = (hash(str(pre_id)) ^ hash(str(post_id))) & 1
                    dt = 1.0 if parity_token == 0 else -1.0
                else:
                    dt = 1.0 if pre_mean <= post_mean else -1.0
                syn = SynapseSTDPBitNet(weight=current_weight, quantization="bitnet_1_58")
                if hasattr(syn, "propagate_many"):
                    syn.propagate_many(n_pairs=pair_count, dt_pre_post_ms=dt)
                else:
                    for i in range(pair_count):
                        if dt > 0.0:
                            syn.update(pre_t_ms=float(i), post_t_ms=float(i) + 1.0)
                        else:
                            syn.update(pre_t_ms=float(i) + 1.0, post_t_ms=float(i))

                new_weight = _to_ternary(float(getattr(syn, "weight", current_weight)))
                if new_weight != _to_ternary(current_weight):
                    changed_any = True
                    consolidated += 1

                updated_edges.append((post_id, new_weight, float(delay_ms)))

            if changed_any:
                self.network.synapses[pre_id] = updated_edges

        return consolidated


__all__ = ["Mempalace"]
