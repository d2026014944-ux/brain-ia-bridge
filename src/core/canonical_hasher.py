from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any


class CanonicalHasher:
    @staticmethod
    def _canonical_id(node_id: Any) -> str:
        return f"{type(node_id).__name__}:{repr(node_id)}"

    @classmethod
    def _extract_neuron_state(cls, neuron: Any) -> dict[str, Any]:
        state: dict[str, Any] = {}

        # Core LIF fields exposed by the native binding.
        for attr in (
            "tau",
            "v_thresh",
            "refractory_period",
            "v_m",
            "last_update_t",
            "last_fire_t",
        ):
            if not hasattr(neuron, attr):
                continue
            value = getattr(neuron, attr)
            if isinstance(value, (int, float, bool, str)) or value is None:
                state[attr] = value
            else:
                state[attr] = repr(value)

        # Include extra public Python-side attributes when present.
        if hasattr(neuron, "__dict__"):
            for key in sorted(neuron.__dict__.keys()):
                if key.startswith("_") or key in state:
                    continue
                value = neuron.__dict__[key]
                if isinstance(value, (int, float, bool, str)) or value is None:
                    state[key] = value
                else:
                    state[key] = repr(value)

        return state

    @classmethod
    def _build_payload(cls, network: Any) -> dict[str, Any]:
        neurons_payload: list[dict[str, Any]] = []
        neuron_items = sorted(network.neurons.items(), key=lambda item: cls._canonical_id(item[0]))
        for node_id, neuron in neuron_items:
            neurons_payload.append(
                {
                    "id": cls._canonical_id(node_id),
                    "state": cls._extract_neuron_state(neuron),
                }
            )

        synapses_payload: list[dict[str, Any]] = []
        for row in network.get_synaptic_strengths():
            synapses_payload.append(
                {
                    "pre_id": cls._canonical_id(row["pre_id"]),
                    "post_id": cls._canonical_id(row["post_id"]),
                    "weight": float(row["weight"]),
                    "delay_ms": float(row["delay_ms"]),
                }
            )

        synapses_payload.sort(
            key=lambda entry: (
                entry["pre_id"],
                entry["post_id"],
                entry["delay_ms"],
                entry["weight"],
            )
        )

        return {
            "neurons": neurons_payload,
            "synapses": synapses_payload,
        }

    @classmethod
    def compute_state_hash(cls, network: Any) -> str:
        payload = cls._build_payload(network)
        canonical_json = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(canonical_json.encode("ascii")).hexdigest()

    @classmethod
    def verify_integrity(cls, network: Any, stored_hash: str) -> bool:
        if not isinstance(stored_hash, str):
            return False
        expected = stored_hash.strip().lower()
        if len(expected) != 64:
            return False
        current = cls.compute_state_hash(network)
        return hmac.compare_digest(current, expected)
