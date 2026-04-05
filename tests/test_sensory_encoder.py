from typing import Any

import importlib

import pytest

from core.hyperbitnet import HyperBitnet


DURATION_MS = 1000
HIGH_FIRING_RATE_THRESHOLD = 80
EXPECTED_REST_SPIKES = 0


def _count_spikes(spike_train: Any) -> int:
    """
    Normalize common spike-train formats into a total spike count for 1 second.
    """
    if spike_train is None:
        return 0

    if isinstance(spike_train, dict):
        for key in ("spikes", "events", "train", "spike_train"):
            if key in spike_train:
                return _count_spikes(spike_train[key])
        raise AssertionError(
            "SensoryEncoder returned a dict without a known spike-train key."
        )

    if isinstance(spike_train, (list, tuple)):
        if not spike_train:
            return 0

        # Binary/bool stream case (e.g., [0, 1, 0, 1, ...]).
        if all(isinstance(x, (bool, int)) for x in spike_train):
            return int(sum(1 for x in spike_train if bool(x)))

        # Timestamp/event stream case (e.g., [0.01, 0.03, ...]).
        return len(spike_train)

    raise AssertionError(
        "Unsupported spike-train format. Return list/tuple or dict with spike data."
    )


@pytest.fixture
def sensory_encoder_cls():
    try:
        module = importlib.import_module("core.sensory_encoder")
    except ModuleNotFoundError as exc:
        pytest.fail(
            "TDD Red: implemente core.sensory_encoder.SensoryEncoder para atender "
            "os testes de Rate Coding (>80 spikes/s) e Silencio (0 spikes/s). "
            f"Erro original: {exc}"
        )

    if not hasattr(module, "SensoryEncoder"):
        pytest.fail(
            "TDD Red: modulo core.sensory_encoder encontrado, mas sem classe "
            "SensoryEncoder."
        )

    return module.SensoryEncoder


def test_focus_intent_generates_high_spike_rate(sensory_encoder_cls):
    net = HyperBitnet(n_nodes=8, seed=42)
    net.inject_state(intent_state=1)

    encoder = sensory_encoder_cls(net)
    spike_train = encoder.encode(duration_ms=DURATION_MS)
    spike_count = _count_spikes(spike_train)

    assert spike_count > HIGH_FIRING_RATE_THRESHOLD, (
        "Focus state must generate high firing rate in 1 second "
        f"(expected > {HIGH_FIRING_RATE_THRESHOLD}, got {spike_count})."
    )


def test_rest_intent_generates_no_spikes(sensory_encoder_cls):
    net = HyperBitnet(n_nodes=8, seed=42)
    net.inject_state(intent_state=-1)

    encoder = sensory_encoder_cls(net)
    spike_train = encoder.encode(duration_ms=DURATION_MS)
    spike_count = _count_spikes(spike_train)

    assert spike_count == EXPECTED_REST_SPIKES, (
        "Rest state must be silent in 1 second "
        f"(expected {EXPECTED_REST_SPIKES}, got {spike_count})."
    )
