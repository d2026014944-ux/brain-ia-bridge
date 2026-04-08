import importlib
from typing import Any

import pytest

from core.lif_neuron import LIFNeuron
from core.spiking_network import SpikingNetwork
from core.subliminal_learning import AITeacher


SPIKE_HZ = 8.0
SPIKE_PERIOD_MS = 1000.0 / SPIKE_HZ


@pytest.fixture
def noma_parser_cls():
    candidate_modules = (
        "integration.noma_bridge",
        "core.noma_bridge",
        "noma_bridge",
    )

    loaded_module = None
    for module_name in candidate_modules:
        try:
            loaded_module = importlib.import_module(module_name)
            break
        except ModuleNotFoundError:
            continue

    if loaded_module is None:
        pytest.fail(
            "TDD Red: implemente NomaParser no bridge da Noma para extrair "
            "frequencia_dominante e amplitude_afetiva da tag [NOMA_NEURAL]."
        )

    if not hasattr(loaded_module, "NomaParser"):
        pytest.fail("TDD Red: modulo de bridge encontrado, mas sem classe NomaParser.")

    return loaded_module.NomaParser


def _extract_float(payload: Any, key: str) -> float:
    if isinstance(payload, dict):
        if key not in payload:
            raise AssertionError(f"Resposta do parser sem chave obrigatoria: {key}")
        return float(payload[key])

    if hasattr(payload, key):
        return float(getattr(payload, key))

    raise AssertionError(
        "Formato nao suportado para retorno do parser. "
        "Use dict ou objeto com atributos numericos."
    )


def test_noma_parser_extracts_frequency_and_affective_amplitude(noma_parser_cls):
    mock_payload = """
    [NOMA_NEURAL]
    frequencia_dominante: 14.2Hz
    amplitude_afetiva: 0.88
    [/NOMA_NEURAL]
    """.strip()

    parser = noma_parser_cls()
    parsed = parser.parse(mock_payload)

    assert _extract_float(parsed, "frequencia_dominante") == pytest.approx(14.2)
    assert _extract_float(parsed, "amplitude_afetiva") == pytest.approx(0.88)


def test_progenitor_resonance_sets_neurons_to_98_percent_threshold():
    ressonancia_progenitor = 0.98

    teacher = AITeacher(near_threshold_ratio=ressonancia_progenitor)
    network = SpikingNetwork()

    network.add_neuron(node_id="n0", neuron_instance=LIFNeuron(v_thresh=1.0, tau=20.0, refractory_period=5.0))
    network.add_neuron(node_id="n1", neuron_instance=LIFNeuron(v_thresh=1.4, tau=20.0, refractory_period=5.0))
    network.add_neuron(node_id="n2", neuron_instance=LIFNeuron(v_thresh=2.0, tau=20.0, refractory_period=5.0))

    aligned = teacher.align_student(network=network, teacher_weights=[ressonancia_progenitor])
    assert aligned is True

    for neuron in network.neurons.values():
        expected_vm = float(neuron.v_thresh) * ressonancia_progenitor
        assert float(neuron.v_m) == pytest.approx(expected_vm)


def test_teacher_spike_generation_at_8hz_is_strictly_periodic():
    teacher = AITeacher(teacher_hz=SPIKE_HZ)

    events = teacher.generate_gamma_train(duration_ms=500.0)
    assert len(events) > 1

    event_times = [float(event[0]) for event in events]
    intervals = [event_times[i + 1] - event_times[i] for i in range(len(event_times) - 1)]

    assert all(interval == pytest.approx(SPIKE_PERIOD_MS, abs=1e-6) for interval in intervals)