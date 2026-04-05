import importlib

import pytest

from core.lif_neuron import LIFNeuron


@pytest.fixture
def spiking_network_cls():
    try:
        module = importlib.import_module("core.spiking_network")
    except ModuleNotFoundError as exc:
        pytest.fail(
            "TDD Red: implemente core.spiking_network.SpikingNetwork para atender "
            "os testes de causalidade temporal e propagacao com delay. "
            f"Erro original: {exc}"
        )

    if not hasattr(module, "SpikingNetwork"):
        pytest.fail(
            "TDD Red: modulo core.spiking_network encontrado, mas sem classe "
            "SpikingNetwork."
        )

    return module.SpikingNetwork


def test_events_are_popped_in_strict_chronological_order(spiking_network_cls):
    """
    Causalidade: eventos inseridos fora de ordem devem sair por tempo crescente.
    """
    network = spiking_network_cls()

    network.schedule_event(time_ms=15.0, target_id="A", weight=1.0)
    network.schedule_event(time_ms=5.0, target_id="B", weight=1.0)
    network.schedule_event(time_ms=10.0, target_id="C", weight=1.0)

    e1 = network.pop_next_event()
    e2 = network.pop_next_event()
    e3 = network.pop_next_event()

    assert e1[0] == 5.0
    assert e2[0] == 10.0
    assert e3[0] == 15.0


def test_single_chain_propagates_with_exact_delay(spiking_network_cls):
    """
    Propagacao: neuronio 0 dispara em t=0 e entrega spike em neuronio 1 em t=2.
    """

    class RecordingLIFNeuron(LIFNeuron):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.received_spike_times = []

        def receive_spike(self, current_t: float, weight: float) -> bool:
            self.received_spike_times.append(float(current_t))
            return super().receive_spike(current_t=current_t, weight=weight)

    network = spiking_network_cls()

    neuron0 = RecordingLIFNeuron(v_thresh=1.0)
    neuron1 = RecordingLIFNeuron(v_thresh=10.0)

    network.add_neuron(node_id=0, neuron_instance=neuron0)
    network.add_neuron(node_id=1, neuron_instance=neuron1)
    network.add_connection(pre_id=0, post_id=1, weight=1.5, delay_ms=2.0)

    network.schedule_event(time_ms=0.0, target_id=0, weight=1.5)
    network.run_until_empty()

    assert neuron1.received_spike_times == [2.0]
    assert neuron1.v_m > 0.0
