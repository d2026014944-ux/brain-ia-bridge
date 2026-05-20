import importlib

import pytest

from core.lif_neuron import LIFNeuron
from core.spiking_network import SpikingNetwork


@pytest.fixture
def thought_decoder_module():
    try:
        module = importlib.import_module("core.thought_decoder")
    except ModuleNotFoundError as exc:
        pytest.fail(
            "TDD Red: implemente core.thought_decoder com API read_thought para leitura causal. "
            f"Erro original: {exc}"
        )

    if not hasattr(module, "read_thought"):
        pytest.fail("TDD Red: modulo core.thought_decoder encontrado, mas sem funcao read_thought.")

    return module


def _new_network() -> SpikingNetwork:
    try:
        return SpikingNetwork(learning_enabled=False)
    except TypeError:
        net = SpikingNetwork()
        if hasattr(net, "learning_enabled"):
            net.learning_enabled = False
        return net


def _add_neurons(net: SpikingNetwork, n_nodes: int) -> None:
    for node_id in range(n_nodes):
        net.add_neuron(node_id=node_id, neuron_instance=LIFNeuron(v_thresh=1.0, tau=20.0, refractory_period=5.0))


def _set_connection(net: SpikingNetwork, pre_id: int, post_id: int, weight: float, delay_ms: float) -> None:
    edges = list(net.synapses.get(pre_id, []))
    updated = False
    for idx, edge in enumerate(edges):
        edge_post, _, _ = edge
        if edge_post == post_id:
            edges[idx] = (post_id, float(weight), float(delay_ms))
            updated = True
            break
    if not updated:
        edges.append((post_id, float(weight), float(delay_ms)))
    net.synapses[pre_id] = edges


def _chain_network() -> SpikingNetwork:
    net = _new_network()
    _add_neurons(net, 4)

    net.add_connection(pre_id=0, post_id=1, weight=1.35, delay_ms=1.0)  # Chuva -> Molhado
    net.add_connection(pre_id=1, post_id=2, weight=1.30, delay_ms=1.0)  # Molhado -> Oceano
    # Fogo (3) permanece desconectado.
    return net


def _branch_network() -> SpikingNetwork:
    net = _new_network()
    _add_neurons(net, 6)

    # Ramo baseline preferido
    net.add_connection(pre_id=0, post_id=1, weight=1.35, delay_ms=1.0)  # Chuva -> Molhado
    net.add_connection(pre_id=1, post_id=2, weight=1.25, delay_ms=1.0)  # Molhado -> Oceano

    # Ramo alternativo
    net.add_connection(pre_id=0, post_id=4, weight=1.15, delay_ms=1.05)  # Chuva -> Vapor
    net.add_connection(pre_id=4, post_id=5, weight=1.25, delay_ms=1.0)   # Vapor -> Nuvem
    return net


def test_rain_wet_ocean_chain_and_fire_exclusion(thought_decoder_module):
    network = _chain_network()

    concept_to_id = {
        "Chuva": 0,
        "Molhado": 1,
        "Oceano": 2,
        "Fogo": 3,
    }
    id_to_concept = {node_id: concept for concept, node_id in concept_to_id.items()}

    result = thought_decoder_module.read_thought(
        network=network,
        start_concept="Chuva",
        energy=1.25,
        concept_to_id=concept_to_id,
        id_to_concept=id_to_concept,
    )

    concepts = [step["concept"] for step in result.sequence]

    assert concepts[:3] == ["Chuva", "Molhado", "Oceano"]
    assert "Fogo" not in concepts
    assert result.natural_language.startswith("Chuva causa Molhado")
    assert result.confidence_score >= 0.65
    assert result.coherence_score >= 0.60


def test_noise_spike_without_descendants_is_rejected(thought_decoder_module):
    network = _chain_network()

    concept_to_id = {
        "Chuva": 0,
        "Molhado": 1,
        "Oceano": 2,
        "Fogo": 3,
    }
    id_to_concept = {node_id: concept for concept, node_id in concept_to_id.items()}

    # Ruido isolado: Fogo dispara sem descendencia causal.
    network.schedule_event(time_ms=0.0, target_id=3, weight=1.2)

    result = thought_decoder_module.read_thought(
        network=network,
        start_concept="Chuva",
        energy=1.25,
        concept_to_id=concept_to_id,
        id_to_concept=id_to_concept,
    )

    concepts = [step["concept"] for step in result.sequence]
    assert concepts[:3] == ["Chuva", "Molhado", "Oceano"]
    assert "Fogo" in result.rejected_noise_nodes


def test_geodesic_stability_under_local_distortion(thought_decoder_module):
    baseline = _branch_network()
    perturbed = _branch_network()

    concept_to_id = {
        "Chuva": 0,
        "Molhado": 1,
        "Oceano": 2,
        "Fogo": 3,
        "Vapor": 4,
        "Nuvem": 5,
    }
    id_to_concept = {node_id: concept for concept, node_id in concept_to_id.items()}

    baseline_result = thought_decoder_module.read_thought(
        network=baseline,
        start_concept="Chuva",
        energy=1.2,
        concept_to_id=concept_to_id,
        id_to_concept=id_to_concept,
    )

    baseline_chain = [step["concept"] for step in baseline_result.sequence[:3]]
    assert baseline_chain == ["Chuva", "Molhado", "Oceano"]

    # Distorcao local controlada no ramo principal.
    _set_connection(perturbed, pre_id=0, post_id=1, weight=0.35, delay_ms=2.2)

    perturbed_result = thought_decoder_module.read_thought(
        network=perturbed,
        start_concept="Chuva",
        energy=1.2,
        concept_to_id=concept_to_id,
        id_to_concept=id_to_concept,
    )

    perturbed_chain = [step["concept"] for step in perturbed_result.sequence[:3]]
    assert perturbed_chain == ["Chuva", "Vapor", "Nuvem"]
