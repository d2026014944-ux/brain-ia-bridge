from __future__ import annotations

from core.lif_neuron import LIFNeuron
from core.mempalace import Mempalace
from core.spiking_network import SpikingNetwork


def _build_sparse_weak_network(n_nodes: int = 20) -> SpikingNetwork:
    try:
        net = SpikingNetwork(learning_enabled=True)
    except TypeError:
        net = SpikingNetwork()
        if hasattr(net, "learning_enabled"):
            net.learning_enabled = True

    for node_id in range(n_nodes):
        net.add_neuron(node_id=node_id, neuron_instance=LIFNeuron(v_thresh=1.0, tau=20.0, refractory_period=5.0))

    # Weak / near-null topology: weights around zero that collapse to ternary states.
    # Alternate signs to create both potentiation and depression opportunities.
    for pre_id in range(n_nodes):
        post_id = (pre_id + 1) % n_nodes
        weak_weight = 0.20 if pre_id % 2 == 0 else -0.20
        net.add_connection(pre_id=pre_id, post_id=post_id, weight=weak_weight, delay_ms=1.0)

    return net


def _weights(network: SpikingNetwork) -> list[float]:
    return [float(row["weight"]) for row in network.get_synaptic_strengths()]


def test_dream_cycle_scults_bitnet_weights_without_semantic_input():
    network = _build_sparse_weak_network(n_nodes=20)
    before = _weights(network)

    assert all(w in {-1.0, 0.0, 1.0} for w in before)

    palace = Mempalace(network=network, seed=1337, pulse_step_ms=4.0)
    result = palace.trigger_dream_cycle(duration_ms=320.0, noise_energy=1.15)

    after = _weights(network)

    assert result["scheduled_events"] > 0
    assert len(result["selected_neurons"]) >= 2
    assert result["consolidated_edges"] > 0

    assert before != after, "Dream cycle must alter weights through spontaneous activity."
    assert all(w in {-1.0, 0.0, 1.0} for w in after)

    # Emergent polarization: spontaneous consolidation should yield both excitatory and inhibitory traces.
    assert any(w == 1.0 for w in after)
    assert any(w == -1.0 for w in after)
