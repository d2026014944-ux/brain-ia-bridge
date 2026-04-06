from core.canonical_hasher import CanonicalHasher
from core.lif_neuron import LIFNeuron
from core.spiking_network import SpikingNetwork


def _build_sample_network() -> SpikingNetwork:
    network = SpikingNetwork()

    for node_id in range(4):
        network.add_neuron(node_id=node_id, neuron_instance=LIFNeuron(v_thresh=1.0 + (node_id * 0.1)))

    network.add_connection(pre_id=0, post_id=1, weight=0.200000, delay_ms=1.0)
    network.add_connection(pre_id=0, post_id=2, weight=0.350000, delay_ms=2.0)
    network.add_connection(pre_id=2, post_id=3, weight=0.775000, delay_ms=1.5)

    return network


def test_compute_state_hash_is_deterministic_for_same_network_state():
    network = _build_sample_network()

    hash_a = CanonicalHasher.compute_state_hash(network)
    hash_b = CanonicalHasher.compute_state_hash(network)

    assert hash_a == hash_b
    assert CanonicalHasher.verify_integrity(network, hash_a)


def test_minimal_weight_change_alters_final_hash():
    network = _build_sample_network()

    original_hash = CanonicalHasher.compute_state_hash(network)

    outgoing = network.synapses[0]
    post_id, weight, delay_ms = outgoing[0]
    outgoing[0] = (post_id, float(weight) + 1e-7, delay_ms)

    changed_hash = CanonicalHasher.compute_state_hash(network)

    assert changed_hash != original_hash
    assert not CanonicalHasher.verify_integrity(network, original_hash)
