import json
import random
import struct
from pathlib import Path

from core.canonical_hasher import CanonicalHasher
from core.lif_neuron import LIFNeuron
from core.spiking_network import SpikingNetwork


def _identity_bytes(network: SpikingNetwork) -> bytes:
    neuron_ids = sorted(int(node_id) for node_id in network.neurons.keys())

    weighted_edges = []
    for pre_id in sorted(list(network.synapses.keys()), key=repr):
        outgoing = list(network.synapses[pre_id])
        for edge_index, (post_id, weight, _delay_ms) in enumerate(outgoing):
            weighted_edges.append((int(pre_id), int(post_id), edge_index, float(weight)))

    weighted_edges.sort(key=lambda edge: (repr(edge[0]), repr(edge[1]), edge[2]))

    out = bytearray()
    out.extend(struct.pack("<I", len(neuron_ids)))
    for node_id in neuron_ids:
        out.extend(struct.pack("<i", node_id))

    out.extend(struct.pack("<Q", len(weighted_edges)))
    for pre_id, post_id, edge_index, weight in weighted_edges:
        out.extend(struct.pack("<iiif", pre_id, post_id, edge_index, weight))

    return bytes(out)


def test_snapshot_binary_clone_preserves_neuron_ids_and_weights(tmp_path: Path):
    rng = random.Random(1337)
    brain_path = tmp_path / "brain_alpha.bin"

    network1 = SpikingNetwork()

    for node_id in range(8):
        network1.add_neuron(node_id=node_id, neuron_instance=LIFNeuron(v_thresh=1.0 + rng.random()))

    has_connection = False
    for pre_id in range(8):
        for post_id in range(8):
            if pre_id == post_id:
                continue
            if rng.random() < 0.55:
                has_connection = True
                network1.add_connection(
                    pre_id=pre_id,
                    post_id=post_id,
                    weight=rng.uniform(0.05, 0.95),
                    delay_ms=rng.uniform(0.25, 3.0),
                )

    if not has_connection:
        network1.add_connection(pre_id=0, post_id=1, weight=rng.uniform(0.05, 0.95), delay_ms=1.0)

    network1.save_weights(brain_path)

    assert brain_path.exists(), "Snapshot binario brain_alpha.bin nao foi criado."

    manifest_path = Path(str(brain_path) + ".manifest.json")
    assert manifest_path.exists(), "Manifesto JSON com hash nao foi criado junto do binario."

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(manifest.keys()) == {
        "timestamp",
        "hash",
        "parent_hash",
        "n_neurons",
        "n_synapses",
    }
    assert manifest["parent_hash"] is None
    persisted = SpikingNetwork()
    persisted.load_weights(brain_path)

    assert manifest["n_neurons"] == len(persisted.neurons)
    assert manifest["n_synapses"] == len(persisted.get_synaptic_strengths())
    assert CanonicalHasher.verify_integrity(persisted, manifest["hash"])

    network2 = SpikingNetwork()
    assert dict(network2.neurons) == {}
    assert dict(network2.synapses) == {}

    network2.load_weights(brain_path)

    assert _identity_bytes(network1) == _identity_bytes(network2), (
        "Rede clonada divergiu da original: IDs de neuronios e/ou pesos nao estao "
        "identicos em comparacao binaria."
    )