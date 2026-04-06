import json
from pathlib import Path

from core.lif_neuron import LIFNeuron
from core.spiking_network import SpikingNetwork
from governance.pentacosagram import Pentacosagram


def _manifest_path(snapshot_path: Path) -> Path:
    return Path(str(snapshot_path) + ".manifest.json")


def _read_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_network() -> SpikingNetwork:
    network = SpikingNetwork()
    for node_id in range(4):
        network.add_neuron(node_id=node_id, neuron_instance=LIFNeuron(v_thresh=1.0 + (node_id * 0.1)))

    network.add_connection(pre_id=0, post_id=1, weight=0.20, delay_ms=1.0)
    network.add_connection(pre_id=1, post_id=2, weight=0.35, delay_ms=1.0)
    network.add_connection(pre_id=2, post_id=3, weight=0.55, delay_ms=2.0)
    return network


def _nudge_first_weight(network: SpikingNetwork, delta: float) -> None:
    pre_id = sorted(list(network.synapses.keys()), key=repr)[0]
    outgoing = network.synapses[pre_id]
    post_id, weight, delay_ms = outgoing[0]
    outgoing[0] = (post_id, float(weight) + delta, delay_ms)


def test_three_snapshots_form_hash_geodesic_chain(tmp_path: Path):
    network = _build_network()

    snapshot1 = tmp_path / "snapshot_1.bin"
    snapshot2 = tmp_path / "snapshot_2.bin"
    snapshot3 = tmp_path / "snapshot_3.bin"

    network.save_weights(snapshot1)
    _nudge_first_weight(network, 1e-5)
    network.save_weights(snapshot2)
    _nudge_first_weight(network, 1e-5)
    network.save_weights(snapshot3)

    manifest1 = _read_manifest(_manifest_path(snapshot1))
    manifest2 = _read_manifest(_manifest_path(snapshot2))
    manifest3 = _read_manifest(_manifest_path(snapshot3))

    for manifest in (manifest1, manifest2, manifest3):
        assert set(manifest.keys()) == {
            "timestamp",
            "hash",
            "parent_hash",
            "n_neurons",
            "n_synapses",
        }

    assert manifest1["parent_hash"] is None
    assert manifest2["parent_hash"] == manifest1["hash"]
    assert manifest3["parent_hash"] == manifest2["hash"]


def test_chain_verification_fails_after_snapshot_two_tampering(tmp_path: Path):
    network = _build_network()

    snapshot1 = tmp_path / "snapshot_1.bin"
    snapshot2 = tmp_path / "snapshot_2.bin"
    snapshot3 = tmp_path / "snapshot_3.bin"

    network.save_weights(snapshot1)
    _nudge_first_weight(network, 1e-5)
    network.save_weights(snapshot2)
    _nudge_first_weight(network, 1e-5)
    network.save_weights(snapshot3)

    pentacosagram = Pentacosagram(tmp_path)
    manifest3_path = _manifest_path(snapshot3)

    assert pentacosagram.verify_integrity(manifest3_path)

    payload = bytearray(snapshot2.read_bytes())
    payload[-1] ^= 0x01
    snapshot2.write_bytes(payload)

    assert not pentacosagram.verify_integrity(manifest3_path)
