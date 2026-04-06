import time
from pathlib import Path

from core.canonical_hasher import CanonicalHasher
from core._native_loader import load_native_core
from governance.pentacosagram import Pentacosagram


_native = load_native_core()
_NativeSpikingNetwork = _native.SpikingNetwork


def _manifest_path_for(binary_path: Path) -> Path:
    return binary_path.with_name(binary_path.name + ".manifest.json")


class SpikingNetwork(_NativeSpikingNetwork):
    def save_weights(self, path) -> None:
        binary_path = Path(path)
        super().save_weights(str(binary_path))

        persisted_state = _NativeSpikingNetwork()
        persisted_state.load_weights(str(binary_path))

        current_manifest = {
            "manifest_path": _manifest_path_for(binary_path),
            "timestamp": time.time(),
            "hash": CanonicalHasher.compute_state_hash(persisted_state),
            "n_neurons": len(persisted_state.neurons),
            "n_synapses": len(persisted_state.get_synaptic_strengths()),
        }
        Pentacosagram(binary_path.parent).append_state(current_manifest)

    def save_snapshot(self, path) -> None:
        self.save_weights(path)

__all__ = ["SpikingNetwork"]
