from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.canonical_hasher import CanonicalHasher


class Pentacosagram:
    def __init__(self, manifests_dir: Path | str) -> None:
        self.manifests_dir = Path(manifests_dir)

    def _manifest_files(self) -> list[Path]:
        if not self.manifests_dir.exists():
            return []
        return sorted(self.manifests_dir.glob("*.manifest.json"))

    @staticmethod
    def _read_manifest(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
            encoding="utf-8",
        )
        tmp_path.replace(path)

    @staticmethod
    def _binary_path_for_manifest(manifest_path: Path) -> Path:
        suffix = ".manifest.json"
        if manifest_path.name.endswith(suffix):
            return manifest_path.with_name(manifest_path.name[: -len(suffix)])
        return manifest_path.with_suffix("")

    def _latest_manifest_hash(self, *, exclude_path: Path | None = None) -> str | None:
        candidates: list[tuple[float, str]] = []
        for manifest_path in self._manifest_files():
            if exclude_path is not None and manifest_path.resolve() == exclude_path.resolve():
                continue
            try:
                manifest = self._read_manifest(manifest_path)
            except (OSError, json.JSONDecodeError):
                continue

            manifest_hash = manifest.get("hash")
            if not isinstance(manifest_hash, str) or len(manifest_hash) != 64:
                continue

            timestamp = manifest.get("timestamp", 0.0)
            if not isinstance(timestamp, (int, float)):
                timestamp = 0.0

            candidates.append((float(timestamp), manifest_hash))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        return candidates[-1][1]

    def append_state(self, current_manifest: dict[str, Any]) -> dict[str, Any]:
        manifest_data = dict(current_manifest)
        manifest_path_raw = manifest_data.pop("manifest_path", None)
        if manifest_path_raw is None:
            raise ValueError("current_manifest must include 'manifest_path'")

        manifest_path = Path(manifest_path_raw)
        parent_hash = self._latest_manifest_hash(exclude_path=manifest_path)

        final_manifest = {
            "timestamp": float(manifest_data["timestamp"]),
            "hash": str(manifest_data["hash"]),
            "parent_hash": parent_hash,
            "n_neurons": int(manifest_data["n_neurons"]),
            "n_synapses": int(manifest_data["n_synapses"]),
        }

        self._atomic_write_json(manifest_path, final_manifest)
        return final_manifest

    @staticmethod
    def _is_valid_manifest_shape(manifest: dict[str, Any]) -> bool:
        required = {"timestamp", "hash", "parent_hash", "n_neurons", "n_synapses"}
        if set(manifest.keys()) != required:
            return False
        if not isinstance(manifest["timestamp"], (int, float)):
            return False
        if not isinstance(manifest["hash"], str) or len(manifest["hash"]) != 64:
            return False
        if manifest["parent_hash"] is not None:
            if not isinstance(manifest["parent_hash"], str) or len(manifest["parent_hash"]) != 64:
                return False
        if not isinstance(manifest["n_neurons"], int):
            return False
        if not isinstance(manifest["n_synapses"], int):
            return False
        return True

    def _verify_manifest_against_snapshot(self, manifest_path: Path, manifest: dict[str, Any]) -> bool:
        binary_path = self._binary_path_for_manifest(manifest_path)
        if not binary_path.exists():
            return False

        try:
            from core.spiking_network import SpikingNetwork

            network = SpikingNetwork()
            network.load_weights(binary_path)

            computed_hash = CanonicalHasher.compute_state_hash(network)
            if computed_hash != manifest["hash"]:
                return False

            if int(manifest["n_neurons"]) != int(len(network.neurons)):
                return False

            if int(manifest["n_synapses"]) != int(len(network.get_synaptic_strengths())):
                return False
        except Exception:
            return False

        return True

    def verify_integrity(self, manifest_path: Path | str) -> bool:
        start_path = Path(manifest_path)
        if not start_path.exists():
            return False

        hash_to_manifest_path: dict[str, Path] = {}
        for item in self._manifest_files():
            try:
                data = self._read_manifest(item)
            except (OSError, json.JSONDecodeError):
                return False

            manifest_hash = data.get("hash")
            if isinstance(manifest_hash, str) and len(manifest_hash) == 64:
                hash_to_manifest_path[manifest_hash] = item

        visited_hashes: set[str] = set()
        current_path = start_path

        while True:
            try:
                current_manifest = self._read_manifest(current_path)
            except (OSError, json.JSONDecodeError):
                return False

            if not self._is_valid_manifest_shape(current_manifest):
                return False

            if not self._verify_manifest_against_snapshot(current_path, current_manifest):
                return False

            current_hash = current_manifest["hash"]
            if current_hash in visited_hashes:
                return False
            visited_hashes.add(current_hash)

            parent_hash = current_manifest["parent_hash"]
            if parent_hash is None:
                return True

            parent_path = hash_to_manifest_path.get(parent_hash)
            if parent_path is None:
                return False

            current_path = parent_path
