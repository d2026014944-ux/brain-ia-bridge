from core._native_loader import load_native_core


_native = load_native_core()
SynapseSTDP = _native.SynapseSTDP
SynapseSTDPBitNet = _native.SynapseSTDPBitNet

__all__ = ["SynapseSTDP", "SynapseSTDPBitNet"]
