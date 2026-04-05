from core._native_loader import load_native_core


_native = load_native_core()
LIFNeuron = _native.LIFNeuron

__all__ = ["LIFNeuron"]
