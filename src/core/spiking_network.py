from core._native_loader import load_native_core


_native = load_native_core()
SpikingNetwork = _native.SpikingNetwork

__all__ = ["SpikingNetwork"]
