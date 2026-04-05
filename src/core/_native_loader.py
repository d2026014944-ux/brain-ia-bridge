import importlib
import subprocess
import sys
from pathlib import Path


def load_native_core():
    try:
        return importlib.import_module("core._native_core")
    except ModuleNotFoundError:
        try:
            repo_root = Path(__file__).resolve().parents[2]
            subprocess.check_call(
                [
                    sys.executable,
                    str(repo_root / "setup_native.py"),
                    "build_ext",
                    "--inplace",
                ],
                cwd=str(repo_root),
            )
            return importlib.import_module("core._native_core")
        except Exception:
            print("AVISO: Microsoft C++ Build Tools não encontrado. Iniciando modo de Fallback Pure-Python.", file=sys.stderr)
            return importlib.import_module("core._native_fallback")
