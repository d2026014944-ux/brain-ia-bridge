from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


ROOT = Path(__file__).resolve().parent

ext_modules = [
    Pybind11Extension(
        "core._native_core",
        [str(ROOT / "src" / "core" / "_native_core.cpp")],
        cxx_std=17,
        extra_compile_args=["-O3"],
    )
]

setup(
    name="brain_ia_bridge_native",
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    package_dir={"": "src"},
)
