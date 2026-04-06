import importlib
from typing import Any

import pytest

from core.spiking_network import SpikingNetwork


GAMMA_HZ = 40.0
GAMMA_PERIOD_MS = 1000.0 / GAMMA_HZ
EXPOSURE_MS = 500.0
RATE_EPSILON = 0.05
SUBLIMINAL_MAX_CONVERGENCE_MS = 50.0


def _relative_error(target: float, value: float) -> float:
    return abs(value - target) / target


def _extract(report: dict[str, Any], key: str) -> Any:
    if key not in report:
        raise AssertionError(
            f"Resultado do aprendizado sublinar sem chave obrigatoria: {key}."
        )
    return report[key]


@pytest.fixture
def subliminal_learning_cls():
    try:
        module = importlib.import_module("core.subliminal_learning")
    except ModuleNotFoundError as exc:
        pytest.fail(
            "TDD Red: implemente core.subliminal_learning.SubliminalLearning para "
            "sincronizar um Professor em 40Hz com o Aluno no motor C++, incluindo "
            "metrica de convergencia temporal e taxa de disparo. "
            f"Erro original: {exc}"
        )

    if not hasattr(module, "SubliminalLearning"):
        pytest.fail(
            "TDD Red: modulo core.subliminal_learning encontrado, mas sem classe "
            "SubliminalLearning."
        )

    return module.SubliminalLearning


def test_teacher_gamma_sync_is_injected_into_cpp_engine_and_entrains_student(
    subliminal_learning_cls,
):
    """
    Sincronia + Entrainment:
    1) Professor em 40Hz gera padrao gamma injetado no motor C++.
    2) Apos 500ms, taxa de disparo do Aluno converge para 40Hz com erro < 5%.
    """
    native_engine = SpikingNetwork()

    learning = subliminal_learning_cls(
        native_engine=native_engine,
        teacher_hz=GAMMA_HZ,
    )

    report = learning.expose_student(
        duration_ms=EXPOSURE_MS,
        initial_alignment="similar",
    )

    teacher_spike_times = _extract(report, "teacher_spike_times_ms")
    student_rate_hz = float(_extract(report, "student_rate_hz"))
    injected_engine = _extract(report, "native_engine")

    assert injected_engine is native_engine
    assert len(teacher_spike_times) > 1

    intervals = [
        teacher_spike_times[i + 1] - teacher_spike_times[i]
        for i in range(len(teacher_spike_times) - 1)
    ]
    assert all(interval == pytest.approx(GAMMA_PERIOD_MS, abs=1e-3) for interval in intervals)

    assert _relative_error(GAMMA_HZ, student_rate_hz) < RATE_EPSILON, (
        "Aluno deve entrar em arrastre com o Professor em ritmo gamma de 40Hz "
        f"apos {EXPOSURE_MS}ms (erro relativo < {RATE_EPSILON * 100:.0f}%). "
        f"Obtido: {student_rate_hz:.3f}Hz"
    )


def test_shared_initialization_controls_convergence_speed(subliminal_learning_cls):
    """
    Inicializacao compartilhada:
    - Pesos opostos => convergencia mais lenta.
    - Pesos similares => aprendizado sublinar quase instantaneo.
    """
    similar_report = subliminal_learning_cls(
        native_engine=SpikingNetwork(),
        teacher_hz=GAMMA_HZ,
    ).expose_student(
        duration_ms=EXPOSURE_MS,
        initial_alignment="similar",
    )

    opposite_report = subliminal_learning_cls(
        native_engine=SpikingNetwork(),
        teacher_hz=GAMMA_HZ,
    ).expose_student(
        duration_ms=EXPOSURE_MS,
        initial_alignment="opposite",
    )

    similar_conv_ms = float(_extract(similar_report, "convergence_time_ms"))
    opposite_conv_ms = float(_extract(opposite_report, "convergence_time_ms"))

    assert opposite_conv_ms > similar_conv_ms, (
        "Com pesos iniciais opostos, o tempo de convergencia deve ser maior "
        "que no caso de pesos similares."
    )

    assert similar_conv_ms <= SUBLIMINAL_MAX_CONVERGENCE_MS, (
        "Com pesos iniciais similares, o aprendizado sublinar deve ser quase "
        f"instantaneo (<= {SUBLIMINAL_MAX_CONVERGENCE_MS}ms). "
        f"Obtido: {similar_conv_ms:.3f}ms"
    )