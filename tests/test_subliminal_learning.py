import importlib
import math
from typing import Any
import pytest

EXPOSURE_MS = 500.0
SUBLIMINAL_MAX_CONVERGENCE_MS = 50.0

def _extract(report: dict[str, Any], key: str) -> Any:
    if key not in report:
        raise AssertionError(f"Resultado do aprendizado sublinar sem chave obrigatoria: {key}.")
    return report[key]

@pytest.fixture
def subliminal_learning_cls():
    module = importlib.import_module("core.subliminal_learning")
    return module.SubliminalLearning

def test_teacher_wave_phase_locking_entrains_student(subliminal_learning_cls):
    """
    Sincronia + Entrainment via Ondas Transformadas de Peso (arXiv:2507.14805):
    O professor transmite uma fase "comportamental" secreta ao aluno via ressonância.
    Após 500ms, o erro de fase deve ser próximo de zero.
    """
    learning = subliminal_learning_cls(teacher_hz=40.0)
    report = learning.expose_student(duration_ms=EXPOSURE_MS, initial_alignment="similar")

    final_phase = float(_extract(report, "final_phase"))
    teacher_phase = learning.teacher.teacher_phase
    
    # O aluno ressoou e aprendeu o traço comportamental escondido na fase?
    diff = abs((final_phase - teacher_phase + math.pi) % (2 * math.pi) - math.pi)
    assert diff < 0.01, f"O aluno nao alinhou a fase subliminar. Erro: {diff}"

def test_shared_initialization_controls_convergence_speed_wave(subliminal_learning_cls):
    """
    Inicializacao compartilhada em Ondas:
    - Fases opostas => convergencia mais lenta.
    - Fases similares => aprendizado sublinar quase instantaneo.
    """
    similar_report = subliminal_learning_cls(teacher_hz=40.0).expose_student(
        duration_ms=EXPOSURE_MS,
        initial_alignment="similar",
    )

    opposite_report = subliminal_learning_cls(teacher_hz=40.0).expose_student(
        duration_ms=EXPOSURE_MS,
        initial_alignment="opposite",
    )

    similar_conv_ms = float(_extract(similar_report, "convergence_time_ms"))
    opposite_conv_ms = float(_extract(opposite_report, "convergence_time_ms"))

    assert opposite_conv_ms > similar_conv_ms, (
        "Com fases iniciais opostas, o tempo de convergencia deve ser maior "
        "que no caso de fases similares."
    )

    assert similar_conv_ms <= SUBLIMINAL_MAX_CONVERGENCE_MS, (
        "Com fases iniciais similares, o aprendizado sublinar deve ser quase "
        f"instantaneo (<= {SUBLIMINAL_MAX_CONVERGENCE_MS}ms). "
        f"Obtido: {similar_conv_ms:.3f}ms"
    )