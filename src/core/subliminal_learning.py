from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from core.wave_plasticity import TransformedWeightWave

@dataclass(frozen=True)
class AITeacher:
    teacher_hz: float = 40.0
    teacher_phase: float = 1.57  # pi/2, um traço comportamental "escondido" na fase
    coupling_strength: float = 0.15

class SubliminalLearning:
    def __init__(self, native_engine: Any = None, teacher_hz: float = 40.0) -> None:
        self.native_engine = native_engine  # Mantido por compatibilidade
        self.teacher = AITeacher(teacher_hz=teacher_hz)

    def expose_student(self, duration_ms: float, initial_alignment: str = "similar") -> dict[str, Any]:
        """
        Expõe o estudante a uma onda transformada de peso subliminar.
        A característica não é passada via semântica, mas sim forçando 
        a ressonância e o phase-locking da onda sináptica.
        """
        if initial_alignment not in {"similar", "opposite"}:
            raise ValueError("initial_alignment must be 'similar' or 'opposite'")

        # Inicializa a onda de peso do estudante
        student_phase = 1.57 if initial_alignment == "similar" else 4.71 # pi/2 vs 3pi/2
        student_wave = TransformedWeightWave(
            amplitude=1.0,
            frequency_hz=self.teacher.teacher_hz,
            phase=student_phase
        )

        steps = int(duration_ms)
        phase_errors = []
        
        for step in range(steps):
            # O professor modula a ponte (a onda do estudante sofre phase-locking)
            error = student_wave.lock_phase(self.teacher.teacher_phase, coupling_strength=self.teacher.coupling_strength)
            phase_errors.append(error)

        convergence_time_ms = float(duration_ms)
        for i, err in enumerate(phase_errors):
            if err < 0.01:
                convergence_time_ms = float(i)
                break

        return {
            "final_phase": float(student_wave.phase),
            "convergence_time_ms": float(convergence_time_ms),
            "phase_errors_initial": phase_errors[:5],
            "phase_errors_final": phase_errors[-5:] if len(phase_errors) >= 5 else phase_errors
        }
