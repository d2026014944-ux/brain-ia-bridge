import math

class TransformedWeightWave:
    def __init__(self, amplitude: float = 1.0, frequency_hz: float = 40.0, phase: float = 0.0):
        """
        O peso não é mais um escalar fixo, mas uma onda contínua no domínio do tempo (Onda Transformada de Peso).
        W(t) = A * cos(2 * pi * f * t + phi)
        """
        self.amplitude = amplitude
        self.frequency_hz = frequency_hz
        self.phase = phase

    def get_weight_at(self, time_s: float) -> float:
        """Retorna o valor momentâneo do peso (a onda) no tempo t (em segundos)."""
        return self.amplitude * math.cos(2 * math.pi * self.frequency_hz * time_s + self.phase)

    def lock_phase(self, target_phase: float, coupling_strength: float = 0.1) -> float:
        """
        Alinhamento subliminar de fase (Ressonância).
        O Aluno ajusta sua fase em direção à fase do professor.
        """
        # Distância circular mais curta
        diff = (target_phase - self.phase + math.pi) % (2 * math.pi) - math.pi
        self.phase += diff * coupling_strength
        self.phase = self.phase % (2 * math.pi)
        return abs(diff)
