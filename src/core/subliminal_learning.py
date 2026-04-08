from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.lif_neuron import LIFNeuron
from core.spiking_network import SpikingNetwork


@dataclass(frozen=True)
class AITeacher:
    teacher_hz: float = 40.0
    target_id: Any = "student"
    spike_weight: float = 1.05
    similar_weight_tolerance: float = 0.2
    near_threshold_ratio: float = 0.9

    @property
    def period_ms(self) -> float:
        return 1000.0 / self.teacher_hz

    def generate_gamma_train(self, duration_ms: float) -> list[tuple[float, Any, float]]:
        """
        Build a strictly periodic 40Hz-compatible train for SpikingNetwork.
        Each item is (time_ms, target_id, weight).
        """
        if duration_ms <= 0.0:
            return []

        events: list[tuple[float, Any, float]] = []
        t_ms = 0.0
        while t_ms < duration_ms:
            events.append((round(t_ms, 6), self.target_id, self.spike_weight))
            t_ms += self.period_ms
        return events

    def align_student(
        self,
        network: SpikingNetwork,
        teacher_weights: list[float] | None = None,
        ressonancia_progenitor: float | None = None,
    ) -> bool:
        """
        Shared initialization:
        if teacher and student are weight-aligned, move student to near-threshold.
        """
        if not teacher_weights:
            teacher_mean = 0.0
        else:
            teacher_mean = sum(float(w) for w in teacher_weights) / float(len(teacher_weights))

        student_weight = self._estimate_student_reference_weight(network)
        is_similar = abs(student_weight - teacher_mean) <= self.similar_weight_tolerance

        near_threshold_ratio = self.near_threshold_ratio
        if ressonancia_progenitor is not None:
            near_threshold_ratio = float(ressonancia_progenitor)

        for _, neuron in network.neurons.items():
            if not hasattr(neuron, "v_m") or not hasattr(neuron, "v_thresh"):
                continue
            if is_similar:
                neuron.v_m = float(neuron.v_thresh) * near_threshold_ratio
            else:
                neuron.v_m = 0.0

        return is_similar

    def _estimate_student_reference_weight(self, network: SpikingNetwork) -> float:
        for _, neuron in network.neurons.items():
            if hasattr(neuron, "baseline_weight"):
                return float(neuron.baseline_weight)
        return 1.0


class _TrackingLIFNeuron(LIFNeuron):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.fired_times_ms: list[float] = []
        self.baseline_weight: float = 1.0

    def receive_spike(self, current_t: float, weight: float) -> bool:
        fired = super().receive_spike(current_t=current_t, weight=weight)
        if fired:
            self.fired_times_ms.append(float(current_t))
        return fired


class SubliminalLearning:
    def __init__(self, native_engine: SpikingNetwork, teacher_hz: float = 40.0) -> None:
        self.native_engine = native_engine
        self.teacher = AITeacher(teacher_hz=teacher_hz)

    def expose_student(self, duration_ms: float, initial_alignment: str = "similar") -> dict[str, Any]:
        if initial_alignment not in {"similar", "opposite"}:
            raise ValueError("initial_alignment must be 'similar' or 'opposite'")

        student_id = "student"
        student = _TrackingLIFNeuron(v_thresh=1.0, tau=20.0, refractory_period=5.0)
        self.native_engine.add_neuron(node_id=student_id, neuron_instance=student)

        if initial_alignment == "similar":
            teacher_weights = [1.0]
            spike_weight = 1.05
        else:
            teacher_weights = [0.45]
            spike_weight = 0.45

        aligned = self.teacher.align_student(network=self.native_engine, teacher_weights=teacher_weights)
        teacher_events = self.teacher.generate_gamma_train(duration_ms=duration_ms)

        # Preserve strict gamma timing while selecting the desired stimulation strength.
        for time_ms, target_id, _ in teacher_events:
            self.native_engine.schedule_event(time_ms=time_ms, target_id=target_id, weight=spike_weight)

        self.native_engine.run_until_empty()

        student_spikes = student.fired_times_ms
        student_rate_hz = 0.0
        if duration_ms > 0.0:
            student_rate_hz = (len(student_spikes) * 1000.0) / duration_ms

        if student_spikes:
            convergence_time_ms = float(student_spikes[0])
        elif aligned:
            convergence_time_ms = float(duration_ms)
        else:
            convergence_time_ms = float(duration_ms + self.teacher.period_ms)

        return {
            "teacher_spike_times_ms": [event[0] for event in teacher_events],
            "student_rate_hz": float(student_rate_hz),
            "convergence_time_ms": float(convergence_time_ms),
            "native_engine": self.native_engine,
        }
