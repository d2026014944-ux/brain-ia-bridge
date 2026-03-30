from dataclasses import dataclass
from typing import Callable, Dict, Optional
import time
import random


@dataclass
class NeurosityConfig:
    device_id: str
    email: str
    password: str


class NeurosityAdapter:
    """
    Adapter (stub) for Neurosity SDK integration.

    Goals:
    - Standardise output as a feature dict with values in [0, 1].
    - Allow easy swapping between simulated and real SDK streams.
    """

    def __init__(self, config: NeurosityConfig, use_mock_stream: bool = True):
        self.config = config
        self.use_mock_stream = use_mock_stream
        self._connected = False
        self._streaming = False

        # Placeholders for the real SDK
        self._sdk = None
        self._subscription = None

    def connect(self) -> None:
        """
        In real mode:
        - Initialise SDK
        - Authenticate
        - Select device_id
        """
        if self.use_mock_stream:
            self._connected = True
            return

        # TODO: integrate real SDK here
        # Example:
        # self._sdk = NeurositySDK(...)
        # self._sdk.login({"email": self.config.email, "password": self.config.password})
        # self._sdk.select_device(self.config.device_id)
        self._connected = True

    def disconnect(self) -> None:
        self.stop_stream()
        self._connected = False

    @staticmethod
    def _normalize_feature(
        value: float, min_v: float = 0.0, max_v: float = 1.0
    ) -> float:
        if max_v <= min_v:
            return 0.0
        x = (value - min_v) / (max_v - min_v)
        return max(0.0, min(1.0, x))

    def _mock_sample(self) -> Dict[str, float]:
        """Simulate already-normalised EEG features (focus/calm/gamma)."""
        return {
            "focus": self._normalize_feature(random.uniform(0.2, 0.9)),
            "calm": self._normalize_feature(random.uniform(0.2, 0.8)),
            "gamma": self._normalize_feature(random.uniform(0.2, 0.9)),
            "timestamp": time.time(),
        }

    def start_stream(
        self,
        on_sample: Callable[[Dict[str, float]], None],
        hz: float = 10.0,
    ) -> None:
        """
        Start a continuous EEG feature stream.

        on_sample: callback that receives a dict of features in [0, 1].
        hz:        sampling rate in Hz (mock only).
        """
        if not self._connected:
            raise RuntimeError(
                "NeurosityAdapter not connected. Call connect() first."
            )

        self._streaming = True

        if self.use_mock_stream:
            interval = 1.0 / max(1.0, hz)
            while self._streaming:
                on_sample(self._mock_sample())
                time.sleep(interval)
            return

        # TODO: real SDK stream
        # self._subscription = self._sdk.calm().subscribe(lambda data: ...)
        # self._sdk.focus().subscribe(lambda data: ...)
        # self._sdk.brainwaves("powerByBand").subscribe(lambda data: ...)
        #
        # Combine async signals into a unified dict:
        # {"focus": ..., "calm": ..., "gamma": ..., "timestamp": ...}

    def stop_stream(self) -> None:
        self._streaming = False
        if self._subscription is not None:
            # TODO: unsubscribe real SDK
            self._subscription = None


def merge_neurosity_signals(
    focus_value: Optional[float],
    calm_value: Optional[float],
    gamma_value: Optional[float],
) -> Dict[str, float]:
    """
    Helper to unify signals coming from separate Neurosity observables.

    All inputs are expected in [0, 1] (or None for missing values).
    """

    def safe(v: Optional[float]) -> float:
        if v is None:
            return 0.0
        return max(0.0, min(1.0, float(v)))

    return {
        "focus": safe(focus_value),
        "calm": safe(calm_value),
        "gamma": safe(gamma_value),
        "timestamp": time.time(),
    }
