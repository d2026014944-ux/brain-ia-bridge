from __future__ import annotations

import json
import random
import re
import unicodedata
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class TheWellConfig:
    timeout_s: float = 1.8
    max_terms: int = 6
    use_web: bool = True


class TheWellAdapter:
    """
    The Well: epistemic adapter that maps neural state to external concepts.

    The adapter is resilient by design:
    - Uses local concept reservoirs per frequency band and emotion.
    - Optionally enriches concepts from a public endpoint (Wikipedia API).
    - Fails gracefully to offline mode when network is unavailable.
    """

    _BAND_SEEDS = {
        "delta": ["sono", "homeostase", "ritmo_circadiano", "restauracao"],
        "theta": ["intuicao", "analogia", "narrativa", "associacao"],
        "alpha": ["atencao", "equilibrio", "respiracao", "regulacao"],
        "beta": ["causalidade", "hipotese", "evidencia", "experimento"],
        "gamma": ["sintese", "complexidade", "integracao", "emergencia"],
    }

    _EMOTION_SEEDS = {
        "duvida_epistemica": ["pergunta", "falsificacao", "criterio"],
        "curiosidade": ["exploracao", "descoberta", "mapa"],
        "foco_elevado": ["metodo", "disciplina", "iteracao"],
        "euforia_sincronica": ["criatividade", "invencao", "salto"],
        "calma": ["observacao", "escuta", "paciencia"],
    }

    _GLOBAL_FALLBACK = ["aprendizado", "contexto", "modelo_mental", "revisao"]

    def __init__(self, seed: int | None = None, config: TheWellConfig | None = None) -> None:
        self._rng = random.Random(seed)
        self.config = config or TheWellConfig()

    def fetch_wisdom(self, frequency_hz: float, emotion: str) -> list[str]:
        freq = max(0.0, float(frequency_hz))
        emotion_key = _normalize_token(emotion)
        band = self._frequency_band(freq)

        local_terms = self._local_concepts(band=band, emotion=emotion_key)
        web_terms: list[str] = []

        if self.config.use_web:
            query = " ".join(local_terms[:3])
            web_terms = self._fetch_wikipedia_terms(query)

        merged = _dedupe_ordered(local_terms + web_terms + list(self._GLOBAL_FALLBACK))
        return merged[: max(1, int(self.config.max_terms))]

    @staticmethod
    def _frequency_band(frequency_hz: float) -> str:
        if frequency_hz < 4.0:
            return "delta"
        if frequency_hz < 8.0:
            return "theta"
        if frequency_hz < 12.0:
            return "alpha"
        if frequency_hz < 30.0:
            return "beta"
        return "gamma"

    def _local_concepts(self, band: str, emotion: str) -> list[str]:
        band_terms = list(self._BAND_SEEDS.get(band, []))
        emotion_terms = list(self._EMOTION_SEEDS.get(emotion, []))

        if not emotion_terms:
            fuzzy = [term for key, terms in self._EMOTION_SEEDS.items() if emotion and emotion in key for term in terms]
            emotion_terms = fuzzy[:3]

        pool = _dedupe_ordered(band_terms + emotion_terms)
        if len(pool) > 6:
            head = pool[:4]
            tail = pool[4:]
            self._rng.shuffle(tail)
            pool = head + tail

        return pool

    def _fetch_wikipedia_terms(self, query: str) -> list[str]:
        if not query:
            return []

        safe_query = quote_plus(query)
        url = (
            "https://pt.wikipedia.org/w/api.php"
            f"?action=opensearch&search={safe_query}&limit=6&namespace=0&format=json"
        )
        req = Request(url, headers={"User-Agent": "brain-ia-bridge/1.0 (TheWellAdapter)"})

        try:
            with urlopen(req, timeout=max(0.2, float(self.config.timeout_s))) as response:
                raw = response.read().decode("utf-8", errors="ignore")
        except (HTTPError, URLError, TimeoutError, OSError):
            return []

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return []

        if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
            return []

        terms: list[str] = []
        for item in payload[1]:
            token = _normalize_token(str(item))
            if token:
                terms.append(token)
        return _dedupe_ordered(terms)


def _normalize_token(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_\-]", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _dedupe_ordered(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = _normalize_token(item)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


__all__ = ["TheWellAdapter", "TheWellConfig"]
