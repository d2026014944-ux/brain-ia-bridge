from __future__ import annotations

import math
import unicodedata
from typing import Mapping, Sequence


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _normalize_text(text: str) -> str:
    return _strip_accents(text).strip().lower()


def _l2_norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    norm_a = _l2_norm(vec_a)
    norm_b = _l2_norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    similarity = dot / (norm_a * norm_b)
    return max(-1.0, min(1.0, similarity))


class GravityEngine:
    """
    Calcula uma geodésica semântica entre conceitos e forja conexões na rede.

    Distância semântica:
        D = 1 - cos_sim(embedding_a, embedding_b)

    Derivação:
        weight = w_max * exp(-alpha * D)
        delay_ms = delay_min + (beta * D)

    Antídoto Topológico:
        Se D ultrapassar critical_distance, o peso é forçado para 0.
    """

    DEFAULT_VOCAB_VECTORS: dict[str, tuple[float, ...]] = {
        "chuva": (1.0, 1.0, 0.0, 0.15),
        "molhado": (0.9, 1.0, 0.0, 0.0),
        "agua": (0.95, 1.0, 0.0, 0.0),
        "fogo": (0.0, 0.0, 0.1, 1.0),
        "seca": (0.0, 0.0, 1.0, 0.3),
        "calor": (0.0, 0.0, 0.2, 0.95),
    }

    def __init__(
        self,
        w_max: float = 1.0,
        alpha: float = 8.0,
        beta: float = 8.0,
        delay_min: float = 1.0,
        critical_distance: float = 0.95,
        vocab_vectors: Mapping[str, Sequence[float]] | None = None,
    ) -> None:
        if w_max < 0.0:
            raise ValueError("w_max must be >= 0")
        if alpha < 0.0:
            raise ValueError("alpha must be >= 0")
        if beta < 0.0:
            raise ValueError("beta must be >= 0")
        if delay_min < 0.0:
            raise ValueError("delay_min must be >= 0")
        if critical_distance < 0.0:
            raise ValueError("critical_distance must be >= 0")

        self.w_max = float(w_max)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.delay_min = float(delay_min)
        self.critical_distance = float(critical_distance)

        source_vectors = vocab_vectors or self.DEFAULT_VOCAB_VECTORS
        self.vocab_vectors: dict[str, tuple[float, ...]] = {
            _normalize_text(key): tuple(float(value) for value in vector)
            for key, vector in source_vectors.items()
        }

        self._vector_size = self._infer_vector_size(self.vocab_vectors)

    @staticmethod
    def _infer_vector_size(vocab_vectors: Mapping[str, Sequence[float]]) -> int:
        sizes = {len(vector) for vector in vocab_vectors.values()}
        if not sizes:
            return 8
        if len(sizes) != 1:
            raise ValueError("All vocab vectors must have the same dimensionality")
        return next(iter(sizes))

    def _hash_embedding(self, concept: str) -> list[float]:
        """Fallback local embedding para manter o motor totalmente offline."""
        text = _normalize_text(concept)
        if not text:
            return [0.0] * self._vector_size

        vec = [0.0] * self._vector_size
        padded = f" {text} "
        for idx in range(len(padded) - 1):
            bigram = padded[idx : idx + 2]
            bucket = (ord(bigram[0]) + (31 * ord(bigram[1]))) % self._vector_size
            vec[bucket] += 1.0

        norm = _l2_norm(vec)
        if norm == 0.0:
            return vec
        return [value / norm for value in vec]

    def _embed(self, concept: str) -> tuple[float, ...]:
        concept_norm = _normalize_text(concept)
        if concept_norm in self.vocab_vectors:
            return self.vocab_vectors[concept_norm]

        tokens = [token for token in concept_norm.split() if token]
        known_vectors = [self.vocab_vectors[token] for token in tokens if token in self.vocab_vectors]
        if known_vectors:
            mean = [0.0] * self._vector_size
            for vector in known_vectors:
                for idx, value in enumerate(vector):
                    mean[idx] += value

            scale = 1.0 / len(known_vectors)
            return tuple(component * scale for component in mean)

        return tuple(self._hash_embedding(concept_norm))

    def semantic_distance(self, concept_a: str, concept_b: str) -> float:
        if not isinstance(concept_a, str) or not isinstance(concept_b, str):
            raise TypeError("concept_a and concept_b must be strings")

        vec_a = self._embed(concept_a)
        vec_b = self._embed(concept_b)

        similarity = _cosine_similarity(vec_a, vec_b)
        distance = 1.0 - similarity
        return max(0.0, min(2.0, distance))

    def derive_weight_and_delay(self, distance: float) -> tuple[float, float]:
        if distance < 0.0:
            raise ValueError("distance must be >= 0")

        delay_ms = self.delay_min + (self.beta * distance)
        if distance > self.critical_distance:
            return 0.0, delay_ms

        weight = self.w_max * math.exp(-self.alpha * distance)
        return weight, delay_ms

    @staticmethod
    def concept_to_id(concept: str) -> str:
        return _normalize_text(concept)

    def forge_geodesic(self, net, concept_a: str, concept_b: str) -> dict[str, float | str | bool]:
        if not hasattr(net, "add_connection"):
            raise TypeError("net must provide add_connection(pre_id, post_id, weight, delay_ms)")

        id_a = self.concept_to_id(concept_a)
        id_b = self.concept_to_id(concept_b)
        distance = self.semantic_distance(concept_a, concept_b)
        weight, delay_ms = self.derive_weight_and_delay(distance)

        net.add_connection(id_a, id_b, weight, delay_ms)
        return {
            "id_a": id_a,
            "id_b": id_b,
            "distance": distance,
            "weight": weight,
            "delay_ms": delay_ms,
            "connected": weight > 0.0,
        }


__all__ = ["GravityEngine"]