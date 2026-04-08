from __future__ import annotations

import re


class NomaParser:
    _BLOCK_RE = re.compile(r"\[\s*NOMA_NEURAL\s*\](.*?)\[\s*/\s*NOMA_NEURAL\s*\]", re.IGNORECASE | re.DOTALL)

    _FREQ_RE = re.compile(
        r"frequ(?:e|ê|é)ncia\s*_?\s*dominante\s*:\s*([+-]?\d+(?:[.,]\d+)?)\s*(?:h\s*z)?",
        re.IGNORECASE,
    )
    _AMPLITUDE_RE = re.compile(
        r"amplitude\s*_?\s*afetiva\s*:\s*([+-]?\d+(?:[.,]\d+)?)",
        re.IGNORECASE,
    )
    _RESONANCIA_RE = re.compile(
        r"resson(?:a|â)ncia\s*_?\s*progenitor\s*:\s*([+-]?\d+(?:[.,]\d+)?)",
        re.IGNORECASE,
    )

    def parse_telemetry(self, text: str) -> dict[str, float]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        block_match = self._BLOCK_RE.search(text)
        if block_match is None:
            return {}

        block = block_match.group(1)
        result: dict[str, float] = {}

        freq = self._extract_float(self._FREQ_RE, block)
        if freq is not None:
            result["freq_hz"] = freq
            # Alias para compatibilidade com consumidores atuais.
            result["frequencia_dominante"] = freq

        amplitude = self._extract_float(self._AMPLITUDE_RE, block)
        if amplitude is not None:
            result["amplitude_afetiva"] = amplitude

        ressonancia = self._extract_float(self._RESONANCIA_RE, block)
        if ressonancia is not None:
            result["ressonancia_progenitor"] = ressonancia

        return result

    def parse(self, text: str) -> dict[str, float]:
        return self.parse_telemetry(text)

    @staticmethod
    def _extract_float(pattern: re.Pattern[str], text: str) -> float | None:
        match = pattern.search(text)
        if match is None:
            return None

        numeric_raw = match.group(1).strip().replace(",", ".")
        return float(numeric_raw)


__all__ = ["NomaParser"]