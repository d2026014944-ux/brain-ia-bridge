import pytest

from mind_panel import _decode_thought, _decode_thought_with_timeout


def _cycle_payload() -> dict:
    return {
        "nodes": [
            {"id": 0, "name": "node_0", "x": 0.0, "y": 0.0},
            {"id": 1, "name": "node_1", "x": 1.0, "y": 0.0},
        ],
        "node_names": {
            "0": "node_0",
            "1": "node_1",
        },
        "synapses": [
            {"pre_id": 0, "post_id": 1, "weight": 1.0, "delay_ms": 1.0},
            {"pre_id": 1, "post_id": 0, "weight": 1.0, "delay_ms": 1.0},
        ],
    }


def test_decode_thought_terminates_on_strong_cycle() -> None:
    payload = _cycle_payload()

    trace = _decode_thought(payload, "node_0")

    assert trace["concept_chain"].startswith("node_0")
    assert len(trace["sequence"]) <= 2
    assert len({step["node_id"] for step in trace["sequence"]}) == len(trace["sequence"])


def test_decode_thought_with_timeout_completes_on_valid_payload() -> None:
    payload = _cycle_payload()

    trace = _decode_thought_with_timeout(payload, "node_0", timeout_s=1.0)

    assert trace["concept_chain"].startswith("node_0")


def test_decode_thought_with_timeout_raises_on_zero_timeout() -> None:
    payload = _cycle_payload()

    with pytest.raises(TimeoutError):
        _decode_thought_with_timeout(payload, "node_0", timeout_s=0.0)
