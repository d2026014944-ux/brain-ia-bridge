import time

import pytest


TERNARY = {-1.0, 0.0, 1.0}


def _read_weight(synapse):
    for attr in ("weight", "w"):
        if hasattr(synapse, attr):
            return float(getattr(synapse, attr))
    raise AssertionError("SynapseSTDP must expose weight via `weight` or `w`.")


def _apply_pair(synapse, pre_t_ms, post_t_ms):
    if hasattr(synapse, "update"):
        return synapse.update(pre_t_ms=pre_t_ms, post_t_ms=post_t_ms)
    if hasattr(synapse, "apply_stdp"):
        return synapse.apply_stdp(pre_t_ms=pre_t_ms, post_t_ms=post_t_ms)
    if hasattr(synapse, "on_spike_pair"):
        return synapse.on_spike_pair(pre_t_ms=pre_t_ms, post_t_ms=post_t_ms)
    raise AssertionError(
        "SynapseSTDP must provide one STDP method: `update`, `apply_stdp`, or `on_spike_pair`."
    )


def _build_float_synapse(initial_weight=0.5):
    from core.synapse_stdp import SynapseSTDP

    kwargs_candidates = (
        {"weight": initial_weight, "quantization": "float32"},
        {"weight": initial_weight, "bitnet": False},
        {"weight": initial_weight},
        {"w": initial_weight},
        {},
    )
    for kwargs in kwargs_candidates:
        try:
            syn = SynapseSTDP(**kwargs)
            if not kwargs:
                if hasattr(syn, "weight"):
                    syn.weight = float(initial_weight)
                elif hasattr(syn, "w"):
                    syn.w = float(initial_weight)
            return syn
        except TypeError:
            continue
    raise AssertionError("Unable to instantiate float SynapseSTDP with known signatures.")


def _build_bitnet_synapse(initial_weight):
    from core import synapse_stdp as stdp_module

    class_candidates = []
    if hasattr(stdp_module, "SynapseSTDPBitNet"):
        class_candidates.append(getattr(stdp_module, "SynapseSTDPBitNet"))
    if hasattr(stdp_module, "BitNetSynapseSTDP"):
        class_candidates.append(getattr(stdp_module, "BitNetSynapseSTDP"))
    if hasattr(stdp_module, "SynapseSTDP"):
        class_candidates.append(getattr(stdp_module, "SynapseSTDP"))

    kwargs_candidates = (
        {"weight": initial_weight, "quantization": "bitnet_1_58"},
        {"weight": initial_weight, "weight_mode": "ternary"},
        {"weight": initial_weight, "bitnet": True},
        {"weight": initial_weight, "ternary": True},
        {"w": initial_weight, "quantization": "bitnet_1_58"},
    )

    for cls in class_candidates:
        for kwargs in kwargs_candidates:
            try:
                syn = cls(**kwargs)
            except TypeError:
                continue

            # If implementation exposes explicit mode flags, ensure BitNet mode is active.
            if hasattr(syn, "is_bitnet") and not bool(getattr(syn, "is_bitnet")):
                continue
            if hasattr(syn, "quantization"):
                mode = str(getattr(syn, "quantization")).lower()
                if "bitnet" not in mode and "ternary" not in mode:
                    continue
            return syn

    pytest.fail(
        "RED: BitNet 1.58bit not available. Expected SynapseSTDP with quantization='bitnet_1_58' "
        "or a dedicated BitNet synapse class."
    )


def _assert_ternary_weight(synapse):
    w = _read_weight(synapse)
    assert w in TERNARY, f"BitNet weight must be ternary (-1, 0, 1). Got {w}."


def _run_many_pairs(synapse, n_pairs, pre_before_post):
    if hasattr(synapse, "propagate_many"):
        start = time.perf_counter()
        synapse.propagate_many(
            n_pairs=n_pairs,
            dt_pre_post_ms=1.0 if pre_before_post else -1.0,
        )
        return time.perf_counter() - start

    start = time.perf_counter()
    t = 0.0
    for _ in range(n_pairs):
        if pre_before_post:
            _apply_pair(synapse, pre_t_ms=t, post_t_ms=t + 1.0)
        else:
            _apply_pair(synapse, pre_t_ms=t + 1.0, post_t_ms=t)
        t += 2.0
    return time.perf_counter() - start


def test_setup_uses_only_ternary_weights():
    for initial in (-1.0, 0.0, 1.0):
        syn = _build_bitnet_synapse(initial_weight=initial)
        _assert_ternary_weight(syn)

    # Invalid initialization must be quantized to ternary or rejected.
    try:
        syn = _build_bitnet_synapse(initial_weight=0.5)
    except (TypeError, ValueError):
        return
    _assert_ternary_weight(syn)


def test_stdp_evolves_in_discrete_jumps_without_intermediate_values():
    # Potentiation path: 0 -> 1
    syn_ltp = _build_bitnet_synapse(initial_weight=0.0)
    ltp_history = [_read_weight(syn_ltp)]
    for i in range(16):
        _apply_pair(syn_ltp, pre_t_ms=float(i) * 10.0, post_t_ms=float(i) * 10.0 + 1.0)
        ltp_history.append(_read_weight(syn_ltp))
        _assert_ternary_weight(syn_ltp)
        if _read_weight(syn_ltp) == 1.0:
            break

    assert ltp_history[0] == 0.0
    assert ltp_history[-1] == 1.0, f"Expected potentiation jump 0 -> 1. History: {ltp_history}"
    assert all(w in {0.0, 1.0} for w in ltp_history), f"No intermediate values allowed. History: {ltp_history}"

    # Depression path: 1 -> 0
    syn_ltd = _build_bitnet_synapse(initial_weight=1.0)
    ltd_history = [_read_weight(syn_ltd)]
    for i in range(16):
        _apply_pair(syn_ltd, pre_t_ms=float(i) * 10.0 + 1.0, post_t_ms=float(i) * 10.0)
        ltd_history.append(_read_weight(syn_ltd))
        _assert_ternary_weight(syn_ltd)
        if _read_weight(syn_ltd) == 0.0:
            break

    assert ltd_history[0] == 1.0
    assert ltd_history[-1] == 0.0, f"Expected depression jump 1 -> 0. History: {ltd_history}"
    assert all(w in {1.0, 0.0} for w in ltd_history), f"No intermediate values allowed. History: {ltd_history}"


def test_bitnet_is_at_least_twice_as_fast_as_float_for_one_million_propagations():
    n_pairs = 1_000_000

    bitnet = _build_bitnet_synapse(initial_weight=0.0)
    float_syn = _build_float_synapse(initial_weight=0.0)

    # Warm-up to reduce first-call noise.
    _run_many_pairs(bitnet, n_pairs=1_000, pre_before_post=True)
    _run_many_pairs(float_syn, n_pairs=1_000, pre_before_post=True)

    bitnet_seconds = _run_many_pairs(bitnet, n_pairs=n_pairs, pre_before_post=True)
    float_seconds = _run_many_pairs(float_syn, n_pairs=n_pairs, pre_before_post=True)

    assert bitnet_seconds * 2.0 <= float_seconds, (
        "BitNet 1.58bit C++ path must be >=2x faster than float path for 1,000,000 propagations. "
        f"bitnet={bitnet_seconds:.6f}s float={float_seconds:.6f}s"
    )