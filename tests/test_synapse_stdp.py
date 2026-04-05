import pytest


@pytest.fixture
def synapse_cls():
    from core.synapse_stdp import SynapseSTDP

    return SynapseSTDP


def _read_weight(synapse):
    for attr in ("weight", "w"):
        if hasattr(synapse, attr):
            return getattr(synapse, attr)
    raise AssertionError("SynapseSTDP must expose weight via `weight` or `w`.")


def _read_w_max(synapse):
    if hasattr(synapse, "W_MAX"):
        return getattr(synapse, "W_MAX")
    if hasattr(type(synapse), "W_MAX"):
        return getattr(type(synapse), "W_MAX")
    raise AssertionError("SynapseSTDP must expose `W_MAX`.")


def _apply_pair(synapse, pre_t_ms, post_t_ms):
    if hasattr(synapse, "update"):
        return synapse.update(pre_t_ms=pre_t_ms, post_t_ms=post_t_ms)
    if hasattr(synapse, "apply_stdp"):
        return synapse.apply_stdp(pre_t_ms=pre_t_ms, post_t_ms=post_t_ms)
    if hasattr(synapse, "on_spike_pair"):
        return synapse.on_spike_pair(pre_t_ms=pre_t_ms, post_t_ms=post_t_ms)
    raise AssertionError(
        "SynapseSTDP must provide one STDP method: "
        "`update`, `apply_stdp`, or `on_spike_pair`."
    )


def _build_synapse(synapse_cls, initial_weight=0.5):
    kwargs_candidates = (
        {"weight": initial_weight},
        {"w": initial_weight},
        {"initial_weight": initial_weight},
        {},
    )
    for kwargs in kwargs_candidates:
        try:
            syn = synapse_cls(**kwargs)
            if not kwargs:
                if hasattr(syn, "weight"):
                    setattr(syn, "weight", initial_weight)
                elif hasattr(syn, "w"):
                    setattr(syn, "w", initial_weight)
            return syn
        except TypeError:
            continue
    raise AssertionError("Unable to instantiate SynapseSTDP with known signatures.")


def test_ltp_pre_before_post_increases_weight(synapse_cls):
    syn = _build_synapse(synapse_cls, initial_weight=0.5)
    w_before = _read_weight(syn)

    # pre at t=10ms and post at t=15ms -> pre occurs 5ms before post
    _apply_pair(syn, pre_t_ms=10.0, post_t_ms=15.0)
    w_after = _read_weight(syn)

    assert w_after > w_before


def test_ltd_post_before_pre_decreases_weight(synapse_cls):
    syn = _build_synapse(synapse_cls, initial_weight=0.5)
    w_before = _read_weight(syn)

    # post at t=10ms and pre at t=15ms -> post occurs before pre
    _apply_pair(syn, pre_t_ms=15.0, post_t_ms=10.0)
    w_after = _read_weight(syn)

    assert w_after < w_before


def test_weight_is_clamped_to_w_max(synapse_cls):
    syn = _build_synapse(synapse_cls, initial_weight=0.99)
    w_max = _read_w_max(syn)

    # Repeat causal pairs to force potentiation pressure toward upper bound.
    for i in range(1000):
        _apply_pair(syn, pre_t_ms=float(i), post_t_ms=float(i) + 5.0)

    assert _read_weight(syn) <= w_max
