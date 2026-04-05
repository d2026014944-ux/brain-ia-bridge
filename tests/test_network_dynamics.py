from realtime_loop import SlidingWindowStateFilter


def test_isolated_spikes_do_not_trigger_stable_firing():
    """
    Firing-rate guard: isolated spikes must not pass temporal consensus.
    """
    window_size = 8
    smoother = SlidingWindowStateFilter(window_size=window_size, min_votes=5)
    stream = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0]

    outputs = [smoother.push(state) for state in stream]

    # After warmup, no non-zero stable state should be emitted from sparse spikes.
    warmed_outputs = outputs[window_size - 1 :]
    assert all(state == 0 for state in warmed_outputs)


def test_network_returns_to_homeostasis_after_intent_burst():
    """
    Homeostasis: after sustained intent activation, neutral input should
    re-stabilize state at 0.
    """
    smoother = SlidingWindowStateFilter(window_size=5, min_votes=3)

    # Warmup + strong intent consensus.
    for state in [1, 1, 1, 1, 1]:
        last = smoother.push(state)
    assert last == 1

    # Neutral phase should drive the window back to stable neutral.
    neutral_outputs = [smoother.push(0) for _ in range(5)]
    assert 0 in neutral_outputs
    assert neutral_outputs[-1] == 0
