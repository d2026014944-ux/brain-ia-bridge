import importlib

import pytest


@pytest.fixture
def lif_neuron_cls():
    try:
        module = importlib.import_module("core.lif_neuron")
    except ModuleNotFoundError as exc:
        pytest.fail(
            "TDD Red: implemente core.lif_neuron.LIFNeuron para atender os "
            "testes de vazamento, disparo e periodo refratario. "
            f"Erro original: {exc}"
        )

    if not hasattr(module, "LIFNeuron"):
        pytest.fail(
            "TDD Red: modulo core.lif_neuron encontrado, mas sem classe "
            "LIFNeuron."
        )

    return module.LIFNeuron


def _build_neuron(lif_neuron_cls):
    return lif_neuron_cls(
        tau=20.0,
        v_thresh=1.0,
        refractory_period=5.0,
    )


def test_membrane_potential_leaks_back_to_rest(lif_neuron_cls):
    """
    Leaky: um spike isolado sobe V_m e, sem novos eventos, V_m decai para ~0.
    """
    neuron = _build_neuron(lif_neuron_cls)

    fired = neuron.receive_spike(current_t=0.0, weight=0.5)

    assert fired is False
    assert neuron.v_m > 0.0

    neuron.receive_spike(current_t=100.0, weight=0.0)

    assert neuron.v_m == pytest.approx(0.0, abs=5e-3)


def test_neuron_fires_when_threshold_is_crossed(lif_neuron_cls):
    """
    Fire: tres spikes rapidos (0, 2, 4 ms) com peso 0.5 devem cruzar limiar.
    """
    neuron = _build_neuron(lif_neuron_cls)

    s1 = neuron.receive_spike(current_t=0.0, weight=0.5)
    s2 = neuron.receive_spike(current_t=2.0, weight=0.5)
    s3 = neuron.receive_spike(current_t=4.0, weight=0.5)

    assert s1 is False
    assert s2 is False
    assert s3 is True
    assert neuron.v_m == pytest.approx(0.0, abs=1e-9)


def test_refractory_period_blocks_charge_accumulation(lif_neuron_cls):
    """
    Refratario: apos disparo em t=4 ms, spike em t=6 ms deve ser ignorado.
    """
    neuron = _build_neuron(lif_neuron_cls)

    neuron.receive_spike(current_t=0.0, weight=0.5)
    neuron.receive_spike(current_t=2.0, weight=0.5)
    did_fire = neuron.receive_spike(current_t=4.0, weight=0.5)

    assert did_fire is True
    assert neuron.v_m == pytest.approx(0.0, abs=1e-9)

    ignored_spike_fire = neuron.receive_spike(current_t=6.0, weight=0.5)

    assert ignored_spike_fire is False
    assert neuron.v_m == pytest.approx(0.0, abs=1e-9)
