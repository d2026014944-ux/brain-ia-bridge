from core.gravity_engine import GravityEngine


class FakeNet:
    def __init__(self):
        self.calls: list[tuple[str, str, float, float]] = []

    def add_connection(self, pre_id, post_id, weight, delay_ms):
        self.calls.append((pre_id, post_id, float(weight), float(delay_ms)))


def test_chuva_molhado_has_high_weight_and_low_delay():
    engine = GravityEngine()
    net = FakeNet()

    result = engine.forge_geodesic(net, "Chuva", "Molhado")

    assert result["weight"] > 0.80
    assert result["delay_ms"] < 2.0
    assert result["connected"] is True

    assert len(net.calls) == 1
    _, _, weight, delay_ms = net.calls[0]
    assert weight == result["weight"]
    assert delay_ms == result["delay_ms"]


def test_chuva_fogo_has_near_zero_weight_and_high_delay():
    engine = GravityEngine()
    net = FakeNet()

    result = engine.forge_geodesic(net, "Chuva", "Fogo")

    assert result["weight"] < 0.01
    assert result["delay_ms"] > 7.0
    assert result["connected"] is True


def test_topological_antidote_zeroes_weight_after_critical_threshold():
    custom_vectors = {
        "nuvem": (1.0, 0.0, 0.0),
        "vulcao": (0.0, 1.0, 0.0),
    }
    engine = GravityEngine(vocab_vectors=custom_vectors, critical_distance=0.70)
    net = FakeNet()

    result = engine.forge_geodesic(net, "Nuvem", "Vulcao")

    assert result["distance"] > 0.70
    assert result["weight"] == 0.0
    assert result["connected"] is False
    assert result["delay_ms"] > engine.delay_min

    _, _, weight, _ = net.calls[0]
    assert weight == 0.0