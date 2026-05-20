from adapters.the_well_adapter import TheWellAdapter, TheWellConfig


def test_fetch_wisdom_maps_frequency_and_emotion_to_epistemic_terms():
    adapter = TheWellAdapter(seed=11, config=TheWellConfig(use_web=False, max_terms=6))

    concepts = adapter.fetch_wisdom(frequency_hz=14.2, emotion="duvida_epistemica")

    assert len(concepts) == 6
    assert "causalidade" in concepts
    assert "hipotese" in concepts
    assert "experimento" in concepts
    assert any(term in concepts for term in {"pergunta", "falsificacao", "criterio"})
