import numpy as np
import pytest

from src import config
from src.event_intelligence import embeddings as emb
from src.event_intelligence.embeddings import (
    EmbeddingBackendUnavailable, build_texts, embed_events, embed_texts, event_text,
)
from src.event_intelligence.event_extractor import extract_event
from src.preprocessing.normalizer import normalize_event
from src.preprocessing.parsers import parse_line


def _events():
    lines = [
        "2026-09-20 10:30:00 [ERROR] api: failed to connect to db at 10.0.0.5: timeout after 300ms",
        "2026-09-20 10:30:05 [ERROR] api: failed to connect to db at 10.0.0.9: timeout after 900ms",
        "2026-09-20 10:30:09 [INFO] api: user 1234 logged in successfully",
        "2026-09-20 10:30:10 [INFO] api: health check passed",
    ]
    return [extract_event(normalize_event(parse_line("f.log", i, l))) for i, l in enumerate(lines, 1)]


def _no_sentence_transformers(monkeypatch):
    def unavailable(texts, model_name):
        raise EmbeddingBackendUnavailable("simulated: model not installed")
    monkeypatch.setattr(emb, "_embed_with_sentence_transformers", unavailable)


def test_three_representations_are_selectable_and_differ():
    e = _events()[0]
    assert event_text(e, "message") == e.normalized_message
    assert event_text(e, "template") == e.template
    assert event_text(e, "hybrid") == f"{e.template} | {e.normalized_message}"
    assert len({event_text(e, r) for r in config.EMBEDDING_REPRESENTATIONS}) == 3


def test_unknown_representation_is_rejected():
    with pytest.raises(ValueError, match="representation"):
        build_texts(_events(), "vibes")


def test_embed_events_records_backend_model_dimension_representation():
    events = _events()
    matrix, info = embed_events(events, representation="hybrid", backend="tfidf")
    d = info.to_dict()
    assert d["embedding_backend"] == "tfidf-svd" and d["embedding_model"] is None
    assert d["embedding_dimension"] == matrix.shape[1] > 0
    assert d["representation"] == "hybrid" and d["fallback_from"] is None
    assert matrix.shape[0] == len(events)
    assert [e.embedding_index for e in events] == [0, 1, 2, 3]


def test_identical_texts_share_a_vector_and_alignment_is_kept():
    events = _events()
    matrix, info = embed_events(events, representation="template", backend="tfidf")
    assert info.n_texts == 4 and info.n_unique_texts == 3     # the two timeout lines share a template
    assert np.array_equal(matrix[0], matrix[1])
    assert not np.array_equal(matrix[0], matrix[2])


def test_rows_are_l2_normalized():
    matrix, _ = embed_events(_events(), backend="tfidf")
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5)


def test_strict_mode_fails_instead_of_switching_backend(monkeypatch):
    _no_sentence_transformers(monkeypatch)
    with pytest.raises(EmbeddingBackendUnavailable, match="simulated"):
        embed_events(_events(), backend="sentence-transformers")          # allow_fallback defaults to False


def test_allowed_fallback_is_recorded_not_hidden(monkeypatch, capsys):
    _no_sentence_transformers(monkeypatch)
    matrix, info = embed_events(_events(), backend="sentence-transformers", allow_fallback=True)
    assert info.embedding_backend == "tfidf-svd"
    assert info.fallback_from == "sentence-transformers"
    assert "WARNING" in capsys.readouterr().out
    assert matrix.shape[0] == 4


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="backend"):
        embed_texts(["a", "b"], backend="word2vec")


def test_empty_input_returns_empty_matrix():
    matrix, info = embed_texts([], backend="tfidf")
    assert matrix.shape[0] == 0 and info.embedding_dimension == 0
