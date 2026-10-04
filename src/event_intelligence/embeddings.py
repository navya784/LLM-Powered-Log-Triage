"""
Semantic embeddings (Phase 2: "Embeddings").

Two things are deliberately explicit here, because both change experimental
results:

1. WHICH TEXT is embedded (`representation`):
     "message"  -> normalized_message (variable tokens kept)
     "template" -> template           (variable tokens masked)
     "hybrid"   -> "<template> | <normalized_message>"
   None of them is assumed to be better. Template masking can remove useful
   signal (status codes, latencies, ids), so the choice is an experimental
   variable -- see `python -m src.evaluate_clustering`.

2. WHICH BACKEND produced the vectors (`backend`). The backend is never
   switched silently. If the requested backend is unavailable:
     * strict (default, use for research): raise EmbeddingBackendUnavailable
     * allow_fallback=True (demo/CI only): use TF-IDF+SVD and say so in the
       returned `EmbeddingInfo` (`fallback_from` is set).
   Every result carries `EmbeddingInfo` (backend, model, dimension,
   representation) so experiments can be told apart afterwards.

Identical texts are embedded once and the vector is shared (log templates
repeat heavily), which keeps large runs cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

from src import config
from src.preprocessing.schema import LogEvent

BACKEND_SENTENCE_TRANSFORMERS = "sentence-transformers"
BACKEND_TFIDF = "tfidf"


class EmbeddingBackendUnavailable(RuntimeError):
    """The requested embedding backend/model could not be used."""


@dataclass
class EmbeddingInfo:
    embedding_backend: str                 # "sentence-transformers" | "tfidf-svd"
    embedding_model: Optional[str]         # model name; None for TF-IDF
    embedding_dimension: int
    representation: str                    # "message" | "template" | "hybrid"
    normalized: bool = True
    n_texts: int = 0
    n_unique_texts: int = 0
    fallback_from: Optional[str] = None    # set only if a fallback was allowed and used
    backend_version: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Text selection
# ---------------------------------------------------------------------------

def event_text(event: LogEvent, representation: str) -> str:
    message = event.normalized_message if event.normalized_message is not None else event.message
    template = event.template if event.template is not None else message
    if representation == "message":
        return message
    if representation == "template":
        return template
    if representation == "hybrid":
        return f"{template} | {message}"
    raise ValueError(
        f"Unknown embedding representation '{representation}'. "
        f"Choose one of {list(config.EMBEDDING_REPRESENTATIONS)}."
    )


def build_texts(events: list[LogEvent], representation: str) -> list[str]:
    return [event_text(e, representation) for e in events]


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def _embed_with_sentence_transformers(texts: list[str], model_name: str) -> tuple[np.ndarray, str]:
    try:
        import sentence_transformers
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        embeddings = model.encode(
            texts, batch_size=config.EMBEDDING_BATCH_SIZE,
            show_progress_bar=False, normalize_embeddings=True,
        )
    except Exception as exc:  # ImportError, or the model cannot be loaded/downloaded
        raise EmbeddingBackendUnavailable(
            f"Embedding backend '{BACKEND_SENTENCE_TRANSFORMERS}' with model '{model_name}' "
            f"is unavailable ({type(exc).__name__}: {exc}). Install requirements "
            f"(sentence-transformers, torch) and make sure the model can be loaded, "
            f"or choose --embedding-backend tfidf explicitly."
        ) from exc
    return np.asarray(embeddings, dtype=np.float32), sentence_transformers.__version__


def _embed_with_tfidf(texts: list[str], n_components: int) -> np.ndarray:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_features=4096, ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform(texts)

    n_components = max(min(n_components, tfidf.shape[1] - 1, tfidf.shape[0] - 1), 1)
    svd = TruncatedSVD(n_components=n_components, random_state=config.RANDOM_SEED)
    reduced = svd.fit_transform(tfidf)

    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (reduced / norms).astype(np.float32)


def embed_texts(
    texts: list[str],
    backend: Optional[str] = None,
    model_name: Optional[str] = None,
    allow_fallback: bool = False,
    representation: str = "unspecified",
) -> tuple[np.ndarray, EmbeddingInfo]:
    """Returns an (N, D) float32 matrix (one row per input text, rows
    L2-normalized) and an `EmbeddingInfo` describing exactly how it was made."""
    backend = backend or config.EMBEDDING_BACKEND
    model_name = model_name or config.EMBEDDING_MODEL_NAME
    if backend not in (BACKEND_SENTENCE_TRANSFORMERS, BACKEND_TFIDF):
        raise ValueError(
            f"Unknown embedding backend '{backend}'. "
            f"Choose '{BACKEND_SENTENCE_TRANSFORMERS}' or '{BACKEND_TFIDF}'."
        )

    if not texts:
        info = EmbeddingInfo(
            embedding_backend=backend if backend == BACKEND_SENTENCE_TRANSFORMERS else "tfidf-svd",
            embedding_model=model_name if backend == BACKEND_SENTENCE_TRANSFORMERS else None,
            embedding_dimension=0, representation=representation,
        )
        return np.zeros((0, 0), dtype=np.float32), info

    unique_texts = list(dict.fromkeys(texts))
    index_of = {t: i for i, t in enumerate(unique_texts)}
    inverse = np.fromiter((index_of[t] for t in texts), dtype=np.int64, count=len(texts))

    fallback_from: Optional[str] = None
    version: Optional[str] = None
    if backend == BACKEND_SENTENCE_TRANSFORMERS:
        try:
            unique_matrix, version = _embed_with_sentence_transformers(unique_texts, model_name)
            used_backend, used_model = BACKEND_SENTENCE_TRANSFORMERS, model_name
        except EmbeddingBackendUnavailable:
            if not allow_fallback:
                raise
            print("[embeddings] WARNING: sentence-transformers unavailable; using the TF-IDF+SVD "
                  "fallback (recorded as fallback_from in the run metadata).")
            fallback_from = BACKEND_SENTENCE_TRANSFORMERS
            unique_matrix = _embed_with_tfidf(unique_texts, config.TFIDF_SVD_COMPONENTS)
            used_backend, used_model = "tfidf-svd", None
    else:
        unique_matrix = _embed_with_tfidf(unique_texts, config.TFIDF_SVD_COMPONENTS)
        used_backend, used_model = "tfidf-svd", None

    matrix = unique_matrix[inverse]
    info = EmbeddingInfo(
        embedding_backend=used_backend, embedding_model=used_model,
        embedding_dimension=int(matrix.shape[1]), representation=representation,
        n_texts=len(texts), n_unique_texts=len(unique_texts),
        fallback_from=fallback_from, backend_version=version,
    )
    return matrix, info


def embed_events(
    events: list[LogEvent],
    save_path: Optional[Path] = None,
    representation: Optional[str] = None,
    backend: Optional[str] = None,
    model_name: Optional[str] = None,
    allow_fallback: bool = False,
) -> tuple[np.ndarray, EmbeddingInfo]:
    """Embeds events (row i <-> events[i]) and sets `embedding_index`."""
    representation = representation or config.EMBEDDING_REPRESENTATION
    texts = build_texts(events, representation)
    matrix, info = embed_texts(
        texts, backend=backend, model_name=model_name,
        allow_fallback=allow_fallback, representation=representation,
    )
    if matrix.shape[0] != len(events):
        raise RuntimeError(
            f"Embedding matrix has {matrix.shape[0]} rows for {len(events)} events."
        )

    for i, event in enumerate(events):
        event.embedding_index = i

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, matrix)

    return matrix, info
