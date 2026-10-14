"""""
Text -> vector embeddings, used for semantic search (see retrieval.py).

The model is loaded lazily (only on first use, not at import time) so
that starting the server doesn't require downloading a ~90MB model
file up front - it downloads once, on the first chat/document request,
and is cached in memory after that.
"""
import numpy as np; np.seterr(all='raise')

from app.core.config import settings

_model = None


def validate_embedding(vector: list[float], expected_dimension: int | None = None) -> list[float]:
    values = [float(value) for value in vector]
    if not values or not np.isfinite(values).all():
        raise ValueError("Embedding must be a finite, non-empty vector")
    if expected_dimension is not None:
        try:
            expected_dimension = int(expected_dimension)
        except (TypeError, ValueError):
            raise ValueError("Embedding expected dimension must be an integer or None")
        if len(values) != expected_dimension:
            raise ValueError(
                f"Embedding has an unexpected dimension: expected {expected_dimension}, got {len(values)}"
            )
    return values


def model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        try:
            _model = SentenceTransformer(settings.embedding_model)
        except Exception as e:
            raise RuntimeError(
                f"Could not load embedding model '{settings.embedding_model}'. "
                f"Check your internet connection (first run downloads the model) "
                f"and that the model name is correct. Original error: {e}"
            )
    return _model


def encode(texts: list[str]) -> list[list[float]]:
    # normalize_embeddings=True scales every vector to length 1. This is
    # what lets cosine() below use a plain dot product instead of the
    # full cosine similarity formula (dot product / (|a| * |b|)) -
    # cheaper to compute, same result, once both vectors are unit length.
    # convert_to_numpy=True keeps the legacy behavior across sentence-transformers
    # 5.x/6.x where encode() can otherwise return a tensor-like object.
    vectors = model().encode(texts, normalize_embeddings=True, convert_to_numpy=True).tolist()
    if not vectors:
        return []
    dimension = len(vectors[0])
    return [validate_embedding(vector, dimension) for vector in vectors]


def cosine(a: list[float], b: list[float]) -> float:
    left = validate_embedding(a)
    right = validate_embedding(b, len(left))
    return float(np.dot(np.asarray(left), np.asarray(right)))
