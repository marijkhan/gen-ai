import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Batch encode texts and L2-normalize. Returns shape (N, 384)."""
    model = get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return (embeddings / norms).astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """Encode a single query, L2-normalize. Returns shape (1, 384)."""
    return embed_texts([query])
