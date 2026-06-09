"""Text -> vector. Uses sentence-transformers if installed (the real path on your
VM); otherwise a deterministic hashing fallback so the harness logic runs and
self-tests anywhere. The fallback is NOT semantically meaningful -- it only lets
the pipeline execute; the real verdict requires real embeddings.
"""
from __future__ import annotations
import hashlib
import numpy as np


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 dim: int = 384):
        self.dim = dim
        self._st = None
        try:
            from sentence_transformers import SentenceTransformer
            self._st = SentenceTransformer(model_name)
            self.dim = self._st.get_sentence_embedding_dimension()
            self.real = True
        except Exception:
            self.real = False  # fallback

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._st is not None:
            v = np.asarray(self._st.encode(texts, normalize_embeddings=True),
                           dtype=np.float32)
            return v
        # deterministic hashing fallback -> unit vectors
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = hashlib.sha256(t.encode()).digest()
            rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
            x = rng.standard_normal(self.dim).astype(np.float32)
            out[i] = x / (np.linalg.norm(x) + 1e-8)
        return out


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))
