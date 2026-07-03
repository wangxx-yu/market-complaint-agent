from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from app.core.config import settings


class HashingEmbeddingFunction:
    """Offline embedding function compatible with Chroma.

    It uses character n-grams hashed into a fixed vector. This is not as strong as
    bge-small-zh, but it is local, deterministic, and has no model download step.
    """

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def name(self) -> str:
        return "local_hashing_embedding"

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in input]

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self(input)

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in self._tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    @staticmethod
    def _tokens(text: str) -> list[str]:
        text = re.sub(r"\s+", "", text or "")
        tokens: list[str] = []
        for size in (2, 3, 4):
            if len(text) >= size:
                tokens.extend(text[index : index + size] for index in range(len(text) - size + 1))
        tokens.extend(re.findall(r"[A-Za-z0-9]+", text.lower()))
        return tokens


class SentenceTransformerEmbeddingFunction:
    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or settings.sentence_transformer_model
        settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = SentenceTransformer(
            self.model_name,
            cache_folder=str(settings.model_cache_dir),
            device="cpu",
        )

    def name(self) -> str:
        return f"sentence_transformer_{self.model_name.replace('/', '_')}"

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.embed_documents(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self.embed_documents(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            input,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()


def create_embedding_function() -> Any:
    provider = settings.embedding_provider.lower()
    if provider in {"sentence-transformer", "sentence_transformer", "bge"}:
        return SentenceTransformerEmbeddingFunction()
    if provider == "hash":
        return HashingEmbeddingFunction()
    try:
        return SentenceTransformerEmbeddingFunction()
    except Exception:
        return HashingEmbeddingFunction()
