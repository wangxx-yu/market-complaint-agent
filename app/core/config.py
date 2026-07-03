from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel


class Settings(BaseModel):
    data_dir: Path = Path("data/runtime")
    max_text_chars: int = 2000
    low_confidence_threshold: float = 0.72
    default_office_code: str = "QTX_XIAOBA"
    default_office_name: str = "小坝市场监管所"
    llm_timeout_seconds: float = 2.0
    reject_reason_model_dir: Path = Path("models/reject_reason_v1")
    reject_reason_high_confidence_threshold: float = 0.70
    use_chroma_retrieval: bool = True
    embedding_provider: str = "hash"
    #切换为BGE
    # embedding_provider: str = "bge"
    sentence_transformer_model: str = "BAAI/bge-small-zh-v1.5"
    orchestrator_backend: str = "langgraph"
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen2.5:7b"

    @property
    def traces_path(self) -> Path:
        return self.data_dir / "traces.jsonl"

    @property
    def reviews_path(self) -> Path:
        return self.data_dir / "reviews.jsonl"

    @property
    def address_aliases_path(self) -> Path:
        return self.data_dir / "address_aliases.json"

    @property
    def dispatch_mapping_path(self) -> Path:
        return Path("data/dispatch/dispatch_mapping.json")

    @property
    def manual_dispatch_rules_path(self) -> Path:
        return Path("data/dispatch/manual_dispatch_rules.json")

    @property
    def reply_templates_path(self) -> Path:
        return self.data_dir / "reply_templates.json"

    @property
    def knowledge_base_path(self) -> Path:
        return Path("data/knowledge/rag_knowledge.json")

    @property
    def chroma_dir(self) -> Path:
        return Path("data/chroma")

    @property
    def model_cache_dir(self) -> Path:
        return Path("models/embeddings")


settings = Settings()
