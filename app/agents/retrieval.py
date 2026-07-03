from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.agents.embeddings import create_embedding_function
from app.core.config import settings
from app.core.enums import ReasonType
from app.core.schemas import RetrievalHit


class RetrievalAgent:
    """Local lightweight RAG retriever.

    首版先不用向量库：从 JSON 知识库读取法规和权责片段，用原因类型、
    关键词和简单文本重合度打分。后续可以把 _score_entry 替换成向量相似度。
    """

    DEFAULT_KB: list[dict[str, Any]] = [
        {
            "id": "article16",
            "title": "市场监督管理投诉举报处理办法 第十六条",
            "content": (
                "第十六条规定了不予受理投诉的情形，包括不属于市场监督管理部门职责或者本机关不具有处理权限、"
                "同一消费争议已被受理或者处理、非生活消费争议、超过三年投诉时效、材料缺失或虚假、"
                "冒用他人名义或者拒不配合身份核验，以及法律法规规章规定的其他情形。"
            ),
            "source": "法规知识库",
            "reason_types": [reason.value for reason in ReasonType if reason != ReasonType.UNKNOWN],
            "keywords": ["不予受理", "第十六条", "处理权限", "职责", "材料", "三年"],
        },
        {
            "id": "out_scope_general",
            "title": "职责外事项转办原则",
            "content": "涉及其他主管部门职责的事项，应建议投诉人向相应主管部门或者属地政府反映。",
            "source": "权责清单",
            "reason_types": [ReasonType.ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY.value],
            "keywords": ["不属于职责", "职责外", "主管部门", "转办"],
        },
        {
            "id": "missing_materials",
            "title": "材料缺失或虚假处理原则",
            "content": "投诉对象、具体诉求、事实理由、交易凭证等必要信息不明确，或者材料存在真实性疑问的，应提示投诉人补充真实、完整材料后再处理。",
            "source": "历史模板",
            "reason_types": [ReasonType.ARTICLE16_5_MISSING_OR_FALSE_MATERIALS.value],
            "keywords": ["主体不明", "商家不详", "地址不详", "无消费凭证", "没有凭证", "材料不全"],
        },
    ]

    def __init__(self, knowledge_base_path: Path | None = None) -> None:
        self.knowledge_base_path = knowledge_base_path or settings.knowledge_base_path
        self._entries: list[dict[str, Any]] | None = None
        self._chroma_collection: Any | None = None
        self._chroma_failed = False
        self.last_status: dict[str, Any] = {
            "retrieval_source": "UNUSED",
            "degraded": False,
            "fallback_reason": None,
            "hit_count": 0,
        }

    @property
    def entries(self) -> list[dict[str, Any]]:
        if self._entries is None:
            self._entries = self._load_entries()
        return self._entries

    def retrieve(self, reason_type: ReasonType, query: str, top_k: int = 3) -> list[RetrievalHit]:
        chroma_hits = self._retrieve_with_chroma(reason_type, query, top_k)
        if chroma_hits:
            self.last_status = {
                "retrieval_source": "CHROMA",
                "degraded": False,
                "fallback_reason": None,
                "hit_count": len(chroma_hits),
            }
            return chroma_hits
        rule_hits = self._retrieve_with_rules(reason_type, query, top_k)
        fallback_reason = "Chroma 不可用或未返回结果，已回退关键词检索"
        if not settings.use_chroma_retrieval:
            fallback_reason = "Chroma 检索未启用，使用关键词检索"
        elif self._chroma_failed:
            fallback_reason = "Chroma 初始化或查询失败，已回退关键词检索"
        self.last_status = {
            "retrieval_source": "RULE_FALLBACK",
            "degraded": bool(settings.use_chroma_retrieval),
            "fallback_reason": fallback_reason,
            "hit_count": len(rule_hits),
        }
        return rule_hits

    def _retrieve_with_rules(self, reason_type: ReasonType, query: str, top_k: int = 3) -> list[RetrievalHit]:
        scored_entries = []
        for entry in self.entries:
            score = self._score_entry(entry, reason_type, query)
            if score <= 0:
                continue
            scored_entries.append((score, entry))

        scored_entries.sort(key=lambda item: item[0], reverse=True)
        if scored_entries and scored_entries[0][0] >= 0.55:
            scored_entries = [item for item in scored_entries if item[0] >= 0.55]
        return [
            RetrievalHit(
                knowledge_id=str(entry.get("id", "")) or None,
                title=str(entry.get("title", "")),
                content=str(entry.get("content", "")),
                score=min(score, 1.0),
                source=str(entry.get("source", "知识库")),
                source_url=entry.get("source_url"),
                law_status=entry.get("law_status"),
                suggested_department=entry.get("suggested_department"),
                explanation=self._explain_hit(entry),
            )
            for score, entry in scored_entries[:top_k]
        ]

    def _retrieve_with_chroma(self, reason_type: ReasonType, query: str, top_k: int) -> list[RetrievalHit]:
        collection = self._get_chroma_collection()
        if collection is None:
            return []
        try:
            where = None if reason_type == ReasonType.UNKNOWN else {"reason_types": {"$contains": reason_type.value}}
            result = collection.query(
                query_texts=[query],
                n_results=max(top_k * 4, 12),
                where=where,
                include=["metadatas", "documents", "distances"],
            )
        except Exception:
            self._chroma_failed = True
            self.last_status = {
                "retrieval_source": "CHROMA_ERROR",
                "degraded": True,
                "fallback_reason": "Chroma 查询异常",
                "hit_count": 0,
            }
            return []

        ids = result.get("ids", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        if not ids:
            return []

        entry_by_id = {str(entry.get("id")): entry for entry in self.entries}
        scored_entries = []
        for index, entry_id in enumerate(ids):
            entry = entry_by_id.get(str(entry_id))
            if not entry:
                continue
            vector_score = max(0.0, 1.0 - float(distances[index])) if index < len(distances) else 0.0
            rule_score = self._score_entry(entry, reason_type, query)
            score = min(1.0, max(vector_score, 0.35) * 0.55 + rule_score * 0.45)
            if self._has_query_keyword(entry, query):
                score += 0.12
            if entry.get("suggested_department") and self._has_query_keyword(entry, query):
                score += 0.18
            score = min(1.0, score)
            scored_entries.append((score, entry))

        if not scored_entries:
            return []
        scored_entries.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievalHit(
                knowledge_id=str(entry.get("id", "")) or None,
                title=str(entry.get("title", "")),
                content=str(entry.get("content", "")),
                score=round(score, 4),
                source=str(entry.get("source", "知识库")),
                source_url=entry.get("source_url"),
                law_status=entry.get("law_status"),
                suggested_department=entry.get("suggested_department"),
                explanation=self._explain_hit(entry),
            )
            for score, entry in scored_entries[:top_k]
        ]

    def _get_chroma_collection(self) -> Any | None:
        if not settings.use_chroma_retrieval or self._chroma_failed:
            return None
        if self._chroma_collection is not None:
            return self._chroma_collection
        try:
            import chromadb

            settings.chroma_dir.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(settings.chroma_dir))
            embedding_function = create_embedding_function()
            collection = client.get_or_create_collection(
                name=f"market_regulation_knowledge_{embedding_function.name()}",
                embedding_function=embedding_function,
                metadata={"hnsw:space": "cosine"},
            )
            self._ensure_chroma_index(collection)
            self._chroma_collection = collection
            return collection
        except Exception:
            self._chroma_failed = True
            self.last_status = {
                "retrieval_source": "CHROMA_ERROR",
                "degraded": True,
                "fallback_reason": "Chroma 初始化失败",
                "hit_count": 0,
            }
            return None

    def _ensure_chroma_index(self, collection: Any) -> None:
        expected_count = len(self.entries)
        try:
            if collection.count() == expected_count:
                return
        except Exception:
            pass
        ids = [str(entry.get("id")) for entry in self.entries]
        documents = [self._entry_text(entry) for entry in self.entries]
        metadatas = [self._entry_metadata(entry) for entry in self.entries]
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def list_law_documents(self) -> list[dict[str, Any]]:
        documents: dict[str, dict[str, Any]] = {}
        for entry in self.entries:
            doc_id = self._doc_id_for_entry(entry)
            if doc_id.startswith("transfer_"):
                continue
            document = documents.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "title": self._document_title(str(entry.get("title", ""))),
                    "article_count": 0,
                    "source": entry.get("source"),
                    "source_url": entry.get("source_url"),
                    "law_status": entry.get("law_status"),
                },
            )
            document["article_count"] += 1
        return sorted(documents.values(), key=lambda item: str(item["title"]))

    def get_law_full_text(self, doc_id: str) -> dict[str, Any] | None:
        articles = [entry for entry in self.entries if self._doc_id_for_entry(entry) == doc_id]
        if not articles:
            return None
        first = articles[0]
        return {
            "doc_id": doc_id,
            "title": self._document_title(str(first.get("title", ""))),
            "source": first.get("source"),
            "source_url": first.get("source_url"),
            "law_status": first.get("law_status"),
            "articles": [
                RetrievalHit(
                    knowledge_id=str(entry.get("id", "")) or None,
                    title=str(entry.get("title", "")),
                    content=str(entry.get("content", "")),
                    score=1.0,
                    source=str(entry.get("source", "知识库")),
                    source_url=entry.get("source_url"),
                    law_status=entry.get("law_status"),
                    suggested_department=entry.get("suggested_department"),
                    explanation=self._explain_hit(entry),
                )
                for entry in articles
            ],
        }

    def _load_entries(self) -> list[dict[str, Any]]:
        if not self.knowledge_base_path.exists():
            return self.DEFAULT_KB
        try:
            raw_entries = json.loads(self.knowledge_base_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self.DEFAULT_KB
        if not isinstance(raw_entries, list):
            return self.DEFAULT_KB

        cleaned_entries = [entry for entry in raw_entries if self._is_valid_entry(entry)]
        return cleaned_entries or self.DEFAULT_KB

    @staticmethod
    def _is_valid_entry(entry: object) -> bool:
        return (
            isinstance(entry, dict)
            and isinstance(entry.get("title"), str)
            and isinstance(entry.get("content"), str)
            and isinstance(entry.get("source"), str)
        )

    @classmethod
    def _score_entry(cls, entry: dict[str, Any], reason_type: ReasonType, query: str) -> float:
        query = query or ""
        score = 0.0

        reason_types = entry.get("reason_types") or []
        if reason_type.value in reason_types:
            score += 0.45
        elif not reason_types:
            score += 0.05

        keywords = [str(keyword) for keyword in entry.get("keywords") or []]
        entry_text = f"{entry.get('title', '')} {entry.get('content', '')}"
        matched_keywords = [
            keyword
            for keyword in keywords
            if keyword and (keyword in query or (keyword in entry_text and keyword in query))
        ]
        if matched_keywords:
            score += min(0.4, 0.12 * len(matched_keywords))

        content_keyword_hits = [keyword for keyword in keywords if keyword and keyword in query and keyword in entry_text]
        suggested_department_hits = [
            keyword
            for keyword in keywords
            if keyword and keyword in query and entry.get("suggested_department")
        ]
        if content_keyword_hits:
            score += min(0.3, 0.15 * len(content_keyword_hits))
        if suggested_department_hits:
            score += 0.25

        overlap = cls._text_overlap(query, entry_text)
        score += min(0.15, overlap * 0.03)
        return round(score, 4)

    @staticmethod
    def _text_overlap(left: str, right: str) -> int:
        left_tokens = {token for token in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9]+", left)}
        right_tokens = {token for token in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9]+", right)}
        return len(left_tokens & right_tokens)

    @staticmethod
    def _entry_text(entry: dict[str, Any]) -> str:
        keywords = " ".join(str(keyword) for keyword in entry.get("keywords") or [])
        return f"{entry.get('title', '')}\n{entry.get('content', '')}\n{keywords}"

    @staticmethod
    def _entry_metadata(entry: dict[str, Any]) -> dict[str, Any]:
        return {
            "title": str(entry.get("title", "")),
            "source": str(entry.get("source", "")),
            "reason_types": "|".join(str(reason) for reason in entry.get("reason_types") or []),
            "suggested_department": str(entry.get("suggested_department") or ""),
            "law_status": str(entry.get("law_status") or ""),
        }

    @staticmethod
    def _has_query_keyword(entry: dict[str, Any], query: str) -> bool:
        return any(str(keyword) and str(keyword) in query for keyword in entry.get("keywords") or [])

    @staticmethod
    def _explain_hit(entry: dict[str, Any]) -> str:
        title = str(entry.get("title", "该条文"))
        content = str(entry.get("content", ""))
        suggested_department = entry.get("suggested_department")
        if suggested_department:
            return f"这条主要用于判断事项是否应转其他主管部门处理，可作为建议向{suggested_department}反映的参考。"
        if "不予受理" in content or "处理权限" in content:
            return "这条主要用于说明哪些投诉可以不予受理，以及受理权限如何判断。"
        if "明码标价" in content or "价格" in title:
            return "这条主要用于价格、收费、明码标价、价外加价等问题的依据说明。"
        if "退货" in content or "消费者" in title:
            return "这条主要用于消费者权益、退货退款、赔偿和经营者义务等问题的依据说明。"
        if "物业" in title or "物业" in content:
            return "这条主要用于物业服务收费、收费公示和物业主管部门职责的依据说明。"
        return "这条可作为办理时查阅的法规依据，具体适用仍需工作人员结合事实判断。"

    @staticmethod
    def _doc_id_for_entry(entry: dict[str, Any]) -> str:
        entry_id = str(entry.get("id", ""))
        if entry_id.endswith(tuple(f"_{index:03d}" for index in range(1, 200))):
            return entry_id.rsplit("_", 1)[0]
        return entry_id

    @staticmethod
    def _document_title(title: str) -> str:
        return re.sub(r"\s+第[一二三四五六七八九十百零〇两]+条$", "", title).strip()
