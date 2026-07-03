"""Agent Memory — 短期多轮对话 + 长期 few-shot 示例检索。

- ConversationMemory: 单次会话上下文管理，支持滑动窗口
- FewShotMemory: 从历史复核中检索相似案例作为 few-shot 注入
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from app.core.config import settings
from app.core.storage import JsonlStore


@dataclass
class ConversationMemory:
    """滑动窗口短期记忆 — 管理单次会话的对话历史。"""

    max_turns: int = 10
    _messages: deque[dict[str, Any]] = field(default_factory=deque)

    def add(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})
        while len(self._messages) > self.max_turns * 2:
            self._messages.popleft()

    def to_messages(self) -> list[dict[str, Any]]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()


@dataclass
class FewShotMemory:
    """长期记忆 — 从历史复核中检索相似案例。

    把人工确认过的 (投诉文本, 回复) 对存入 ChromaDB，
    新投诉到来时检索相似历史案例作为 few-shot 上下文注入 LLM。
    """

    review_store: JsonlStore = field(default_factory=lambda: JsonlStore(settings.reviews_path))
    trace_store: JsonlStore = field(default_factory=lambda: JsonlStore(settings.traces_path))

    def get_similar_cases(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """从复核记录中检索与当前投诉相似的历史案例。

        当前实现用简单关键词匹配，后续可升级为向量检索。
        """
        reviews = self.review_store.all()
        if not reviews:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        query_tokens = set(query)

        for review_entry in reviews:
            trace_id = review_entry.get("trace_id", "")
            review = review_entry.get("review", {})
            if not isinstance(review, dict):
                continue

            # 跳过未标注的
            if review.get("is_market") is None:
                continue

            # 获取原始投诉文本
            trace = self.trace_store.find_by_key("trace_id", trace_id)
            if not trace:
                continue

            classification = trace.get("classification", {})
            problem_text = ""
            if isinstance(classification, dict):
                # 从 trace 的 agent_steps 中找原始输入
                steps = trace.get("agent_steps", [])
                for step in steps:
                    if step.get("name") == "preprocess":
                        problem_text = step.get("output_summary", {}).get("problem_text", "")
                        break

            if not problem_text:
                continue

            # 简单 Jaccard 相似度
            case_tokens = set(problem_text)
            intersection = len(query_tokens & case_tokens)
            union = len(query_tokens | case_tokens)
            score = intersection / union if union > 0 else 0

            if score > 0.15:
                scored.append((score, {
                    "trace_id": trace_id,
                    "problem_text": problem_text[:200],
                    "is_market": review.get("is_market"),
                    "reason_type": review.get("reason_type"),
                    "reply_text": review.get("reply_text", ""),
                    "office_name": review.get("office_name"),
                    "similarity": round(score, 3),
                }))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def format_for_prompt(self, cases: list[dict[str, Any]]) -> str:
        """将历史案例格式化为 LLM prompt 中的 few-shot 示例。"""
        if not cases:
            return ""

        lines = ["\n## 历史相似案例（仅供参考）\n"]
        for i, case in enumerate(cases, 1):
            lines.append(f"### 案例 {i}")
            lines.append(f"投诉: {case['problem_text']}")
            if case["is_market"] is True:
                lines.append(f"处理: 受理 → 分派 {case.get('office_name', '待定')}")
            else:
                lines.append(f"处理: 不受理 ({case.get('reason_type', '')})")
            if case["reply_text"]:
                lines.append(f"回复: {case['reply_text'][:200]}")
            lines.append("")
        return "\n".join(lines)
