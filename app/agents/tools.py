"""Tool definitions for the LLM Agent — wraps existing agents as callable tools.

Each tool follows the OpenAI function-calling schema convention so it can be
serialized into an LLM tool_choice / function_call request, regardless of
whether the backend is Ollama, OpenAI, or another OpenAI-compatible provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agents.classifier import ClassifierAgent
from app.agents.dispatch import DispatchAgent
from app.agents.retrieval import RetrievalAgent
from app.agents.reply import ReplyAgent
from app.core.enums import ReasonType
from app.core.schemas import (
    ClassificationResult,
    ComplaintAnalyzeRequest,
    DispatchResult,
    RetrievalHit,
)


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Any  # callable

    def to_openai_schema(self) -> dict[str, Any]:
        """Serialize to OpenAI function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Holds available tools and maps them to the LLM function-calling schema."""

    def __init__(
        self,
        classifier: ClassifierAgent | None = None,
        dispatch_agent: DispatchAgent | None = None,
        retrieval_agent: RetrievalAgent | None = None,
        reply_agent: ReplyAgent | None = None,
    ) -> None:
        self.classifier = classifier or ClassifierAgent()
        self.dispatch_agent = dispatch_agent or DispatchAgent()
        self.retrieval_agent = retrieval_agent or RetrievalAgent()
        self.reply_agent = reply_agent or ReplyAgent()

        self._tools: dict[str, ToolDefinition] = {
            t.name: t
            for t in [
                ToolDefinition(
                    name="classify_complaint",
                    description=(
                        "分析投诉内容，判断是否属于市场监管职责范围、是否应当受理。"
                        "返回受理建议、不受理原因类型、置信度等信息。"
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "problem_text": {
                                "type": "string",
                                "description": "投诉问题描述文本",
                            },
                        },
                        "required": ["problem_text"],
                    },
                    fn=self._classify,
                ),
                ToolDefinition(
                    name="dispatch_to_office",
                    description=(
                        "根据投诉地址信息，匹配并分派到对应的市场监管所。"
                        "返回机构代码、机构名称、匹配置信度。"
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "problem_text": {
                                "type": "string",
                                "description": "投诉问题描述（含地址信息）",
                            },
                            "incident_location": {
                                "type": "string",
                                "description": "事发地点",
                            },
                            "enterprise_address": {
                                "type": "string",
                                "description": "企业地址",
                            },
                            "enterprise_name": {
                                "type": "string",
                                "description": "企业名称",
                            },
                        },
                        "required": ["problem_text"],
                    },
                    fn=self._dispatch,
                ),
                ToolDefinition(
                    name="search_regulations",
                    description=(
                        "检索与投诉相关的法律法规条文，包括《消费者权益保护法》《食品安全法》"
                        "《市场监督管理投诉举报处理办法》等。返回相关法条及释义。"
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "法规检索查询文本",
                            },
                            "reason_type": {
                                "type": "string",
                                "description": "不受理原因类型（如 ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY）",
                            },
                        },
                        "required": ["query"],
                    },
                    fn=self._retrieve,
                ),
                ToolDefinition(
                    name="generate_reply",
                    description=(
                        "根据分类结果和检索到的法规依据，生成给投诉人的回复建议。"
                        "包含受理建议或退回理由。"
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "classification_summary": {
                                "type": "string",
                                "description": "分类结果的 JSON 摘要",
                            },
                            "regulation_reference": {
                                "type": "string",
                                "description": "法规依据文本（可选）",
                            },
                        },
                        "required": ["classification_summary"],
                    },
                    fn=self._reply,
                ),
            ]
        }

    @property
    def openai_tools(self) -> list[dict[str, Any]]:
        """Return tools in OpenAI function-calling format."""
        return [t.to_openai_schema() for t in self._tools.values()]

    def list_tools(self) -> list[ToolDefinition]:
        """Return all registered tool definitions."""
        return list(self._tools.values())

    def execute(self, name: str, arguments: dict[str, Any], _request: Any = None) -> dict[str, Any]:
        """Execute a tool by name with given arguments. Returns a JSON-serializable dict.
        
        _request is an optional ComplaintAnalyzeRequest for tools that need full request context.
        """
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"Unknown tool: {name}"}
        try:
            result = tool.fn(**arguments)
            return result
        except Exception as exc:
            return {"error": str(exc), "tool": name}

    # ── tool implementations ──────────────────────────────────────────

    def _classify(self, problem_text: str) -> dict[str, Any]:
        request = ComplaintAnalyzeRequest(problem_text=problem_text)
        result, review_reasons = self.classifier.classify(request)
        return {
            "is_market": result.is_market,
            "accept_suggestion": result.accept_suggestion.value,
            "reason_type": result.reason_type.value,
            "confidence": result.confidence,
            "decision_source": result.decision_source.value,
            "evidence_fields": result.evidence_fields,
            "review_reasons": review_reasons,
        }

    def _dispatch(
        self,
        problem_text: str,
        incident_location: str = "",
        enterprise_address: str = "",
        enterprise_name: str = "",
    ) -> dict[str, Any]:
        request = ComplaintAnalyzeRequest(
            problem_text=problem_text,
            incident_location=incident_location or None,
            enterprise_address=enterprise_address or None,
            enterprise_name=enterprise_name or None,
        )
        result: DispatchResult = self.dispatch_agent.dispatch(request)
        return {
            "office_code": result.office_code,
            "office_name": result.office_name,
            "confidence": result.confidence,
            "decision_source": result.decision_source.value,
            "matched_rule": result.matched_rule,
            "needs_review": result.needs_review,
        }

    def _retrieve(self, query: str, reason_type: str = "UNKNOWN") -> dict[str, Any]:
        try:
            rt = ReasonType(reason_type)
        except ValueError:
            rt = ReasonType.UNKNOWN
        hits: list[RetrievalHit] = self.retrieval_agent.retrieve(rt, query, top_k=3)
        return {
            "hits": [
                {
                    "title": h.title,
                    "content": h.content,
                    "score": h.score,
                    "source": h.source,
                    "suggested_department": h.suggested_department,
                    "explanation": h.explanation,
                }
                for h in hits
            ],
            "hit_count": len(hits),
        }

    def _reply(self, classification_summary: str, regulation_reference: str = "") -> dict[str, Any]:
        # Build a minimal ClassificationResult from summary
        import json

        try:
            summary = json.loads(classification_summary)
        except json.JSONDecodeError:
            summary = {}

        is_market = summary.get("is_market", False)
        from app.core.enums import AcceptSuggestion, DecisionSource

        classification = ClassificationResult(
            is_market=is_market,
            accept_suggestion=AcceptSuggestion(summary.get("accept_suggestion", "REVIEW")),
            reason_type=ReasonType(summary.get("reason_type", "UNKNOWN")),
            confidence=float(summary.get("confidence", 0.5)),
            decision_source=DecisionSource.RULE,
            evidence_fields=summary.get("evidence_fields", []),
        )

        # If we have regulation hits, mock them as RetrievalHit for the reply agent
        retrieval_hits: list[RetrievalHit] = []
        if regulation_reference:
            retrieval_hits.append(
                RetrievalHit(
                    title="法规依据",
                    content=regulation_reference,
                    score=0.8,
                    source="llm_agent",
                )
            )

        reply = self.reply_agent.draft(classification, None, retrieval_hits)
        return {
            "text": reply.text,
            "template_id": reply.template_id,
            "validation_passed": reply.validation_passed,
            "fallback_reason": reply.fallback_reason,
        }


def build_tools_registry(
    classifier: ClassifierAgent | None = None,
    dispatch_agent: DispatchAgent | None = None,
    retrieval_agent: RetrievalAgent | None = None,
    reply_agent: ReplyAgent | None = None,
) -> ToolRegistry:
    """Factory for ToolRegistry — wires agent instances into tool definitions."""
    return ToolRegistry(
        classifier=classifier,
        dispatch_agent=dispatch_agent,
        retrieval_agent=retrieval_agent,
        reply_agent=reply_agent,
    )
