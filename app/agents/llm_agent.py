from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from app.agents.base import agent_step
from app.agents.classifier import ClassifierAgent
from app.agents.dispatch import DispatchAgent
from app.agents.reply import ReplyAgent
from app.agents.retrieval import RetrievalAgent
from app.agents.tools import (
    
    ToolDefinition,
    ToolRegistry,
    build_tools_registry,
)
from app.core.config import settings
from app.core.enums import AcceptSuggestion, DecisionSource, ReasonType
from app.core.schemas import (
    AgentStep,
    AnalyzeResponse,
    ClassificationResult,
    ComplaintAnalyzeRequest,
    DispatchResult,
    RejectReasonSuggestion,
    ReplyDraft,
    RetrievalHit,
)
from app.core.storage import JsonlStore
from app.core.text import mask_pii, normalize_text
from uuid import uuid4
class LLMClient(ABC):
    """Abstract LLM backend — pluggable: Ollama, OpenAI, vLLM, etc."""

    @abstractmethod
    async def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Return OpenAI-compatible chat response dict with optional tool_calls."""
        ...

    @abstractmethod
    async def chat_stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield SSE chunks: {"type": "token", "content": "..."} | {"type": "tool_call", ...} | {"type": "done"}."""
        ...
class OllamaClient(LLMClient):
    """Ollama backend using its OpenAI-compatible /v1/chat/completions endpoint."""

    def __init__(
        self, base_url: str = "http://localhost:11434/v1", model: str = "qwen2.5:7b", timeout: float = 60.0
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    async def _request(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, stream: bool):
        import aiohttp

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if stream:
            payload["stream"] = True

        session_timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=session_timeout) as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as resp:
                if stream:
                    return resp
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"LLM API error {resp.status}: {text[:500]}")
                return await resp.json()

    async def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        return await self._request(messages, tools, stream=False)

    async def chat_stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        resp = await self._request(messages, tools, stream=True)
        buffer = ""
        async for line in resp.content:
            text = line.decode("utf-8").strip()
            if not text or text == "data: [DONE]":
                continue
            if text.startswith("data: "):
                text = text[6:]
            try:
                chunk = json.loads(text)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            if "content" in delta and delta["content"]:
                yield {"type": "token", "content": delta["content"]}
            if "tool_calls" in delta:
                for tc in delta["tool_calls"]:
                    idx = tc.get("index", 0)
                    func = tc.get("function", {})
                    yield {
                        "type": "tool_call_delta",
                        "index": idx,
                        "id": tc.get("id"),
                        "name": func.get("name"),
                        "arguments": func.get("arguments", ""),
                    }
            if choices[0].get("finish_reason"):
                yield {"type": "done", "finish_reason": choices[0]["finish_reason"]}
                return
        yield {"type": "done", "finish_reason": "stop"}
SYSTEM_PROMPT = """你是一个市场监管投诉智能处理助手。你的任务是分析市民投诉，判断是否属于市场监管职责范围，并给出处理建议。

你可以使用以下工具来完成分析：
- **classify_complaint**: 判断投诉是否属于市场监管受理范围，识别不受理原因
- **search_regulations**: 检索相关法规条文作为处理依据
- **dispatch_to_office**: 将投诉分派到具体的市场监管所
- **generate_reply**: 生成给市民的回复建议

工作流程：
1. 首先使用 classify_complaint 判断是否受理
2. 如果不受理：用 search_regulations 查法规依据，然后 generate_reply 生成退回回复
3. 如果受理：用 dispatch_to_office 分派机构，然后 generate_reply 生成受理回复
4. 最后用中文总结你的分析结果和处理建议

注意：
- 所有工具调用必须基于投诉原文，不要编造信息
- 如果工具返回 confidence < 0.6，建议人工复核
- 涉农药/兽药→转农业农村局；物业/供暖→转住建部门；治安事件→转公安机关
"""
class LLMAgentOrchestrator:
    """ReAct-pattern orchestrator: LLM reasons → calls tools → observes → responds.

    Demonstrates: tool definition, structured function calling, streaming,
    fallback handling, and human-in-the-loop confidence gating.
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        classifier: ClassifierAgent | None = None,
        dispatch_agent: DispatchAgent | None = None,
        retrieval_agent: RetrievalAgent | None = None,
        reply_agent: ReplyAgent | None = None,
        trace_store: JsonlStore | None = None,
    ) -> None:
        self.llm = llm or OllamaClient()
        self.classifier = classifier or ClassifierAgent()
        self.dispatch_agent = dispatch_agent or DispatchAgent()
        self.retrieval_agent = retrieval_agent or RetrievalAgent()
        self.reply_agent = reply_agent or ReplyAgent()
        self.trace_store = trace_store or JsonlStore(settings.traces_path)
        self.tool_registry = build_tools_registry(
            classifier=self.classifier,
            dispatch_agent=self.dispatch_agent,
            retrieval_agent=self.retrieval_agent,
            reply_agent=self.reply_agent,
        )

    async def analyze(self, request: ComplaintAnalyzeRequest) -> AnalyzeResponse:
        trace_id = str(uuid4())
        steps: list[AgentStep] = []
        review_reasons: list[str] = []

        # Preprocess
        with agent_step(steps, "preprocess", {"registration_id": request.registration_id or ""}) as step:
            clean_request = request.model_copy(
                update={
                    "problem_text": normalize_text(mask_pii(request.problem_text), settings.max_text_chars),
                    "appeal_text": normalize_text(mask_pii(request.appeal_text), settings.max_text_chars),
                    "enterprise_address": normalize_text(request.enterprise_address, 300),
                    "incident_location": normalize_text(request.incident_location, 300),
                }
            )
            step["output_summary"] = {
                "problem_text_chars": len(clean_request.problem_text),
                "has_incident_location": bool(clean_request.incident_location),
                "has_enterprise_address": bool(clean_request.enterprise_address),
            }

        # Build messages
        user_message = self._build_user_message(clean_request)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        # Tool definitions for LLM
        tool_defs = [t.to_openai_schema() for t in self.tool_registry.list_tools()]

        # State trackers
        classification: ClassificationResult | None = None
        dispatch: DispatchResult | None = None
        retrieval_hits: list[RetrievalHit] = []
        reject_reason_suggestion: RejectReasonSuggestion | None = None
        reply: ReplyDraft | None = None
        llm_summary = ""
        max_turns = 5

        with agent_step(steps, "llm_agent", {"model": self.llm.model, "max_turns": max_turns}) as step:
            for turn in range(max_turns):
                try:
                    response = await self.llm.chat(messages, tool_defs)
                except Exception as exc:
                    step["error"] = f"LLM error at turn {turn}: {exc}"
                    step["degraded"] = True
                    review_reasons.append(f"LLM 调用异常(第{turn + 1}轮): {exc}")
                    break

                choice = response.get("choices", [{}])[0]
                msg = choice.get("message", {})

                # If LLM returns a text response (no tool call), it's the final answer
                if msg.get("content") and not msg.get("tool_calls"):
                    llm_summary = msg["content"]
                    messages.append({"role": "assistant", "content": msg["content"]})
                    break

                # Handle tool calls
                tool_calls = msg.get("tool_calls", [])
                if not tool_calls:
                    break

                # Record assistant message with tool calls
                messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tool_calls})

                for tc in tool_calls:
                    func = tc.get("function", {})
                    tool_name = func.get("name", "")
                    raw_args = func.get("arguments", "{}")
                    if isinstance(raw_args, str):
                        try:
                            args = json.loads(raw_args)
                        except json.JSONDecodeError:
                            args = {}
                    else:
                        args = raw_args

                    result = await self._execute_tool(tool_name, args, clean_request)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", tool_name),
                        "content": json.dumps(result, ensure_ascii=False),
                    })

                    # Extract structured results from tool outputs
                    if tool_name == "classify_complaint" and "classification" in result:
                        classification = ClassificationResult(**result["classification"])
                        review_reasons.extend(result.get("review_reasons", []))
                    elif tool_name == "dispatch_to_office" and "dispatch" in result:
                        dispatch = DispatchResult(**result["dispatch"])
                        if result.get("needs_review"):
                            review_reasons.append("分派结果置信度不足或使用默认所")
                    elif tool_name == "search_regulations":
                        retrieval_hits = [RetrievalHit(**h) for h in result.get("hits", [])]
                    elif tool_name == "generate_reply" and "reply" in result:
                        reply = ReplyDraft(**result["reply"])
                        if not reply.validation_passed or reply.fallback_reason:
                            review_reasons.append(reply.fallback_reason or "回复生成降级")
            else:
                # Max turns reached — ask for final summary
                messages.append({"role": "user", "content": "请基于以上工具调用结果，用中文给出最终分析总结。"})
                try:
                    response = await self.llm.chat(messages)
                    llm_summary = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                except Exception:
                    llm_summary = "（LLM 异常，无法生成总结。请人工复核。）"
                    review_reasons.append("LLM 总结生成失败")

            step["output_summary"] = {
                "llm_model": self.llm.model,
                "turns": min(turn + 1, max_turns),
                "llm_summary": llm_summary[:500],
                "tool_calls_made": self._extract_tool_names(messages),
            }

        # Fallback: if classification is missing, run it directly
        if classification is None:
            with agent_step(steps, "classify_fallback", {}) as fb_step:
                classification, fb_reasons = self.classifier.classify(clean_request)
                review_reasons.extend(fb_reasons)
                fb_step["output_summary"] = classification.model_dump(mode="json")
                fb_step["degraded"] = True

        # Fallback: if reply is missing, generate one
        if reply is None:
            with agent_step(steps, "reply_fallback", {}) as fb_step:
                reply = self.reply_agent.draft(classification, dispatch, retrieval_hits)
                fb_step["output_summary"] = reply.model_dump(mode="json")
                fb_step["degraded"] = True

        # Validate
        with agent_step(steps, "validate", {}) as step:
            if classification.confidence < settings.low_confidence_threshold:
                review_reasons.append("分类置信度低")
            if classification.accept_suggestion == AcceptSuggestion.REVIEW:
                review_reasons.append("分类建议为人工复核")
            review_reasons = list(dict.fromkeys(r for r in review_reasons if r))
            step["output_summary"] = {"review_required": bool(review_reasons), "review_reasons": review_reasons}

        response = AnalyzeResponse(
            trace_id=trace_id,
            classification=classification,
            reject_reason_suggestion=reject_reason_suggestion,
            dispatch=dispatch,
            retrieval_hits=retrieval_hits,
            reply_draft=reply,
            review_required=bool(review_reasons),
            review_reasons=review_reasons,
            agent_steps=steps,
        )
        self.trace_store.append(response)
        return response

    async def analyze_stream(self, request: ComplaintAnalyzeRequest) -> AsyncIterator[dict[str, Any]]:
        """Streaming analysis — yields SSE events for frontend live display."""
        trace_id = str(uuid4())
        clean_request = request.model_copy(
            update={
                "problem_text": normalize_text(mask_pii(request.problem_text), settings.max_text_chars),
                "appeal_text": normalize_text(mask_pii(request.appeal_text), settings.max_text_chars),
            }
        )

        user_message = self._build_user_message(clean_request)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        tool_defs = [t.to_openai_schema() for t in self.tool_registry.list_tools()]

        yield {"event": "start", "data": {"trace_id": trace_id}}

        max_turns = 5
        for turn in range(max_turns):
            yield {"event": "thinking", "data": {"turn": turn + 1}}

            async for chunk in self.llm.chat_stream(messages, tool_defs):
                yield {"event": chunk["type"], "data": chunk}

            # Non-streaming fallback for tool execution
            try:
                response = await self.llm.chat(messages, tool_defs)
            except Exception as exc:
                yield {"event": "error", "data": {"message": str(exc)}}
                break

            choice = response.get("choices", [{}])[0]
            msg = choice.get("message", {})
            if msg.get("content") and not msg.get("tool_calls"):
                yield {"event": "final_answer", "data": {"content": msg["content"]}}
                break

            tool_calls = msg.get("tool_calls", [])
            if not tool_calls:
                break

            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tool_calls})
            for tc in tool_calls:
                func = tc.get("function", {})
                args = json.loads(func.get("arguments", "{}"))
                tool_name = func.get("name", "unknown")
                yield {"event": "node_start", "data": {"node": tool_name}}
                result = await self._execute_tool(tool_name, args, clean_request)
                yield {"event": "node_end", "data": {"node": tool_name, "result_summary": str(result)[:200]}}
                yield {"event": "tool_result", "data": {"tool": tool_name, "result": result}}
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": json.dumps(result, ensure_ascii=False),
                })
        else:
            yield {"event": "max_turns", "data": {"turns": max_turns}}

        yield {"event": "done", "data": {"trace_id": trace_id}}

    def _build_user_message(self, request: ComplaintAnalyzeRequest) -> str:
        parts = [f"投诉内容：{request.problem_text}"]
        if request.appeal_text:
            parts.append(f"诉求：{request.appeal_text}")
        if request.enterprise_name:
            parts.append(f"被投诉方：{request.enterprise_name}")
        if request.incident_location:
            parts.append(f"事发地点：{request.incident_location}")
        if request.enterprise_address:
            parts.append(f"企业地址：{request.enterprise_address}")
        return "\n".join(parts)

    async def _execute_tool(
        self, name: str, args: dict[str, Any], request: ComplaintAnalyzeRequest
    ) -> dict[str, Any]:
        try:
            return self.tool_registry.execute(name, args, request)
        except Exception as exc:
            return {"error": str(exc), "tool": name}

    @staticmethod
    def _extract_tool_names(messages: list[dict[str, Any]]) -> list[str]:
        names: list[str] = []
        for msg in messages:
            for tc in msg.get("tool_calls", []):
                name = tc.get("function", {}).get("name", "")
                if name:
                    names.append(name)
        return names
