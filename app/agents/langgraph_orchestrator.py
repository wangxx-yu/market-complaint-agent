from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypedDict
from uuid import uuid4

from langgraph.graph import END, StateGraph

from app.agents.base import agent_step
from app.agents.answer_verifier import AnswerVerifier
from app.agents.classifier import ClassifierAgent
from app.agents.dispatch import DispatchAgent
from app.agents.reject_reason import RejectReasonAgent
from app.agents.reply import ReplyAgent
from app.agents.retrieval import RetrievalAgent
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


class ComplaintGraphState(TypedDict, total=False):
    trace_id: str
    request: ComplaintAnalyzeRequest
    clean_request: ComplaintAnalyzeRequest
    steps: list[AgentStep]
    review_reasons: list[str]
    classification: ClassificationResult
    reject_reason_suggestion: RejectReasonSuggestion | None
    dispatch: DispatchResult | None
    retrieval_hits: list[RetrievalHit]
    reply: ReplyDraft
    response: AnalyzeResponse


class LangGraphOrchestrator:
    """LangGraph-based orchestrator that reuses the existing worker agents."""

    def __init__(
        self,
        classifier: ClassifierAgent | None = None,
        dispatch_agent: DispatchAgent | None = None,
        reject_reason_agent: RejectReasonAgent | None = None,
        retrieval_agent: RetrievalAgent | None = None,
        reply_agent: ReplyAgent | None = None,
        answer_verifier: AnswerVerifier | None = None,
        trace_store: JsonlStore | None = None,
    ) -> None:
        self.classifier = classifier or ClassifierAgent()
        self.dispatch_agent = dispatch_agent or DispatchAgent()
        self.reject_reason_agent = reject_reason_agent or RejectReasonAgent()
        self.retrieval_agent = retrieval_agent or RetrievalAgent()
        self.reply_agent = reply_agent or ReplyAgent()
        self.answer_verifier = answer_verifier or AnswerVerifier()
        self.trace_store = trace_store or JsonlStore(settings.traces_path)
        self.graph = self._build_graph()

    def analyze(self, request: ComplaintAnalyzeRequest) -> AnalyzeResponse:
        initial_state: ComplaintGraphState = {
            "trace_id": str(uuid4()),
            "request": request,
            "steps": [],
            "review_reasons": [],
            "dispatch": None,
            "reject_reason_suggestion": None,
            "retrieval_hits": [],
        }
        final_state = self.graph.invoke(initial_state)
        response = final_state["response"]
        self.trace_store.append(response)
        return response

    def _build_graph(self):
        graph = StateGraph(ComplaintGraphState)
        graph.add_node("preprocess", self._preprocess)
        graph.add_node("classify", self._classify)
        graph.add_node("reject_reason", self._reject_reason)
        graph.add_node("run_dispatch", self._dispatch)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("fanout_accept", self._fanout_accept)
        graph.add_node("fanout_reject", self._fanout_reject)
        graph.add_node("do_reply", self._reply)
        graph.add_node("validate", self._validate)
        graph.add_node("audit_log", self._audit_log)

        graph.set_entry_point("preprocess")
        graph.add_edge("preprocess", "classify")
        graph.add_conditional_edges(
            "classify",
            self._route_after_classify,
            {
                "fanout_accept": "fanout_accept",
                "fanout_reject": "fanout_reject",
                "reject_reason": "reject_reason",
                "retrieve": "retrieve",
                "do_reply": "do_reply",
            },
        )
        graph.add_edge("fanout_accept", "do_reply")
        graph.add_edge("fanout_reject", "do_reply")
        graph.add_edge("reject_reason", "retrieve")
        graph.add_edge("retrieve", "do_reply")
        graph.add_edge("run_dispatch", "do_reply")
        graph.add_edge("do_reply", "validate")
        graph.add_edge("validate", "audit_log")
        graph.add_edge("audit_log", END)
        return graph.compile()

    def _preprocess(self, state: ComplaintGraphState) -> ComplaintGraphState:
        request = state["request"]
        with agent_step(state["steps"], "preprocess", {"registration_id": request.registration_id or ""}) as step:
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
                "problem_text": clean_request.problem_text,
                "has_incident_location": bool(clean_request.incident_location),
                "has_enterprise_address": bool(clean_request.enterprise_address),
            }
        state["clean_request"] = clean_request
        return state

    def _classify(self, state: ComplaintGraphState) -> ComplaintGraphState:
        clean_request = state["clean_request"]
        with agent_step(state["steps"], "classify", {"problem_text": clean_request.problem_text[:120]}) as step:
            try:
                classification, review_reasons = self.classifier.classify(clean_request)
            except Exception as exc:
                classification = ClassificationResult(
                    is_market=True,
                    accept_suggestion=AcceptSuggestion.REVIEW,
                    reason_type=ReasonType.UNKNOWN,
                    confidence=0.0,
                    decision_source=DecisionSource.FALLBACK,
                    evidence_fields=["classifier_error"],
                )
                review_reasons = ["是否受理模型异常，需人工判断"]
                step["error"] = str(exc)
                step["degraded"] = True
            state["review_reasons"].extend(review_reasons)
            step["output_summary"] = classification.model_dump(mode="json")
            step["confidence"] = classification.confidence
            step["decision_source"] = classification.decision_source.value
        state["classification"] = classification
        return state

    def _reject_reason(self, state: ComplaintGraphState) -> ComplaintGraphState:
        clean_request = state["clean_request"]
        classification = state["classification"]
        with agent_step(
            state["steps"],
            "reject_reason",
            {"reason_type": classification.reason_type, "accept_suggestion": classification.accept_suggestion},
        ) as step:
            try:
                suggestion = self.reject_reason_agent.suggest(clean_request, classification)
            except Exception as exc:
                suggestion = RejectReasonSuggestion(
                    reason_type=ReasonType.UNKNOWN,
                    confidence=0.0,
                    decision_source=DecisionSource.FALLBACK,
                    evidence_fields=["reject_reason_agent_error"],
                    needs_review=True,
                    note="不受理原因建议失败，请人工填写具体原因。",
                )
                step["error"] = str(exc)
                step["degraded"] = True
            step["output_summary"] = suggestion.model_dump(mode="json") if suggestion else {}
            step["confidence"] = suggestion.confidence if suggestion else None
            if suggestion and suggestion.needs_review:
                state["review_reasons"].append(suggestion.note or "不受理原因建议需人工确认")
        state["reject_reason_suggestion"] = suggestion
        return state

    def _dispatch(self, state: ComplaintGraphState) -> ComplaintGraphState:
        clean_request = state["clean_request"]
        with agent_step(
            state["steps"],
            "dispatch",
            {"address": " ".join(filter(None, [clean_request.incident_location, clean_request.enterprise_address]))[:160]},
        ) as step:
            try:
                dispatch = self.dispatch_agent.dispatch(clean_request)
            except Exception as exc:
                dispatch = DispatchResult(
                    office_code="",
                    office_name="待人工选择",
                    confidence=0.0,
                    decision_source=DecisionSource.FALLBACK,
                    matched_rule="dispatch_error",
                    needs_review=True,
                )
                step["error"] = str(exc)
                step["degraded"] = True
                state["review_reasons"].append("分派 Agent 异常，需人工选择市场监管所")
            step["output_summary"] = dispatch.model_dump(mode="json")
            step["confidence"] = dispatch.confidence
            step["decision_source"] = dispatch.decision_source.value
            if dispatch.needs_review:
                state["review_reasons"].append("分派结果置信度不足或使用默认所")
        state["dispatch"] = dispatch
        return state

    def _retrieve(self, state: ComplaintGraphState) -> ComplaintGraphState:
        classification = state["classification"]
        clean_request = state["clean_request"]
        if classification.reason_type == ReasonType.ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY:
            state["review_reasons"].append("职责外事项需人工确认转办部门")
        with agent_step(state["steps"], "retrieve", {"reason_type": classification.reason_type}) as step:
            try:
                hits = self.retrieval_agent.retrieve(classification.reason_type, clean_request.problem_text)
            except Exception as exc:
                hits = []
                step["error"] = str(exc)
                step["degraded"] = True
                state["review_reasons"].append("RAG 检索异常，回复依据需人工核对")
            retrieval_status = dict(getattr(self.retrieval_agent, "last_status", {}))
            if step["degraded"] and not retrieval_status:
                retrieval_status = {
                    "retrieval_source": "ERROR_FALLBACK",
                    "degraded": True,
                    "fallback_reason": "RAG 检索异常，未返回法规依据",
                    "hit_count": 0,
                }
            step["output_summary"] = {
                "hits": [hit.model_dump(mode="json") for hit in hits],
                "retrieval_status": retrieval_status,
            }
            step["confidence"] = hits[0].score if hits else None
            step["degraded"] = bool(retrieval_status.get("degraded"))
            if retrieval_status.get("fallback_reason"):
                state["review_reasons"].append(str(retrieval_status["fallback_reason"]))
        state["retrieval_hits"] = hits
        return state

    def _reply(self, state: ComplaintGraphState) -> ComplaintGraphState:
        classification = state["classification"]
        with agent_step(state["steps"], "reply", {"is_market": classification.is_market}) as step:
            try:
                if "classifier_error" in classification.evidence_fields:
                    raise RuntimeError("classification unavailable")
                reply = self.reply_agent.draft(classification, state.get("dispatch"), state.get("retrieval_hits", []))
            except Exception as exc:
                reply = ReplyDraft(
                    text="系统暂时无法自动生成可靠办理建议，请工作人员人工复核后处理。",
                    decision_source=DecisionSource.FALLBACK,
                    template_id="system_fallback",
                    validation_passed=False,
                    fallback_reason="回复 Agent 异常，已降级为人工复核提示",
                )
                step["error"] = str(exc)
                step["degraded"] = True
            # AnswerVerifier 校验
            if reply.text and reply.validation_passed:
                verification = self.answer_verifier.verify(reply.text, state.get("retrieval_hits", []))
                if verification.issues:
                    step["output_summary"]["verification_issues"] = verification.issues
                if verification.fallback_reply:
                    reply = ReplyDraft(
                        text=verification.fallback_reply,
                        decision_source=DecisionSource.FALLBACK,
                        template_id="verification_fallback",
                        validation_passed=False,
                        fallback_reason="; ".join(verification.issues) if verification.issues else "法规引用校验失败",
                    )
                    step["degraded"] = True
                elif verification.requires_review:
                    state["review_reasons"].extend(verification.issues)
            step["output_summary"] = reply.model_dump(mode="json")
            step["confidence"] = 0.9 if reply.validation_passed else 0.45
            step["decision_source"] = reply.decision_source.value
            if not reply.validation_passed or reply.fallback_reason:
                state["review_reasons"].append(reply.fallback_reason or "回复生成降级")
        state["reply"] = reply
        return state

    def _validate(self, state: ComplaintGraphState) -> ComplaintGraphState:
        classification = state["classification"]
        with agent_step(state["steps"], "validate", {}) as step:
            if classification.confidence < settings.low_confidence_threshold:
                state["review_reasons"].append("分类置信度低")
            if classification.accept_suggestion.value == "REVIEW":
                state["review_reasons"].append("分类建议为人工复核")
            state["review_reasons"] = list(dict.fromkeys(reason for reason in state["review_reasons"] if reason))
            step["output_summary"] = {
                "review_required": bool(state["review_reasons"]),
                "review_reasons": state["review_reasons"],
            }
        return state

    def _audit_log(self, state: ComplaintGraphState) -> ComplaintGraphState:
        response = AnalyzeResponse(
            trace_id=state["trace_id"],
            classification=state["classification"],
            reject_reason_suggestion=state.get("reject_reason_suggestion"),
            dispatch=state.get("dispatch"),
            retrieval_hits=state.get("retrieval_hits", []),
            reply_draft=state["reply"],
            review_required=bool(state["review_reasons"]),
            review_reasons=state["review_reasons"],
            agent_steps=state["steps"],
        )
        state["response"] = response
        return state

    @staticmethod
    def _route_after_classify(state: ComplaintGraphState) -> str:
        """Route to parallel fanout nodes when both dispatch and retrieve are needed."""
        classification = state["classification"]
        if "classifier_error" in classification.evidence_fields:
            return "do_reply"
        invalid_input = "invalid_input" in classification.evidence_fields
        has_reject_reason = classification.reason_type != ReasonType.UNKNOWN
        if invalid_input:
            return "do_reply"
        if classification.is_market and not has_reject_reason:
            # ACCEPT — parallel: dispatch + retrieve
            return "fanout_accept"
        if not classification.is_market or has_reject_reason:
            # REJECT — parallel: reject_reason + retrieve
            return "fanout_reject"
        return "do_reply"

    def _fanout_accept(self, state: ComplaintGraphState) -> ComplaintGraphState:
        """Parallel fan-out for ACCEPT path: dispatch ∥ retrieve."""
        clean_request = state["clean_request"]

        def run_dispatch():
            with agent_step(state["steps"], "dispatch",
                            {"address": " ".join(filter(None, [clean_request.incident_location, clean_request.enterprise_address]))[:160]}) as step:
                try:
                    result = self.dispatch_agent.dispatch(clean_request)
                except Exception as exc:
                    result = DispatchResult(
                        office_code="", office_name="待人工选择", confidence=0.0,
                        decision_source=DecisionSource.FALLBACK, matched_rule="dispatch_error", needs_review=True,
                    )
                    step["error"] = str(exc)
                    step["degraded"] = True
                    state["review_reasons"].append("分派 Agent 异常，需人工选择市场监管所")
                step["output_summary"] = result.model_dump(mode="json")
                step["confidence"] = result.confidence
                if result.needs_review:
                    state["review_reasons"].append("分派结果置信度不足或使用默认所")
                return result

        def run_retrieve():
            classification = state["classification"]
            with agent_step(state["steps"], "retrieve", {"reason_type": classification.reason_type}) as step:
                try:
                    hits = self.retrieval_agent.retrieve(classification.reason_type, clean_request.problem_text)
                except Exception as exc:
                    hits = []
                    step["error"] = str(exc)
                    step["degraded"] = True
                retrieval_status = dict(getattr(self.retrieval_agent, "last_status", {}))
                step["output_summary"] = {
                    "hits": [hit.model_dump(mode="json") for hit in hits],
                    "retrieval_status": retrieval_status,
                }
                step["confidence"] = hits[0].score if hits else None
                step["degraded"] = bool(retrieval_status.get("degraded"))
                if retrieval_status.get("fallback_reason"):
                    state["review_reasons"].append(str(retrieval_status["fallback_reason"]))
                return hits

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_dispatch = executor.submit(run_dispatch)
            future_retrieve = executor.submit(run_retrieve)
            dispatch_result = future_dispatch.result()
            retrieve_result = future_retrieve.result()

        state["dispatch"] = dispatch_result
        state["retrieval_hits"] = retrieve_result
        return state

    def _fanout_reject(self, state: ComplaintGraphState) -> ComplaintGraphState:
        """Parallel fan-out for REJECT path: reject_reason ∥ retrieve."""
        clean_request = state["clean_request"]
        classification = state["classification"]

        if classification.reason_type == ReasonType.ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY:
            state["review_reasons"].append("职责外事项需人工确认转办部门")

        def run_reject_reason():
            with agent_step(state["steps"], "reject_reason",
                            {"reason_type": classification.reason_type, "accept_suggestion": classification.accept_suggestion}) as step:
                try:
                    suggestion = self.reject_reason_agent.suggest(clean_request, classification)
                except Exception as exc:
                    suggestion = RejectReasonSuggestion(
                        reason_type=ReasonType.UNKNOWN, confidence=0.0,
                        decision_source=DecisionSource.FALLBACK,
                        evidence_fields=["reject_reason_agent_error"],
                        needs_review=True, note="不受理原因建议失败，请人工填写具体原因。",
                    )
                    step["error"] = str(exc)
                    step["degraded"] = True
                step["output_summary"] = suggestion.model_dump(mode="json") if suggestion else {}
                step["confidence"] = suggestion.confidence if suggestion else None
                if suggestion and suggestion.needs_review:
                    state["review_reasons"].append(suggestion.note or "不受理原因建议需人工确认")
                return suggestion

        def run_retrieve():
            with agent_step(state["steps"], "retrieve", {"reason_type": classification.reason_type}) as step:
                try:
                    hits = self.retrieval_agent.retrieve(classification.reason_type, clean_request.problem_text)
                except Exception as exc:
                    hits = []
                    step["error"] = str(exc)
                    step["degraded"] = True
                    state["review_reasons"].append("RAG 检索异常，回复依据需人工核对")
                retrieval_status = dict(getattr(self.retrieval_agent, "last_status", {}))
                step["output_summary"] = {
                    "hits": [hit.model_dump(mode="json") for hit in hits],
                    "retrieval_status": retrieval_status,
                }
                step["confidence"] = hits[0].score if hits else None
                step["degraded"] = bool(retrieval_status.get("degraded"))
                if retrieval_status.get("fallback_reason"):
                    state["review_reasons"].append(str(retrieval_status["fallback_reason"]))
                return hits

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_reject = executor.submit(run_reject_reason)
            future_retrieve = executor.submit(run_retrieve)
            reject_suggestion = future_reject.result()
            retrieve_result = future_retrieve.result()

        state["reject_reason_suggestion"] = reject_suggestion
        state["retrieval_hits"] = retrieve_result
        return state
