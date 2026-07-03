from __future__ import annotations

from uuid import uuid4

from app.agents.base import agent_step
from app.agents.classifier import ClassifierAgent
from app.agents.dispatch import DispatchAgent
from app.agents.reject_reason import RejectReasonAgent
from app.agents.reply import ReplyAgent
from app.agents.retrieval import RetrievalAgent
from app.core.config import settings
from app.core.enums import ReasonType
from app.core.schemas import AgentStep, AnalyzeResponse, ComplaintAnalyzeRequest
from app.core.storage import JsonlStore
from app.core.text import mask_pii, normalize_text


class Orchestrator:
    def __init__(
        self,
        classifier: ClassifierAgent | None = None,
        dispatch_agent: DispatchAgent | None = None,
        reject_reason_agent: RejectReasonAgent | None = None,
        retrieval_agent: RetrievalAgent | None = None,
        reply_agent: ReplyAgent | None = None,
        trace_store: JsonlStore | None = None,
    ) -> None:
        self.classifier = classifier or ClassifierAgent()
        self.dispatch_agent = dispatch_agent or DispatchAgent()
        self.reject_reason_agent = reject_reason_agent or RejectReasonAgent()
        self.retrieval_agent = retrieval_agent or RetrievalAgent()
        self.reply_agent = reply_agent or ReplyAgent()
        self.trace_store = trace_store or JsonlStore(settings.traces_path)

    def analyze(self, request: ComplaintAnalyzeRequest) -> AnalyzeResponse:
        trace_id = str(uuid4())
        steps: list[AgentStep] = []
        review_reasons: list[str] = []

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
                "problem_text": clean_request.problem_text,
                "has_incident_location": bool(clean_request.incident_location),
                "has_enterprise_address": bool(clean_request.enterprise_address),
            }

        with agent_step(steps, "classify", {"problem_text": clean_request.problem_text[:120]}) as step:
            classification, classifier_review_reasons = self.classifier.classify(clean_request)
            review_reasons.extend(classifier_review_reasons)
            step["output_summary"] = classification.model_dump(mode="json")
            step["confidence"] = classification.confidence

        dispatch = None
        retrieval_hits = []
        reject_reason_suggestion = None
        invalid_input = "invalid_input" in classification.evidence_fields
        has_reject_reason = classification.reason_type != ReasonType.UNKNOWN
        should_suggest_reject_reason = not invalid_input and (
            not classification.is_market
            or has_reject_reason
        )
        if should_suggest_reject_reason:
            with agent_step(
                steps,
                "reject_reason",
                {"reason_type": classification.reason_type, "accept_suggestion": classification.accept_suggestion},
            ) as step:
                reject_reason_suggestion = self.reject_reason_agent.suggest(clean_request, classification)
                step["output_summary"] = reject_reason_suggestion.model_dump(mode="json") if reject_reason_suggestion else {}
                step["confidence"] = reject_reason_suggestion.confidence if reject_reason_suggestion else None
                if reject_reason_suggestion and reject_reason_suggestion.needs_review:
                    review_reasons.append(reject_reason_suggestion.note or "不受理原因建议需人工确认")

        if classification.is_market and not invalid_input and not has_reject_reason:
            with agent_step(
                steps,
                "dispatch",
                {"address": " ".join(filter(None, [clean_request.incident_location, clean_request.enterprise_address]))[:160]},
            ) as step:
                dispatch = self.dispatch_agent.dispatch(clean_request)
                step["output_summary"] = dispatch.model_dump(mode="json")
                step["confidence"] = dispatch.confidence
                if dispatch.needs_review:
                    review_reasons.append("分派结果置信度不足或使用默认所")
        elif not invalid_input and (not classification.is_market or has_reject_reason):
            if classification.reason_type == ReasonType.ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY:
                review_reasons.append("职责外事项需人工确认转办部门")
            with agent_step(steps, "retrieve", {"reason_type": classification.reason_type}) as step:
                retrieval_hits = self.retrieval_agent.retrieve(classification.reason_type, clean_request.problem_text)
                retrieval_status = dict(getattr(self.retrieval_agent, "last_status", {}))
                step["output_summary"] = {
                    "hits": [hit.model_dump(mode="json") for hit in retrieval_hits],
                    "retrieval_status": retrieval_status,
                }
                step["confidence"] = retrieval_hits[0].score if retrieval_hits else None
                step["degraded"] = bool(retrieval_status.get("degraded"))
                if retrieval_status.get("fallback_reason"):
                    review_reasons.append(str(retrieval_status["fallback_reason"]))

        with agent_step(steps, "reply", {"is_market": classification.is_market}) as step:
            reply = self.reply_agent.draft(classification, dispatch, retrieval_hits)
            step["output_summary"] = reply.model_dump(mode="json")
            step["confidence"] = 0.9 if reply.validation_passed else 0.45
            if not reply.validation_passed or reply.fallback_reason:
                review_reasons.append(reply.fallback_reason or "回复生成降级")

        with agent_step(steps, "validate", {}) as step:
            if classification.confidence < settings.low_confidence_threshold:
                review_reasons.append("分类置信度低")
            if classification.accept_suggestion.value == "REVIEW":
                review_reasons.append("分类建议为人工复核")
            review_reasons = list(dict.fromkeys(reason for reason in review_reasons if reason))
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
