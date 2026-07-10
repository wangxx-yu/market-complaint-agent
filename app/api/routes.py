from __future__ import annotations

import json
from argparse import Namespace
from collections import Counter
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.agents.langgraph_orchestrator import LangGraphOrchestrator
from app.agents.llm_agent import LLMAgentOrchestrator, OllamaClient
from app.agents.orchestrator import Orchestrator
from app.agents.reply import ReplyAgent
from app.agents.debate import DebateOrchestrator
from app.agents.retrieval import RetrievalAgent
from app.core.training_config import ACCEPT_MODEL_DIR
from app.core.enums import AcceptSuggestion, DecisionSource, ReasonType
from app.core.config import settings
from app.core.schemas import (
    AnalyzeResponse,
    ClassificationResult,
    ComplaintAnalyzeRequest,
    LawDocumentResponse,
    LawFullTextResponse,
    RagSearchRequest,
    RagSearchResponse,
    ReviewConfirmRequest,
    ReviewConfirmResponse,
)
from app.core.storage import JsonlStore
from app.tools import (
    export_reject_reason_review_data,
    export_review_training_data,
    mine_accept_rule_candidates,
    replay_accept_model,
    simulate_accept_rule_candidates,
)

router = APIRouter(prefix="/api/v1")
orchestrator = LangGraphOrchestrator() if settings.orchestrator_backend == "langgraph" else Orchestrator()
llm_orchestrator = LLMAgentOrchestrator(
    llm=OllamaClient(base_url=settings.ollama_base_url, model=settings.ollama_model),
)
review_store = JsonlStore(settings.reviews_path)
rag_retrieval_agent = RetrievalAgent()
rag_reply_agent = ReplyAgent()


def build_review_stats() -> dict:
    rows = review_store.all()
    total = 0
    accepted = 0
    rejected = 0
    unknown = 0
    reason_counts: Counter[str] = Counter()
    office_counts: Counter[str] = Counter()

    for row in rows:
        payload = row.get("review", {})
        if not isinstance(payload, dict):
            continue
        total += 1
        is_market = payload.get("is_market")
        if is_market is True:
            accepted += 1
            office_name = payload.get("office_name")
            if office_name:
                office_counts[str(office_name)] += 1
        elif is_market is False:
            rejected += 1
            reason_type = payload.get("reason_type") or "UNKNOWN"
            reason_counts[str(reason_type)] += 1
        else:
            unknown += 1

    return {
        "total": total,
        "accepted": accepted,
        "rejected": rejected,
        "unknown": unknown,
        "reason_counts": dict(reason_counts.most_common()),
        "office_counts": dict(office_counts.most_common()),
    }


def build_system_status() -> dict:
    law_documents = rag_retrieval_agent.list_law_documents()
    knowledge_entries = len(rag_retrieval_agent.entries)
    chroma_available = bool(settings.use_chroma_retrieval and rag_retrieval_agent._get_chroma_collection() is not None)
    return {
        "orchestrator_backend": settings.orchestrator_backend,
        "orchestrator_class": orchestrator.__class__.__name__,
        "rag": {
            "use_chroma": settings.use_chroma_retrieval,
            "chroma_available": chroma_available,
            "embedding_provider": settings.embedding_provider,
            "sentence_transformer_model": settings.sentence_transformer_model,
            "knowledge_entries": knowledge_entries,
            "law_documents": len(law_documents),
            "chroma_dir": settings.chroma_dir.as_posix(),
            "knowledge_base_path": settings.knowledge_base_path.as_posix(),
        },
        "models": {
            "accept_model_dir": ACCEPT_MODEL_DIR.as_posix(),
            "accept_model_exists": (ACCEPT_MODEL_DIR / "accept_model.joblib").exists(),
            "reject_reason_model_dir": settings.reject_reason_model_dir.as_posix(),
            "reject_reason_model_exists": (settings.reject_reason_model_dir / "reject_reason_model.joblib").exists(),
        },
        "runtime": {
            "max_text_chars": settings.max_text_chars,
            "low_confidence_threshold": settings.low_confidence_threshold,
            "reviews_path": settings.reviews_path.as_posix(),
            "traces_path": settings.traces_path.as_posix(),
        },
    }


@router.post("/complaints/analyze", response_model=AnalyzeResponse)
def analyze_complaint(request: ComplaintAnalyzeRequest) -> AnalyzeResponse:
    return orchestrator.analyze(request)


@router.post("/complaints/analyze-llm", response_model=AnalyzeResponse)
async def analyze_complaint_llm(request: ComplaintAnalyzeRequest) -> AnalyzeResponse:
    """ReAct-pattern LLM Agent: 使用大模型进行工具调用推理分析投诉。

    需要本地 Ollama 服务运行中（默认 qwen2.5:7b）。
    如果 LLM 不可用，会自动降级到规则 Agent。
    """
    return await llm_orchestrator.analyze(request)


@router.post("/complaints/analyze-llm/stream")
async def analyze_complaint_llm_stream(request: ComplaintAnalyzeRequest):
    """SSE streaming: LLM Agent 实时推理流。

    事件类型:
    - start: 分析开始，携带 trace_id
    - thinking: LLM 正在推理
    - token: LLM 输出的文本片段
    - tool_call_delta: 工具调用参数流
    - tool_result: 工具执行结果
    - final_answer: LLM 最终回复
    - error: 出错信息
    - done: 分析完成
    """

    async def event_generator():
        try:
            async for chunk in llm_orchestrator.analyze_stream(request):
                event = chunk.get("event", "unknown")
                data = chunk.get("data", {})
                yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'message': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )



@router.post("/complaints/debate")
def debate_complaint(request: ComplaintAnalyzeRequest) -> dict:
    """Multi-Agent Debate: 双 Agent 独立分类 + 交叉验证。

    两个 Agent 独立分析同一投诉，结果一致则采纳，分歧则触发仲裁并标记人工复核。
    """
    from app.core.schemas import ClassificationResult

    verdict = debate_orchestrator.debate(request)
    classification = verdict.classification
    return {
        "trace_id": "debate_" + classification.reason_type.value,
        "classification": classification.model_dump(mode="json"),
        "agent_a": verdict.agent_a_result.model_dump(mode="json"),
        "agent_b": verdict.agent_b_result.model_dump(mode="json"),
        "consensus": verdict.consensus,
        "debate_required": verdict.debate_required,
        "judge_reason": verdict.judge_reason,
        "review_required": verdict.debate_required or classification.accept_suggestion == "REVIEW",
        "review_reasons": verdict.review_reasons,
    }


@router.post("/reviews/{trace_id}/confirm", response_model=ReviewConfirmResponse)
def confirm_review(trace_id: str, request: ReviewConfirmRequest) -> ReviewConfirmResponse:
    trace = orchestrator.trace_store.find_by_key("trace_id", trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="trace_id not found")
    saved_at = datetime.now(timezone.utc)
    review_store.append(
        {
            "trace_id": trace_id,
            "saved_at": saved_at.isoformat(),
            "review": request.model_dump(mode="json"),
        }
    )
    return ReviewConfirmResponse(trace_id=trace_id, saved=True, saved_at=saved_at)


@router.get("/reviews/stats")
def review_stats() -> dict:
    return build_review_stats()


@router.get("/reviews/pending")
def list_pending_reviews(limit: int = 20, offset: int = 0) -> dict:
    """返回待人工复核的投诉列表。"""
    traces = orchestrator.trace_store.all()
    reviewed_ids = {r.get("trace_id") for r in review_store.all()}
    pending = [
        {
            "trace_id": t.get("trace_id"),
            "problem_text": t.get("request", {}).get("problem_text", "")[:100],
            "accept_suggestion": t.get("classification", {}).get("accept_suggestion"),
            "reason_type": t.get("classification", {}).get("reason_type"),
            "confidence": t.get("classification", {}).get("confidence"),
            "review_reasons": t.get("review_reasons", []),
        }
        for t in traces
        if t.get("review_reasons") and t.get("trace_id") not in reviewed_ids
    ]
    total = len(pending)
    return {"total": total, "items": pending[offset:offset + limit]}


@router.get("/reviews/{trace_id}")
def get_review_detail(trace_id: str) -> dict:
    """获取单条复核详情，包含 trace 和 review 记录。"""
    trace = orchestrator.trace_store.find_by_key("trace_id", trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="trace_id not found")
    review = review_store.find_by_key("trace_id", trace_id)
    return {
        "trace_id": trace_id,
        "trace": trace,
        "review": review.get("review") if review else None,
        "is_reviewed": review is not None,
    }


@router.post("/reviews/{trace_id}/reject", response_model=ReviewConfirmResponse)
def reject_review(trace_id: str, request: ReviewConfirmRequest) -> ReviewConfirmResponse:
    """驳回系统建议，记录人工判断。"""
    trace = orchestrator.trace_store.find_by_key("trace_id", trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="trace_id not found")
    saved_at = datetime.now(timezone.utc)
    review_store.append(
        {
            "trace_id": trace_id,
            "saved_at": saved_at.isoformat(),
            "decision": "REJECT",
            "review": request.model_dump(mode="json"),
        }
    )
    return ReviewConfirmResponse(trace_id=trace_id, saved=True, saved_at=saved_at)


@router.get("/system/status")
def system_status() -> dict:
    return build_system_status()

@router.get("/metrics")
def metrics() -> dict:
    """Prometheus-compatible metrics endpoint for Agent observability."""
    from app.core.metrics import (
        agent_calls_total, agent_errors_total, agent_latency_seconds,
        reviews_total, degradation_total,
    )

    def counter_to_text(counter, metric_name):
        lines = [f"# HELP {metric_name} {counter.help}", f"# TYPE {metric_name} counter"]
        for labels, value in counter.labels.items():
            if labels:
                lines.append(f"{metric_name}{{{labels}}} {value}")
            else:
                lines.append(f"{metric_name} {value}")
        return chr(10).join(lines)

    def hist_to_text(hist, metric_name):
        lines = [f"# HELP {metric_name} {hist.help}", f"# TYPE {metric_name} histogram"]
        for v in hist._values[-100:]:
            lines.append(f"{metric_name}_seconds {v}")
        return chr(10).join(lines)

    parts = [
        counter_to_text(agent_calls_total, "mca_agent_calls_total"),
        counter_to_text(agent_errors_total, "mca_agent_errors_total"),
        hist_to_text(agent_latency_seconds, "mca_agent_latency_seconds"),
        counter_to_text(reviews_total, "mca_reviews_total"),
        counter_to_text(degradation_total, "mca_degradation_total"),
    ]
    return {"text": chr(10).join(parts)}



@router.post("/reviews/export-training")
def export_review_training() -> dict:
    accept_base_csv = export_review_training_data.ACCEPT_TRAINING_CSV
    reject_reason_base_csv = export_reject_reason_review_data.DEFAULT_BASE_CSV
    accept_summary = export_review_training_data.export(
        Namespace(
            base_csv=str(accept_base_csv),
            traces=str(settings.traces_path),
            reviews=str(settings.reviews_path),
            out_dir="data/training",
            output_name="accept_training_from_reviews.csv",
            include_base=accept_base_csv.exists(),
        )
    )
    reject_reason_summary = export_reject_reason_review_data.export(
        Namespace(
            base_csv=str(reject_reason_base_csv),
            traces=str(settings.traces_path),
            reviews=str(settings.reviews_path),
            out_dir="data/training",
            output_name="reject_reason_training_from_reviews.csv",
            include_base=reject_reason_base_csv.exists(),
        )
    )
    return {
        "accept_training": accept_summary,
        "reject_reason_training": reject_reason_summary,
    }


@router.post("/evaluation/replay-accept")
def replay_accept_evaluation() -> dict:
    return replay_accept_model.replay(
        Namespace(
            csv=str(replay_accept_model.choose_default_csv()),
            text_col="text",
            label_col="label",
            out_dir="data/evaluation",
            min_chars=5,
            limit=0,
        )
    )


@router.post("/evaluation/mine-accept-rules")
def mine_accept_rules() -> dict:
    return mine_accept_rule_candidates.analyze(
        Namespace(
            csv="data/evaluation/accept_replay_review.csv",
            out_dir="data/evaluation",
            min_support=8,
            max_size=3,
            accept_rate=0.9,
            reject_rate=0.75,
            min_accept_count=6,
            min_reject_count=6,
        )
    )


@router.post("/evaluation/simulate-accept-rules")
def simulate_accept_rules() -> dict:
    return simulate_accept_rule_candidates.simulate(
        Namespace(
            review_csv="data/evaluation/accept_replay_review.csv",
            candidates_csv="data/evaluation/accept_rule_candidates_high_accept.csv",
            out_dir="data/evaluation",
            min_support=8,
            min_accept_rate=0.9,
        )
    )


@router.post("/rag/reject-reply", response_model=RagSearchResponse)
def search_reject_reply_reference(request: RagSearchRequest) -> RagSearchResponse:
    reason_type = request.reason_type
    if reason_type == ReasonType.UNKNOWN:
        reason_type = ReasonType.ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY
    hits = rag_retrieval_agent.retrieve(reason_type, request.query, top_k=request.top_k)
    classification = ClassificationResult(
        is_market=False,
        accept_suggestion=AcceptSuggestion.REVIEW,
        reason_type=reason_type,
        confidence=0.8,
        decision_source=DecisionSource.RAG_LLM,
        evidence_fields=[],
    )
    reply = rag_reply_agent.draft(classification, None, hits)
    return RagSearchResponse(
        query=request.query,
        mode="reject_reply",
        hits=hits,
        reply_suggestion=reply.text,
    )


@router.post("/rag/law-search", response_model=RagSearchResponse)
def search_law_knowledge(request: RagSearchRequest) -> RagSearchResponse:
    hits = rag_retrieval_agent.retrieve(request.reason_type, request.query, top_k=request.top_k)
    return RagSearchResponse(
        query=request.query,
        mode="law_search",
        hits=hits,
        reply_suggestion=None,
    )


@router.get("/rag/laws", response_model=LawDocumentResponse)
def list_law_documents() -> LawDocumentResponse:
    return LawDocumentResponse(documents=rag_retrieval_agent.list_law_documents())


@router.get("/rag/laws/{doc_id}", response_model=LawFullTextResponse)
def get_law_full_text(doc_id: str) -> LawFullTextResponse:
    document = rag_retrieval_agent.get_law_full_text(doc_id)
    if document is None:
        raise HTTPException(status_code=404, detail="law document not found")
    return LawFullTextResponse(**document)


@router.get("/traces/{trace_id}")
def get_trace(trace_id: str) -> dict:
    trace = orchestrator.trace_store.find_by_key("trace_id", trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="trace_id not found")
    return trace
