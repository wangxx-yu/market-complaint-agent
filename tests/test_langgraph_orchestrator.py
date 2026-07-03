from app.agents.langgraph_orchestrator import LangGraphOrchestrator
from app.agents.retrieval import RetrievalAgent
from app.core.enums import AcceptSuggestion, DecisionSource, ReasonType
from app.core.schemas import ClassificationResult
from app.core.schemas import ComplaintAnalyzeRequest


class FailingClassifier:
    def classify(self, request: ComplaintAnalyzeRequest):
        raise RuntimeError("classifier unavailable")


class AcceptingClassifier:
    def classify(self, request: ComplaintAnalyzeRequest):
        return (
            ClassificationResult(
                is_market=True,
                accept_suggestion=AcceptSuggestion.ACCEPT,
                reason_type=ReasonType.UNKNOWN,
                confidence=0.91,
                decision_source=DecisionSource.RULE,
                evidence_fields=["test_accept"],
            ),
            [],
        )


class FailingDispatchAgent:
    def dispatch(self, request: ComplaintAnalyzeRequest):
        raise RuntimeError("dispatch unavailable")


class FailingReplyAgent:
    def draft(self, classification, dispatch, hits):
        raise RuntimeError("reply unavailable")


def test_langgraph_orchestrator_accept_path_dispatches() -> None:
    response = LangGraphOrchestrator().analyze(
        ComplaintAnalyzeRequest(
            problem_text="商户名称：金石健身俱乐部，市民办理年卡会员卡后产生退费纠纷，请求协调处理。",
            enterprise_address="青铜峡市新百CCMALL四楼",
        )
    )

    step_names = [step.name for step in response.agent_steps]
    assert response.classification.is_market is True
    assert response.dispatch is not None
    assert response.reply_draft.template_id == "accept_no_return_reply"
    assert "dispatch" in step_names
    assert "retrieve" not in step_names


def test_langgraph_orchestrator_reject_path_retrieves_basis() -> None:
    response = LangGraphOrchestrator().analyze(
        ComplaintAnalyzeRequest(problem_text="小区物业费和停车位产权纠纷，要求市场监管部门处理。")
    )

    step_names = [step.name for step in response.agent_steps]
    assert response.dispatch is None
    assert response.retrieval_hits
    assert response.reject_reason_suggestion is not None
    assert "reject_reason" in step_names
    assert "retrieve" in step_names
    assert "reply" in step_names
    retrieve_step = next(step for step in response.agent_steps if step.name == "retrieve")
    assert "retrieval_status" in retrieve_step.output_summary


def test_langgraph_orchestrator_marks_retrieval_fallback_degraded() -> None:
    retrieval_agent = RetrievalAgent()
    retrieval_agent._chroma_failed = True
    response = LangGraphOrchestrator(retrieval_agent=retrieval_agent).analyze(
        ComplaintAnalyzeRequest(problem_text="小区物业费和停车位产权纠纷，要求市场监管部门处理。")
    )

    retrieve_step = next(step for step in response.agent_steps if step.name == "retrieve")
    assert retrieve_step.degraded is True
    assert retrieve_step.output_summary["retrieval_status"]["retrieval_source"] == "RULE_FALLBACK"
    assert "回退关键词检索" in retrieve_step.output_summary["retrieval_status"]["fallback_reason"]


def test_langgraph_orchestrator_classifier_failure_degrades_to_review() -> None:
    response = LangGraphOrchestrator(classifier=FailingClassifier()).analyze(
        ComplaintAnalyzeRequest(problem_text="商户食品变质，要求协调处理退款。")
    )

    classify_step = next(step for step in response.agent_steps if step.name == "classify")
    reply_step = next(step for step in response.agent_steps if step.name == "reply")
    assert response.review_required is True
    assert response.classification.decision_source == DecisionSource.FALLBACK
    assert response.classification.accept_suggestion == AcceptSuggestion.REVIEW
    assert classify_step.degraded is True
    assert "classifier unavailable" in classify_step.error
    assert reply_step.degraded is True
    assert response.reply_draft.decision_source == DecisionSource.FALLBACK
    assert any("是否受理模型异常" in reason for reason in response.review_reasons)


def test_langgraph_orchestrator_dispatch_failure_degrades_to_manual_office() -> None:
    response = LangGraphOrchestrator(
        classifier=AcceptingClassifier(),
        dispatch_agent=FailingDispatchAgent(),
    ).analyze(ComplaintAnalyzeRequest(problem_text="商户食品变质，要求协调处理退款。"))

    dispatch_step = next(step for step in response.agent_steps if step.name == "dispatch")
    assert response.dispatch is not None
    assert response.dispatch.office_name == "待人工选择"
    assert response.dispatch.needs_review is True
    assert response.review_required is True
    assert dispatch_step.degraded is True
    assert "dispatch unavailable" in dispatch_step.error
    assert "分派 Agent 异常，需人工选择市场监管所" in response.review_reasons


def test_langgraph_orchestrator_reply_failure_returns_fallback_draft() -> None:
    response = LangGraphOrchestrator(
        classifier=AcceptingClassifier(),
        reply_agent=FailingReplyAgent(),
    ).analyze(ComplaintAnalyzeRequest(problem_text="商户食品变质，要求协调处理退款。"))

    reply_step = next(step for step in response.agent_steps if step.name == "reply")
    assert response.review_required is True
    assert response.reply_draft.decision_source == DecisionSource.FALLBACK
    assert response.reply_draft.template_id == "system_fallback"
    assert reply_step.degraded is True
    assert "reply unavailable" in reply_step.error
    assert "回复 Agent 异常，已降级为人工复核提示" in response.review_reasons
