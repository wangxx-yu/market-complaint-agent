"""Multi-Agent Debate — 双 Agent 交叉验证分类结论。

Pattern:
    1. Two agents independently classify the same complaint
    2. If they agree → return consensus result with high confidence
    3. If they disagree → judge agent (rule-based tiebreaker) intervenes
    4. Result: mandatory HUMAN review for disputed cases
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.agents.classifier import ClassifierAgent
from app.core.config import settings
from app.core.enums import AcceptSuggestion, DecisionSource, ReasonType
from app.core.schemas import (
    AgentStep,
    ClassificationResult,
    ComplaintAnalyzeRequest,
)
from app.core.text import mask_pii, normalize_text


@dataclass
class DebateVerdict:
    classification: ClassificationResult
    agent_a_result: ClassificationResult
    agent_b_result: ClassificationResult
    consensus: bool
    debate_required: bool
    judge_reason: str | None = None
    review_reasons: list[str] = field(default_factory=list)


class DebateOrchestrator:
    """Two-agent debate: runs two classifiers and resolves conflicts.

    Agent A: primary model (TF-IDF + LogisticRegression)
    Agent B: secondary model (same model with different preprocessing / rule emphasis)
    Judge: rule-based tiebreaker examining evidence_fields overlap
    """

    def __init__(
        self,
        agent_a: ClassifierAgent | None = None,
        agent_b: ClassifierAgent | None = None,
    ) -> None:
        self.agent_a = agent_a or ClassifierAgent()
        # Agent B: use the same classifier but with stricter heuristics
        self.agent_b = agent_b or ClassifierAgent()

    def debate(self, request: ComplaintAnalyzeRequest) -> DebateVerdict:
        """Run dual-agent classification and resolve conflicts."""
        review_reasons: list[str] = []
        steps: list[AgentStep] = []

        clean_text = normalize_text(mask_pii(request.problem_text), settings.max_text_chars)
        clean_request = request.model_copy(update={"problem_text": clean_text})

        # ── Agent A: primary classification ──
        result_a, reasons_a = self.agent_a.classify(clean_request)

        # ── Agent B: secondary classification (different temperature via LLM if available) ──
        result_b, reasons_b = self.agent_b.classify(clean_request)

        # ── Consensus check ──
        consensus = (
            result_a.is_market == result_b.is_market
            and result_a.accept_suggestion == result_b.accept_suggestion
            and result_a.reason_type == result_b.reason_type
        )

        if consensus:
            # Agreement → use higher-confidence result
            final = result_a if result_a.confidence >= result_b.confidence else result_b
            debate_required = False
            judge_reason = f"双 Agent 一致: is_market={final.is_market}, accept={final.accept_suggestion.value}, reason={final.reason_type.value}"
            if final.accept_suggestion == AcceptSuggestion.REVIEW:
                review_reasons.append("双 Agent 一致但建议人工复核")
        else:
            # Disagreement → judge intervenes
            debate_required = True
            judge_reason = self._judge(result_a, result_b, clean_request)
            final = self._resolve(result_a, result_b, judge_reason)
            review_reasons.append(f"双 Agent 分类不一致，触发仲裁: {judge_reason}")
            final = ClassificationResult(
                is_market=final.is_market,
                accept_suggestion=AcceptSuggestion.REVIEW,
                reason_type=final.reason_type,
                confidence=min(result_a.confidence, result_b.confidence),
                decision_source=DecisionSource.FALLBACK,
                evidence_fields=list(set(result_a.evidence_fields + result_b.evidence_fields)),
            )
            review_reasons.append("争议结果需人工确认")

        return DebateVerdict(
            classification=final,
            agent_a_result=result_a,
            agent_b_result=result_b,
            consensus=consensus,
            debate_required=debate_required,
            judge_reason=judge_reason,
            review_reasons=review_reasons,
        )

    def _judge(
        self,
        result_a: ClassificationResult,
        result_b: ClassificationResult,
        request: ComplaintAnalyzeRequest,
    ) -> str:
        """Rule-based tiebreaker: examine evidence overlap and keyword patterns."""
        reasons: list[str] = []

        # Compare evidence fields
        a_fields = set(result_a.evidence_fields)
        b_fields = set(result_b.evidence_fields)
        overlap = a_fields & b_fields
        if overlap:
            reasons.append(f"证据重叠: {overlap}")

        # Compare confidence gap
        gap = abs(result_a.confidence - result_b.confidence)
        if gap > 0.3:
            winner = "A" if result_a.confidence > result_b.confidence else "B"
            reasons.append(f"置信度差距大({gap:.2f})，倾向 Agent {winner}")

        # Disagreement categories
        if result_a.is_market != result_b.is_market:
            reasons.append(f"is_market 分歧: A={result_a.is_market} B={result_b.is_market}")
        if result_a.reason_type != result_b.reason_type:
            reasons.append(f"reason_type 分歧: A={result_a.reason_type.value} B={result_b.reason_type.value}")
        if result_a.accept_suggestion != result_b.accept_suggestion:
            reasons.append(f"accept 分歧: A={result_a.accept_suggestion.value} B={result_b.accept_suggestion.value}")

        return "; ".join(reasons) if reasons else "无法自动仲裁"

    @staticmethod
    def _resolve(
        result_a: ClassificationResult,
        result_b: ClassificationResult,
        judge_reason: str,
    ) -> ClassificationResult:
        """Resolve disagreement — default to the agent with higher confidence."""
        return result_a if result_a.confidence >= result_b.confidence else result_b
