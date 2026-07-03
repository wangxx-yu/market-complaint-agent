from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from app.core.config import settings
from app.core.enums import DecisionSource, ReasonType
from app.core.schemas import ClassificationResult, ComplaintAnalyzeRequest, RejectReasonSuggestion
from app.core.text import normalize_text


class RejectReasonAgent:
    def __init__(
        self,
        model_dir: Path | None = None,
        high_confidence_threshold: float | None = None,
    ) -> None:
        self.model_dir = model_dir or settings.reject_reason_model_dir
        self.model_path = self.model_dir / "reject_reason_model.joblib"
        self.high_confidence_threshold = high_confidence_threshold or settings.reject_reason_high_confidence_threshold
        self._model: Any | None = None

    @property
    def model(self) -> Any | None:
        if self._model is None and self.model_path.exists():
            self._model = joblib.load(self.model_path)
        return self._model

    def suggest(
        self,
        request: ComplaintAnalyzeRequest,
        classification: ClassificationResult,
    ) -> RejectReasonSuggestion | None:
        if classification.reason_type != ReasonType.UNKNOWN:
            return RejectReasonSuggestion(
                reason_type=classification.reason_type,
                confidence=classification.confidence,
                decision_source=classification.decision_source,
                evidence_fields=classification.evidence_fields,
                needs_review=True,
                note="分类规则已给出不受理原因，人工确认后保存。",
            )

        model = self.model
        if model is None:
            return RejectReasonSuggestion(
                reason_type=ReasonType.UNKNOWN,
                confidence=0.0,
                decision_source=DecisionSource.FALLBACK,
                evidence_fields=["reject_reason_model_missing"],
                needs_review=True,
                note="不受理原因模型文件不可用，请人工选择原因。",
            )

        text = normalize_text(" ".join(filter(None, [request.problem_text, request.appeal_text])))
        try:
            probabilities = model.predict_proba([text])[0]
            classes = list(model.classes_)
        except Exception:
            return RejectReasonSuggestion(
                reason_type=ReasonType.UNKNOWN,
                confidence=0.0,
                decision_source=DecisionSource.FALLBACK,
                evidence_fields=["reject_reason_model_error"],
                needs_review=True,
                note="不受理原因模型预测失败，请人工选择原因。",
            )

        best_index = max(range(len(probabilities)), key=lambda index: float(probabilities[index]))
        reason_type = ReasonType(classes[best_index])
        confidence = float(probabilities[best_index])
        prob_fields = [f"{label}={float(probabilities[index]):.4f}" for index, label in enumerate(classes)]
        needs_review = confidence < self.high_confidence_threshold
        note = "不受理原因模型建议，置信度较高，仍需人工确认。"
        if needs_review:
            note = f"不受理原因模型置信度低于{self.high_confidence_threshold:.0%}，请人工选择原因。"

        return RejectReasonSuggestion(
            reason_type=reason_type,
            confidence=confidence,
            decision_source=DecisionSource.MODEL,
            evidence_fields=[f"model={self.model_path.as_posix()}", *prob_fields],
            needs_review=needs_review,
            note=note,
        )
