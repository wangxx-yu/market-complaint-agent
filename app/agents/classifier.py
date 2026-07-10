"""投诉分类 Agent — 规则引擎优先 + BERT 模型兜底。"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import joblib

from app.agents.rule_engine import RuleEngine
from app.core.config import settings
from app.core.enums import AcceptSuggestion, DecisionSource, ReasonType
from app.core.schemas import ClassificationResult, ComplaintAnalyzeRequest
from app.core.text import normalize_text
from app.core.training_config import ACCEPT_MODEL_DIR


class ClassifierAgent:
    # 输入有效性校验用关键词（非业务规则，不抽到 YAML）
    complaint_clues = [
        "投诉", "举报", "购买", "消费", "商户", "商家", "店",
        "超市", "药店", "餐馆", "酒店", "退款", "退货", "赔偿",
        "质量", "价格", "食品", "过期", "发票", "服务", "不予",
        "协调", "处理", "诉求", "问题", "花费", "支付", "充值",
        "会员", "售后",
    ]

    # 预付卡受理的上下文关键词——与 prepaid 规则联合判断
    _PREPAID_CONTEXT_KEYWORDS = ["退费", "退款", "退卡", "消费", "会员卡", "年卡"]

    def __init__(
        self,
        model_dir: Path | None = None,
        accept_threshold: float = 0.65,
        reject_threshold: float = 0.20,
        rule_engine: RuleEngine | None = None,
    ) -> None:
        self.model_dir = model_dir or ACCEPT_MODEL_DIR
        self.model_path = self.model_dir / "accept_model.joblib"
        self.accept_threshold = accept_threshold
        self.reject_threshold = reject_threshold
        self._model: Any | None = None
        self.rule_engine = rule_engine or RuleEngine()

    @property
    def model(self) -> Any | None:
        if self._model is None and self.model_path.exists():
            self._model = joblib.load(self.model_path)
        return self._model

    def classify(self, request: ComplaintAnalyzeRequest) -> tuple[ClassificationResult, list[str]]:
        text = normalize_text(" ".join(filter(None, [request.problem_text, request.appeal_text])))
        review_reasons: list[str] = []

        # 1. 输入有效性检测
        invalid_reason = self._invalid_input_reason(text)
        if invalid_reason:
            review_reasons.append(invalid_reason)
            return (
                ClassificationResult(
                    is_market=True,
                    accept_suggestion=AcceptSuggestion.REVIEW,
                    reason_type=ReasonType.UNKNOWN,
                    confidence=0.2,
                    decision_source=DecisionSource.RULE,
                    evidence_fields=["invalid_input"],
                ),
                review_reasons,
            )

        # 2. 敏感词检测
        sensitive_result = self.rule_engine.match_sensitive(text)
        if sensitive_result.has_match:
            match = sensitive_result.highest_priority_match
            review_reasons.append(f"命中敏感词({match.rule_name})，需人工复核")

        # 3. 职责外/转办规则（有 suggest_department → is_market=False）
        reject_result = self.rule_engine.match_reject(text)
        transfer_matches = [m for m in reject_result.matches if m.suggest_department]
        if transfer_matches:
            match = transfer_matches[0]
            review_reasons.append(
                f"命中{match.rule_name}职责外规则，建议转{match.suggest_department}，需人工确认"
            )
            return (
                ClassificationResult(
                    is_market=False,
                    accept_suggestion=AcceptSuggestion.REVIEW,
                    reason_type=ReasonType.ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY,
                    confidence=0.88,
                    decision_source=DecisionSource.RULE,
                    evidence_fields=[*match.matched_keywords, f"suggest_department={match.suggest_department}"],
                ),
                review_reasons,
            )

        # 4. 第16条不予受理规则（有 reason_type ≠ UNKNOWN → is_market=True, REVIEW）
        article16_matches = [
            m for m in reject_result.matches
            if not m.suggest_department and m.reason_type != "UNKNOWN"
        ]
        if article16_matches:
            match = article16_matches[0]
            confidence = 0.86 if match.reason_type != ReasonType.ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY else 0.78
            review_reasons.append("命中第16条可能不予受理规则，需人工确认")
            return (
                ClassificationResult(
                    is_market=True,
                    accept_suggestion=AcceptSuggestion.REVIEW,
                    reason_type=ReasonType(match.reason_type),
                    confidence=confidence,
                    decision_source=DecisionSource.RULE,
                    evidence_fields=match.matched_keywords,
                ),
                review_reasons,
            )

        # 5. 预付卡/健身退费受理规则（带上下文校验）
        accept_result = self.rule_engine.match_accept(text)
        prepaid_matches = [m for m in accept_result.matches if m.rule_id == "accept_prepaid_service"]
        if prepaid_matches:
            prepaid_kws = prepaid_matches[0].matched_keywords
            if any(kw in text for kw in self._PREPAID_CONTEXT_KEYWORDS):
                return (
                    ClassificationResult(
                        is_market=True,
                        accept_suggestion=AcceptSuggestion.ACCEPT,
                        reason_type=ReasonType.UNKNOWN,
                        confidence=min(0.92, 0.78 + len(prepaid_kws) * 0.02),
                        decision_source=DecisionSource.RULE,
                        evidence_fields=prepaid_kws[:8],
                    ),
                    review_reasons,
                )

        # 6. 模型分类
        model_result = self._classify_with_model(text)
        if model_result is not None:
            if model_result.accept_suggestion == AcceptSuggestion.REVIEW:
                review_reasons.append("模型受理概率处于人工复核区间")
            return model_result, review_reasons

        # 7. 市场监管关键词兜底
        market_matches = [m for m in accept_result.matches if m.rule_id == "accept_market_general"]
        if market_matches:
            market_kws = market_matches[0].matched_keywords
            review_reasons.append("模型文件不可用，使用市场监管关键词规则")
            return (
                ClassificationResult(
                    is_market=True,
                    accept_suggestion=AcceptSuggestion.ACCEPT,
                    reason_type=ReasonType.UNKNOWN,
                    confidence=min(0.94, 0.72 + len(market_kws) * 0.03),
                    decision_source=DecisionSource.RULE,
                    evidence_fields=market_kws[:8],
                ),
                review_reasons,
            )

        # 8. 默认人工复核
        review_reasons.append("未命中明确职责规则")
        return (
            ClassificationResult(
                is_market=True,
                accept_suggestion=AcceptSuggestion.REVIEW,
                reason_type=ReasonType.UNKNOWN,
                confidence=0.58,
                decision_source=DecisionSource.RULE,
                evidence_fields=[],
            ),
            review_reasons,
        )

    def _invalid_input_reason(self, text: str) -> str | None:
        if len(text) < 8:
            return "输入内容过短，请补充商户、地点、具体问题和诉求。"
        if not re.search(r"[\u4e00-\u9fff]", text):
            return "输入内容缺少中文投诉描述，请补充有效投诉内容。"
        # 用 accept + reject 全部规则的关键词做联合校验
        accept_rules = self.rule_engine.loader.load("accept")
        reject_rules = self.rule_engine.loader.load("reject")
        all_keywords = set(self.complaint_clues)
        for rules in [accept_rules, reject_rules]:
            for rule in rules:
                for kw in rule.get("keywords", []):
                    all_keywords.add(kw)
        if not any(clue in text for clue in all_keywords):
            return "输入内容缺少投诉要素，请补充商户、地点、具体问题和诉求。"
        return None

    def _classify_with_model(self, text: str) -> ClassificationResult | None:
        model = self.model
        if model is None:
            return None
        try:
            classes = list(model.classes_)
            accept_index = classes.index(1)
            prob_accept = float(model.predict_proba([text])[0][accept_index])
        except Exception:
            return None

        if prob_accept >= self.accept_threshold:
            return ClassificationResult(
                is_market=True,
                accept_suggestion=AcceptSuggestion.ACCEPT,
                reason_type=ReasonType.UNKNOWN,
                confidence=prob_accept,
                decision_source=DecisionSource.MODEL,
                evidence_fields=[f"prob_accept={prob_accept:.4f}", f"model={self.model_path.as_posix()}"],
            )
        return ClassificationResult(
            is_market=True,
            accept_suggestion=AcceptSuggestion.REVIEW,
            reason_type=ReasonType.UNKNOWN,
            confidence=max(prob_accept, 1 - prob_accept),
            decision_source=DecisionSource.MODEL,
            evidence_fields=[f"prob_accept={prob_accept:.4f}", f"model={self.model_path.as_posix()}"],
        )
