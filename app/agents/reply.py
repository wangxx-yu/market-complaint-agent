from __future__ import annotations

import json
from pathlib import Path

from app.core.config import settings
from app.core.enums import DecisionSource, ReasonType
from app.core.schemas import ClassificationResult, DispatchResult, ReplyDraft, RetrievalHit


class ReplyAgent:
    def __init__(self, templates_path: Path | None = None) -> None:
        self.templates_path = templates_path or settings.reply_templates_path
        self.templates = self._load_templates()

    def _load_templates(self) -> dict[str, str]:
        defaults = {
            "accept_default": "您的投诉已登记。建议由{office_name}进一步核查处理，工作人员将结合事实、证据和相关规定依法办理。",
            "reject_out_of_scope": "经初步判断，该事项可能不属于我局职责范围或本机关无处理权限。建议您向相关主管部门反映，具体部门需结合事实和属地进一步确认。",
            "reject_already_processed": "该同一消费争议可能已由人民法院、仲裁机构、其他行政机关、消费者协会等单位受理或处理。建议您通过原受理渠道查询办理进展。",
            "reject_not_consumer": "该事项可能不属于为生活消费需要购买、使用商品或者接受服务产生的消费者权益争议。建议您通过相应主管部门或司法途径反映。",
            "reject_expired": "该事项可能已超过三年投诉时效。建议您补充争议发生时间、交易凭证等材料后由工作人员复核。",
            "reject_missing_or_false_materials": "该投诉可能缺少被投诉人、具体诉求、事实理由、消费凭证等必要材料，或存在材料真实性疑问。请补充真实、完整材料后再提交。",
            "reject_identity_verify": "该投诉可能存在冒用他人名义或不配合身份核验的情形。请由投诉人本人提交并配合身份核验。",
            "reject_other_legal_reasons": "该事项可能属于法律、法规、规章规定的其他不予受理情形，建议由工作人员结合材料进一步复核。",
        }
        if self.templates_path.exists():
            defaults.update(json.loads(self.templates_path.read_text(encoding="utf-8")))
        return defaults

    def draft(
        self,
        classification: ClassificationResult,
        dispatch: DispatchResult | None,
        hits: list[RetrievalHit],
    ) -> ReplyDraft:
        if "invalid_input" in classification.evidence_fields:
            return ReplyDraft(
                text="请补充有效投诉内容，包括商户名称或地点、具体问题、发生经过和主要诉求后再提交分析。",
                decision_source=DecisionSource.RULE,
                template_id="invalid_input",
            )

        suggested_department = self._suggested_department(classification.evidence_fields, hits)

        should_accept_reply = classification.is_market and classification.reason_type == ReasonType.UNKNOWN
        if should_accept_reply:
            return ReplyDraft(
                text="",
                decision_source=DecisionSource.RULE,
                template_id="accept_no_return_reply",
            )

        template_by_reason = {
            ReasonType.ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY: "reject_out_of_scope",
            ReasonType.ARTICLE16_2_ALREADY_ACCEPTED_OR_PROCESSED: "reject_already_processed",
            ReasonType.ARTICLE16_3_NOT_CONSUMER_DISPUTE: "reject_not_consumer",
            ReasonType.ARTICLE16_4_EXPIRED: "reject_expired",
            ReasonType.ARTICLE16_5_MISSING_OR_FALSE_MATERIALS: "reject_missing_or_false_materials",
            ReasonType.ARTICLE16_6_IMPERSONATION_OR_REFUSE_VERIFY: "reject_identity_verify",
            ReasonType.ARTICLE16_7_OTHER_LEGAL_REASONS: "reject_other_legal_reasons",
        }
        template_id = template_by_reason.get(classification.reason_type, "reject_out_of_scope")
        text = self.templates[template_id]
        if suggested_department:
            text = f"经初步判断，该事项可能不属于我局职责范围或本机关无处理权限。建议您向{suggested_department}反映，由其结合事实和职责核实处理。"
        if hits:
            text = f"{text} 依据参考：{hits[0].title}。"
        if classification.reason_type == ReasonType.ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY and hits:
            if self._validate_llm_like_output(text):
                return ReplyDraft(text=text, decision_source=DecisionSource.RAG_LLM, template_id=template_id)
            return ReplyDraft(
                text=self.templates["reject_out_of_scope"],
                decision_source=DecisionSource.FALLBACK,
                template_id="reject_out_of_scope",
                validation_passed=False,
                fallback_reason="职责外回复未通过合规关键词校验",
            )
        source = DecisionSource.RAG_LLM if hits else DecisionSource.RULE
        return ReplyDraft(text=text, decision_source=source, template_id=template_id)

    @staticmethod
    def _validate_llm_like_output(text: str) -> bool:
        required_any = ["建议您向", "不属于我局职责", "不属于市场监督管理部门职责"]
        return any(keyword in text for keyword in required_any)

    @staticmethod
    def _suggested_department(evidence_fields: list[str], hits: list[RetrievalHit]) -> str | None:
        prefix = "suggest_department="
        for field in evidence_fields:
            if field.startswith(prefix):
                return field.removeprefix(prefix)
        for hit in hits:
            if hit.suggested_department:
                return hit.suggested_department
        return None
