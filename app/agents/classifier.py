from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import joblib

from app.core.config import settings
from app.core.enums import AcceptSuggestion, DecisionSource, ReasonType
from app.core.schemas import ClassificationResult, ComplaintAnalyzeRequest
from app.core.text import normalize_text
from app.core.training_config import ACCEPT_MODEL_DIR


class ClassifierAgent:
    complaint_clues = [
        "投诉",
        "举报",
        "购买",
        "消费",
        "商户",
        "商家",
        "店",
        "超市",
        "药店",
        "餐馆",
        "酒店",
        "退款",
        "退货",
        "赔偿",
        "质量",
        "价格",
        "食品",
        "过期",
        "发票",
        "服务",
        "不予",
        "协调",
        "处理",
        "诉求",
        "问题",
        "花费",
        "支付",
        "充值",
        "会员",
        "售后",
    ]
    out_scope_keywords = [
        "物业",
        "供暖",
        "自来水",
        "供水",
        "燃气开户",
        "交通事故",
        "医保",
        "社保",
        "教育局",
        "住建局",
        "城管",
        "公安",
        "交警",
        "法院",
        "劳动监察",
    ]
    agriculture_transfer_keywords = [
        "农药",
        "农用机械",
        "农机",
        "农产品生产",
        "牛用",
        "兽用",
        "兽药",
        "兽医",
        "饲料",
        "饲料添加剂",
        "养殖",
        "畜牧",
        "牲畜",
        "牛打架",
        "农业农村局",
    ]
    housing_transfer_keywords = [
        "物业费",
        "物业管理",
        "自行车棚",
        "车棚",
        "公租房",
        "房管局",
        "水费",
        "供热",
        "供暖",
        "住建局",
        "物监办",
    ]
    public_security_transfer_keywords = [
        "派出所",
        "公安",
        "报警",
        "打架",
        "人身伤害",
        "故意伤害",
        "盗窃",
    ]
    tobacco_transfer_keywords = ["香烟", "卷烟", "烟草", "烟草专卖"]
    accepted_keywords = [
        "已受理",
        "已经受理",
        "重复投诉",
        "已经处理",
        "已处理",
        "同一事项",
        "再次投诉",
        "法院已受理",
        "仲裁已受理",
        "消协已处理",
    ]
    not_consumer_keywords = [
        "劳动纠纷",
        "工资",
        "拖欠工资",
        "邻里",
        "借款",
        "贷款",
        "租房",
        "房屋租赁",
        "医疗纠纷",
        "投资",
        "加盟",
        "合伙",
        "经营纠纷",
        "工程款",
        "货款",
    ]
    expired_keywords = ["超过三年", "三年前", "多年以前", "十年前", "已过三年"]
    missing_or_false_material_keywords = [
        "商家不详",
        "地址不详",
        "无法提供",
        "没有店名",
        "主体不明",
        "不知道商家",
        "无消费凭证",
        "没有凭证",
        "无法提供凭证",
        "材料不全",
        "虚假材料",
        "虚假信息",
    ]
    identity_verify_keywords = ["冒用他人", "冒名", "身份核验", "拒绝身份核验", "不配合身份核验"]
    market_keywords = [
        "食品",
        "药",
        "价格",
        "退货",
        "退款",
        "质量",
        "虚假宣传",
        "三包",
        "餐饮",
        "超市",
        "发票",
        "计量",
        "广告",
        "消费",
        "商家",
    ]
    prepaid_service_accept_keywords = [
        "健身",
        "健身房",
        "健身俱乐部",
        "会员卡",
        "年卡",
        "预付卡",
        "储值卡",
        "充值",
        "退费",
        "退卡",
        "私教",
        "课程",
    ]
    sensitive_keywords = ["自杀", "爆炸", "投毒", "群体性", "上访", "涉密"]

    def __init__(
        self,
        model_dir: Path | None = None,
        accept_threshold: float = 0.65,
        reject_threshold: float = 0.20,
    ) -> None:
        self.model_dir = model_dir or ACCEPT_MODEL_DIR
        self.model_path = self.model_dir / "accept_model.joblib"
        self.accept_threshold = accept_threshold
        self.reject_threshold = reject_threshold
        self._model: Any | None = None

    @property
    def model(self) -> Any | None:
        if self._model is None and self.model_path.exists():
            self._model = joblib.load(self.model_path)
        return self._model

    def classify(self, request: ComplaintAnalyzeRequest) -> tuple[ClassificationResult, list[str]]:
        text = normalize_text(" ".join(filter(None, [request.problem_text, request.appeal_text])))
        review_reasons: list[str] = []
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
        if any(word in text for word in self.sensitive_keywords):
            review_reasons.append("命中敏感词，需人工复核")

        transfer_rules = [
            (self.agriculture_transfer_keywords, "农业农村局", "命中农业农村、兽药兽用、农药或农机类职责外规则，需人工确认"),
            (self.housing_transfer_keywords, "住建部门", "命中物业、公租房、供水供暖等职责外规则，需人工确认"),
            (self.public_security_transfer_keywords, "公安机关", "命中治安、人身伤害或公安已处理类职责外规则，需人工确认"),
            (self.tobacco_transfer_keywords, "烟草专卖部门", "命中烟草专卖类职责外规则，需人工确认"),
        ]
        for keywords, department, reason_text in transfer_rules:
            matched = [word for word in keywords if word in text]
            if not matched:
                continue
            review_reasons.append(reason_text)
            return (
                ClassificationResult(
                    is_market=False,
                    accept_suggestion=AcceptSuggestion.REVIEW,
                    reason_type=ReasonType.ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY,
                    confidence=0.88,
                    decision_source=DecisionSource.RULE,
                    evidence_fields=[*matched, f"suggest_department={department}"],
                ),
                review_reasons,
            )

        for keywords, reason in [
            (self.accepted_keywords, ReasonType.ARTICLE16_2_ALREADY_ACCEPTED_OR_PROCESSED),
            (self.out_scope_keywords, ReasonType.ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY),
            (self.not_consumer_keywords, ReasonType.ARTICLE16_3_NOT_CONSUMER_DISPUTE),
            (self.expired_keywords, ReasonType.ARTICLE16_4_EXPIRED),
            (self.missing_or_false_material_keywords, ReasonType.ARTICLE16_5_MISSING_OR_FALSE_MATERIALS),
            (self.identity_verify_keywords, ReasonType.ARTICLE16_6_IMPERSONATION_OR_REFUSE_VERIFY),
        ]:
            matched = [word for word in keywords if word in text]
            if matched:
                confidence = 0.86 if reason != ReasonType.ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY else 0.78
                review_reasons.append("命中第16条可能不予受理规则，需人工确认")
                return (
                    ClassificationResult(
                        is_market=True,
                        accept_suggestion=AcceptSuggestion.REVIEW,
                        reason_type=reason,
                        confidence=confidence,
                        decision_source=DecisionSource.RULE,
                        evidence_fields=matched,
                ),
                    review_reasons,
                )

        prepaid_service_matches = [word for word in self.prepaid_service_accept_keywords if word in text]
        if prepaid_service_matches and any(word in text for word in ["退费", "退款", "退卡", "消费", "会员卡", "年卡"]):
            return (
                ClassificationResult(
                    is_market=True,
                    accept_suggestion=AcceptSuggestion.ACCEPT,
                    reason_type=ReasonType.UNKNOWN,
                    confidence=min(0.92, 0.78 + len(prepaid_service_matches) * 0.02),
                    decision_source=DecisionSource.RULE,
                    evidence_fields=prepaid_service_matches[:8],
                ),
                review_reasons,
            )

        model_result = self._classify_with_model(text)
        if model_result is not None:
            if model_result.accept_suggestion == AcceptSuggestion.REVIEW:
                review_reasons.append("模型受理概率处于人工复核区间")
            return model_result, review_reasons

        matched_market = [word for word in self.market_keywords if word in text]
        if matched_market:
            review_reasons.append("模型文件不可用，使用市场监管关键词规则")
            return (
                ClassificationResult(
                    is_market=True,
                    accept_suggestion=AcceptSuggestion.ACCEPT,
                    reason_type=ReasonType.UNKNOWN,
                    confidence=min(0.94, 0.72 + len(matched_market) * 0.03),
                    decision_source=DecisionSource.RULE,
                    evidence_fields=matched_market[:8],
                ),
                review_reasons,
            )

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
        if not any(clue in text for clue in self.complaint_clues + self.out_scope_keywords + self.agriculture_transfer_keywords):
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
