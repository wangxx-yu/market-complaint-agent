from __future__ import annotations

import json
import re
from pathlib import Path

from app.core.config import settings
from app.core.enums import DecisionSource
from app.core.schemas import ComplaintAnalyzeRequest, DispatchResult
from app.core.text import normalize_text


class DispatchAgent:
    drug_keywords = [
        "药品",
        "药店",
        "药房",
        "药企",
        "药师",
        "处方药",
        "非处方药",
        "中成药",
        "中药饮片",
        "保健药",
        "感冒药",
        "降压药",
        "胰岛素",
        "疫苗",
        "医疗器械",
    ]
    veterinary_drug_keywords = ["兽药", "兽用", "牛用", "饲料", "饲料添加剂", "养殖", "畜牧"]

    def __init__(self, aliases_path: Path | None = None) -> None:
        self.aliases_path = aliases_path or settings.address_aliases_path
        self.aliases = self._load_aliases()

    def _load_aliases(self) -> dict[str, dict[str, str | float]]:
        built_in = {
            "铝厂": {"office_code": "QTX_HEXI", "office_name": "河西市场监管所", "confidence": 0.95},
            "小坝镇": {"office_code": "QTX_XIAOBA", "office_name": "小坝市场监管所", "confidence": 0.88},
            "小坝": {"office_code": "QTX_XIAOBA", "office_name": "小坝市场监管所", "confidence": 0.82},
            "裕民": {"office_code": "QTX_YUMIN", "office_name": "裕民市场监管所", "confidence": 0.88},
            "河西": {"office_code": "QTX_HEXI", "office_name": "河西市场监管所", "confidence": 0.88},
            "河东": {"office_code": "QTX_HEDONG", "office_name": "河东市场监管所", "confidence": 0.88},
            "瞿靖": {"office_code": "QTX_QUJING", "office_name": "瞿靖市场监管所", "confidence": 0.88},
            "叶盛": {"office_code": "QTX_YESHENG", "office_name": "叶盛市场监管所", "confidence": 0.88},
            "大坝": {"office_code": "QTX_DABA", "office_name": "大坝市场监管所", "confidence": 0.88},
        }
        loaded: dict[str, dict[str, str | float]] = {}
        for path in [self.aliases_path, settings.dispatch_mapping_path, settings.manual_dispatch_rules_path]:
            if path.exists():
                loaded.update(self._filter_loaded_aliases(json.loads(path.read_text(encoding="utf-8"))))
        # Built-in jurisdiction rules are curated and must win over noisy historical
        # aliases where bureau-level handling can mask the actual local office.
        merged = loaded | built_in
        return dict(sorted(merged.items(), key=lambda item: (float(item[1]["confidence"]), len(item[0])), reverse=True))

    @staticmethod
    def _filter_loaded_aliases(raw: dict[str, dict[str, str | float]]) -> dict[str, dict[str, str | float]]:
        filtered: dict[str, dict[str, str | float]] = {}
        noise = {"", "青铜峡市", "吴忠市", "宁夏", "宁夏青铜峡市"}
        field_markers = {
            "发生时间",
            "消费金额",
            "问题描述",
            "主要诉求",
            "诉求内容",
            "联系人",
            "联系电话",
        }
        for alias, target in raw.items():
            clean_alias = re.sub(r"\s+", "", str(alias)).strip("，。,.;；:：、 ")
            if clean_alias in noise:
                continue
            if any(marker in clean_alias for marker in field_markers):
                continue
            if "*****" in clean_alias:
                continue
            if len(clean_alias) < 2 or len(clean_alias) > 30:
                continue
            if clean_alias.count(",") + clean_alias.count("，") >= 2:
                continue
            confidence = float(target.get("confidence", 0))
            if confidence < 0.8:
                continue
            if str(target.get("office_name", "")) == "青铜峡市市场监督管理局":
                continue
            filtered[clean_alias] = target
        return filtered

    def dispatch(self, request: ComplaintAnalyzeRequest) -> DispatchResult:
        full_text = normalize_text(
            " ".join(
                filter(
                    None,
                    [
                        request.problem_text,
                        request.appeal_text,
                        request.incident_location,
                        request.enterprise_address,
                        request.enterprise_name,
                    ],
                )
            )
        )
        drug_match = self._match_drug_keyword(full_text)
        if drug_match:
            return DispatchResult(
                office_code="QTX_BUREAU",
                office_name="青铜峡市市场监督管理局",
                confidence=0.98,
                decision_source=DecisionSource.RULE,
                matched_rule=f"药品事项:{drug_match}",
                needs_review=False,
            )

        address = normalize_text(
            " ".join(
                filter(
                    None,
                    [
                        request.incident_location,
                        request.enterprise_address,
                        self._extract_address_from_problem(request.problem_text),
                    ],
                )
            )
        )
        for alias, target in self.aliases.items():
            if alias and alias in address:
                return DispatchResult(
                    office_code=str(target["office_code"]),
                    office_name=str(target["office_name"]),
                    confidence=float(target["confidence"]),
                    decision_source=DecisionSource.RULE,
                    matched_rule=alias,
                    needs_review=float(target["confidence"]) < 0.8,
                )

        return DispatchResult(
            office_code=settings.default_office_code,
            office_name=settings.default_office_name,
            confidence=0.35,
            decision_source=DecisionSource.FALLBACK,
            matched_rule="default_office",
            needs_review=True,
        )

    @staticmethod
    def _match_drug_keyword(text: str) -> str | None:
        if any(keyword in text for keyword in DispatchAgent.veterinary_drug_keywords):
            return None
        for keyword in DispatchAgent.drug_keywords:
            if keyword in text:
                return keyword
        return None

    @staticmethod
    def _extract_address_from_problem(problem_text: str | None) -> str:
        text = normalize_text(problem_text)
        if not text:
            return ""
        patterns = [
            r"(?:商户名称|企业名称|店名)[:：]\s*([^，。,；;。]{2,30})",
            r"(?:商户详细位置|商户地址|详细位置|地址|地点)[:：]\s*([^，。,；;。]{2,40})",
            r"(?:位于|位于：|在)\s*([^，。,；;。]{2,40}?)(?:，|。|消费|购买|花费|发生|店|商户|超市|药店|餐馆|酒店)",
            r"(青铜峡市[^，。,；;。]{2,40})",
        ]
        parts: list[str] = []
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                value = DispatchAgent._clean_extracted_location(match.group(1))
                if value and value not in parts:
                    parts.append(value)
        return " ".join(parts)

    @staticmethod
    def _clean_extracted_location(value: str) -> str:
        value = re.sub(r"\s+", "", value).strip("，。,.;；:：、 ")
        stop_markers = [
            "商户地址",
            "详细位置",
            "发生时间",
            "消费金额",
            "问题描述",
            "主要诉求",
            "诉求内容",
            "联系人",
            "联系电话",
        ]
        for marker in stop_markers:
            index = value.find(marker)
            if index > 0:
                value = value[:index]
        return value.strip("，。,.;；:：、 ")
