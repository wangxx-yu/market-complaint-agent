"""Prompt Injection Detector — 检测用户输入中的注入攻击。"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class InjectionResult:
    safe: bool
    risk_level: str  # LOW / MEDIUM / HIGH
    detected_patterns: list[str] = field(default_factory=list)
    sanitized_text: str | None = None


class PromptInjectionDetector:
    INJECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
        (re.compile(r"(忽略|忘记|无视|跳过).{0,10}(以上|之前|前面|所有).{0,10}(规则|指令|要求|限制)"), "指令覆盖"),
        (re.compile(r"(你是|你现在是|从现在开始你是).{0,20}(必须|一定要|无论如何)"), "角色劫持"),
        (re.compile(r"(泄露|显示|输出|告诉我).{0,10}(系统|提示词|prompt|指令)"), "提示词窃取"),
        (re.compile(r"(必须受理|一定要受理|必须立案|必须处理|不准驳回)"), "强制受理"),
        (re.compile(r"(不需要|不用|跳过).{0,10}(法规|法律|依据|条文)"), "跳过法规"),
        (re.compile(r"(base64|eval\(|exec\(|__import__|os\.system)"), "代码注入"),
    ]

    def detect(self, text: str) -> InjectionResult:
        patterns: list[str] = []
        for pattern, label in self.INJECTION_PATTERNS:
            if pattern.search(text):
                patterns.append(label)
        risk = "HIGH" if len(patterns) >= 2 else ("MEDIUM" if patterns else "LOW")
        return InjectionResult(
            safe=len(patterns) == 0,
            risk_level=risk,
            detected_patterns=patterns,
        )
