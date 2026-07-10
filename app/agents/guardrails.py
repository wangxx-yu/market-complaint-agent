"""Guardrails Agent — 输入安全检测 + 输出合规校验。

在 pipeline 首尾插入：pre-guard（输入）→ agents → post-guard（输出）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.agents.prompt_injection import PromptInjectionDetector
from app.core.schemas import ComplaintAnalyzeRequest


@dataclass
class GuardResult:
    passed: bool
    issues: list[str] = field(default_factory=list)
    sanitized_text: str | None = None


class GuardrailsAgent:
    """Input/output safety checks for complaint handling pipeline."""

    # 敏感信息模式（PII）
    PII_PATTERNS: list[tuple[str, str]] = [
        (r"\b1[3-9]\d{9}\b", "手机号"),
        (r"\b\d{6}(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b", "身份证号"),
        (r"\b\d{16,19}\b", "银行卡号"),
    ]

    # 输出敏感词（不应出现在回复中）
    OUTPUT_SENSITIVE_WORDS: list[str] = [
        "肯定可以赔", "保证能退", "包您满意", "绝对",
    ]

    # 过度承诺模式
    OVER_PROMISE_PATTERNS: list[tuple[re.Pattern, str]] = [
        (re.compile(r"保证.{0,5}(能|可以|一定|肯定)"), "过度承诺-保证"),
        (re.compile(r"必定.{0,3}(退|赔|罚|处理)"), "过度承诺-必定"),
        (re.compile(r"绝对.{0,3}(没问题|可以|能行)"), "过度承诺-绝对"),
        (re.compile(r"包(您|你).{0,3}(满意|解决|退)"), "过度承诺-包办"),
    ]

    # 法规引用校验 — 必须包含法规名称或条文
    REGULATION_PATTERN: re.Pattern = re.compile(
        r"(第[一二三四五六七八九十百零〇两]+条|《[^》]+》|规定|办法|条例|依法)"
    )

    def __init__(self) -> None:
        self.injection_detector = PromptInjectionDetector()

    def check_input(self, request: ComplaintAnalyzeRequest) -> GuardResult:
        """Pre-guard: 检测输入中的 PII、注入攻击和无效内容。"""
        issues: list[str] = []
        text = request.problem_text or ""

        if len(text.strip()) < 2:
            issues.append("输入内容过短，疑似无效投诉")

        for pattern, label in self.PII_PATTERNS:
            if re.search(pattern, text):
                issues.append(f"检测到疑似{label}，已脱敏处理")

        # 提示词注入检测
        injection = self.injection_detector.detect(text)
        if not injection.safe:
            issues.append(f"检测到提示词注入风险(risk={injection.risk_level}): {', '.join(injection.detected_patterns)}")

        return GuardResult(passed=len(issues) == 0, issues=issues)

    def check_output(self, reply_text: str) -> GuardResult:
        """Post-guard: 校验回复内容合规。"""
        issues: list[str] = []

        if not reply_text.strip():
            issues.append("回复为空")
            return GuardResult(passed=False, issues=issues)

        for word in self.OUTPUT_SENSITIVE_WORDS:
            if word in reply_text:
                issues.append(f"回复含敏感措辞: {word}")

        # 过度承诺检测
        for pattern, label in self.OVER_PROMISE_PATTERNS:
            if pattern.search(reply_text):
                issues.append(f"回复含{label}")

        if not self.REGULATION_PATTERN.search(reply_text):
            issues.append("回复未引用法规依据，建议补充条文引用")

        return GuardResult(passed=len(issues) == 0, issues=issues)
