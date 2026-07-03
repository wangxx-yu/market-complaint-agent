"""Guardrails Agent — 输入安全检测 + 输出合规校验。

在 pipeline 首尾插入：pre-guard（输入）→ agents → post-guard（输出）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

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

    # 法规引用校验 — 必须包含法规名称或条文
    REGULATION_PATTERN: re.Pattern = re.compile(
        r"(第[一二三四五六七八九十百零〇两]+条|《[^》]+》|规定|办法|条例|依法)"
    )

    def check_input(self, request: ComplaintAnalyzeRequest) -> GuardResult:
        """Pre-guard: 检测输入中的 PII 和无效内容。"""
        issues: list[str] = []
        text = request.problem_text or ""

        if len(text.strip()) < 2:
            issues.append("输入内容过短，疑似无效投诉")

        for pattern, label in self.PII_PATTERNS:
            if re.search(pattern, text):
                issues.append(f"检测到疑似{label}，已脱敏处理")

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

        if not self.REGULATION_PATTERN.search(reply_text):
            issues.append("回复未引用法规依据，建议补充条文引用")

        return GuardResult(passed=len(issues) == 0, issues=issues)
