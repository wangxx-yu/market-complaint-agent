"""Answer Verifier — 校验回复中的法规引用是否真实存在于 RAG 检索结果中。"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.schemas import RetrievalHit


@dataclass
class VerificationResult:
    passed: bool
    issues: list[str] = field(default_factory=list)
    requires_review: bool = False
    fallback_reply: str | None = None


class AnswerVerifier:
    """校验回复生成质量：法规引用真实性、降级检测、冲突检测、绝对化措辞。"""

    ABSOLUTE_PATTERNS: list[tuple[re.Pattern, str]] = [
        (re.compile(r"保证.{0,5}(能|可以|一定|肯定)"), "过度承诺-保证"),
        (re.compile(r"必定.{0,3}(退|赔|罚|处理)"), "过度承诺-必定"),
        (re.compile(r"绝对.{0,3}(没问题|可以|能行)"), "过度承诺-绝对"),
        (re.compile(r"包(您|你).{0,3}(满意|解决|退)"), "过度承诺-包办"),
        (re.compile(r"一定(能|会|可以)"), "绝对化措辞-一定"),
    ]

    REGULATION_REF_PATTERN: re.Pattern = re.compile(
        r"(第[一二三四五六七八九十百零〇两]+条|《[^》]+》)"
    )

    def verify(self, reply_text: str, retrieval_hits: list[RetrievalHit]) -> VerificationResult:
        issues: list[str] = []

        issues.extend(self._check_absolute_wording(reply_text))
        issues.extend(self._check_regulation_authenticity(reply_text, retrieval_hits))
        issues.extend(self._check_degraded_retrieval(retrieval_hits))
        issues.extend(self._check_conflicting_hits(retrieval_hits))

        requires_review = (
            len(issues) > 0
            or (not retrieval_hits and self.REGULATION_REF_PATTERN.search(reply_text))
        )

        fallback = None
        if not retrieval_hits and self.REGULATION_REF_PATTERN.search(reply_text):
            fallback = "系统无法检索到相关法规依据，该投诉建议人工处理。"

        return VerificationResult(
            passed=len(issues) == 0 and fallback is None,
            issues=issues,
            requires_review=requires_review,
            fallback_reply=fallback,
        )

    def _check_absolute_wording(self, text: str) -> list[str]:
        issues: list[str] = []
        for pattern, label in self.ABSOLUTE_PATTERNS:
            if pattern.search(text):
                issues.append(f"回复含{label}")
        return issues

    def _check_regulation_authenticity(
        self, reply_text: str, hits: list[RetrievalHit]
    ) -> list[str]:
        issues: list[str] = []
        if not self.REGULATION_REF_PATTERN.search(reply_text):
            return issues
        if not hits:
            issues.append("回复引用了法规但RAG未检索到结果，可能产生幻觉")
            return issues
        for hit in hits:
            if hit.title and self._title_fuzzy_match(hit.title, reply_text):
                return []
            if hit.content and any(
                snippet.strip() in reply_text
                for snippet in hit.content[:60].split("。")
                if len(snippet.strip()) > 6
            ):
                return []
        issues.append("回复中的法规引用未在检索结果中找到对应条文，需人工复核")
        return issues

    def _check_degraded_retrieval(self, hits: list[RetrievalHit]) -> list[str]:
        issues: list[str] = []
        for hit in hits:
            if hit.source and "FALLBACK" in hit.source.upper():
                issues.append("检索源降级(FALLBACK)，建议人工复核")
                break
        return issues

    def _check_conflicting_hits(self, hits: list[RetrievalHit]) -> list[str]:
        if len(hits) < 2:
            return []
        high_score = [h for h in hits if h.score >= 0.7]
        if len(high_score) >= 2 and len({h.title for h in high_score}) >= 2:
            titles = ", ".join(h.title for h in high_score[:3])
            return [f"多条高分法规建议可能冲突: {titles}，建议人工复核"]
        return []

    @staticmethod
    def _title_fuzzy_match(title: str, reply_text: str) -> bool:
        """模糊匹配：标题或回复中是否有核心法规名重合。"""
        if title in reply_text:
            return True
        # 提取书名号内容做匹配
        title_refs = re.findall(r"《([^》]+)》", reply_text)
        for ref in title_refs:
            if ref in title or title in ref:
                return True
        # 双向子串：核心词（>=5字）命中
        for i in range(len(title) - 4):
            chunk = title[i:i + 5]
            if chunk in reply_text:
                return True
        return False
