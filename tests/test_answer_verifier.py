"""AnswerVerifier 测试（Wave 2.3）。"""
from __future__ import annotations

import pytest

from app.agents.answer_verifier import AnswerVerifier, VerificationResult
from app.core.schemas import RetrievalHit


def _hit(title: str = "", content: str = "", score: float = 0.8, source: str = "KB") -> RetrievalHit:
    return RetrievalHit(title=title, content=content, score=score, source=source)


class TestVerificationResult:
    def test_defaults(self):
        r = VerificationResult(passed=True)
        assert r.passed
        assert r.issues == []
        assert not r.requires_review
        assert r.fallback_reply is None


class TestAbsoluteWording:
    @pytest.fixture
    def verifier(self):
        return AnswerVerifier()

    def test_guarantee_detected(self, verifier):
        result = verifier.verify("我们保证能退款", [])
        assert not result.passed
        assert any("过度承诺" in i for i in result.issues)

    def test_biding_detected(self, verifier):
        result = verifier.verify("必定会处罚商家", [])
        assert not result.passed

    def test_certain_detected(self, verifier):
        result = verifier.verify("一定可以退款", [])
        assert not result.passed
        assert any("绝对化" in i for i in result.issues)

    def test_clean_text_passes(self, verifier):
        result = verifier.verify("您的投诉已登记，工作人员将依法处理。", [])
        assert result.passed


class TestRegulationAuthenticity:
    @pytest.fixture
    def verifier(self):
        return AnswerVerifier()

    def test_no_regulation_ref_ok(self, verifier):
        """无法规引用时不报错。"""
        result = verifier.verify("请补充材料后提交。", [])
        assert result.passed

    def test_regulation_without_hits(self, verifier):
        """引用法规但无检索结果 → 幻觉风险。"""
        result = verifier.verify("根据《消费者权益保护法》第五十五条", [])
        assert not result.passed
        assert result.fallback_reply is not None

    def test_regulation_with_matching_hit(self, verifier):
        """法规引用在检索结果中 → 通过。"""
        hits = [_hit(title="中华人民共和国消费者权益保护法", content="第五十五条规定...")]
        result = verifier.verify("根据《消费者权益保护法》第五十五条", hits)
        assert result.passed

    def test_regulation_without_matching_hit(self, verifier):
        """法规引用不在检索结果中 → 标记。"""
        hits = [_hit(title="市场监督管理投诉举报处理办法", content="第十六条规定...")]
        result = verifier.verify("根据《消费者权益保护法》第五十五条", hits)
        assert any("未在检索结果" in i for i in result.issues)


class TestDegradedRetrieval:
    @pytest.fixture
    def verifier(self):
        return AnswerVerifier()

    def test_fallback_source(self, verifier):
        hits = [_hit(title="T", source="FALLBACK")]
        result = verifier.verify("根据有关规定处理", hits)
        assert result.requires_review
        assert any("降级" in i for i in result.issues)


class TestConflictingHits:
    @pytest.fixture
    def verifier(self):
        return AnswerVerifier()

    def test_no_conflict_single_hit(self, verifier):
        result = verifier.verify("文本", [_hit(title="法规A")])
        assert result.passed

    def test_conflict_multiple_high_score(self, verifier):
        hits = [
            _hit(title="法规A", score=0.9),
            _hit(title="法规B", score=0.85),
        ]
        result = verifier.verify("文本", hits)
        assert any("冲突" in i for i in result.issues)

    def test_no_conflict_same_title(self, verifier):
        hits = [
            _hit(title="法规A", score=0.9),
            _hit(title="法规A", score=0.85),
        ]
        result = verifier.verify("文本", hits)
        assert result.passed


class TestCombinedChecks:
    @pytest.fixture
    def verifier(self):
        return AnswerVerifier()

    def test_multiple_issues(self, verifier):
        """同时检测绝对化措辞和法规幻觉。"""
        result = verifier.verify("保证能退款，根据《消费者权益保护法》", [])
        assert len(result.issues) >= 2
        assert result.requires_review

    def test_empty_reply(self, verifier):
        result = verifier.verify("", [])
        assert result.passed
