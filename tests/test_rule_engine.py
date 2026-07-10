"""RuleLoader + RuleEngine + Classifier 集成测试（Wave 1.2）。"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.agents.classifier import ClassifierAgent
from app.agents.rule_engine import RuleEngine, RuleEngineResult, RuleMatch
from app.core.enums import AcceptSuggestion, DecisionSource, ReasonType
from app.core.rule_loader import RuleLoader
from app.core.schemas import ComplaintAnalyzeRequest


# ─── RuleLoader ───────────────────────────────────────────────

class TestRuleLoader:
    def test_load_accept(self):
        loader = RuleLoader()
        rules = loader.load("accept")
        assert len(rules) >= 2
        ids = {r["id"] for r in rules}
        assert "accept_prepaid_service" in ids
        assert "accept_market_general" in ids
        # 按 priority 降序
        priorities = [r["priority"] for r in rules]
        assert priorities == sorted(priorities, reverse=True)

    def test_load_reject(self):
        loader = RuleLoader()
        rules = loader.load("reject")
        assert len(rules) >= 8
        ids = {r["id"] for r in rules}
        assert "out_of_scope_agriculture" in ids
        assert "already_accepted_or_processed" in ids
        assert "missing_or_false_material" in ids

    def test_load_dispatch(self):
        loader = RuleLoader()
        rules = loader.load("dispatch")
        assert len(rules) >= 6
        ids = {r["id"] for r in rules}
        assert "dispatch_xiaoba" in ids
        assert "dispatch_drug_bureau" in ids

    def test_load_sensitive(self):
        loader = RuleLoader()
        rules = loader.load("sensitive")
        assert len(rules) >= 4
        ids = {r["id"] for r in rules}
        assert "sensitive_suicide" in ids

    def test_load_missing_file_returns_empty(self):
        loader = RuleLoader()
        rules = loader.load("nonexistent")
        assert rules == []

    def test_load_all(self):
        loader = RuleLoader()
        all_rules = loader.load_all()
        assert set(all_rules.keys()) == {"accept", "reject", "dispatch", "sensitive"}
        assert len(all_rules["accept"]) >= 2
        assert len(all_rules["reject"]) >= 8

    def test_load_from_custom_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            rules_dir = Path(tmp)
            (rules_dir / "accept_rules.yaml").write_text(
                "rules:\n  - id: test_rule\n    keywords: [测试]\n    priority: 50\n",
                encoding="utf-8",
            )
            loader = RuleLoader(rules_dir=rules_dir)
            rules = loader.load("accept")
            assert len(rules) == 1
            assert rules[0]["id"] == "test_rule"


# ─── RuleEngine ────────────────────────────────────────────────

class TestRuleEngine:
    @pytest.fixture
    def engine(self):
        return RuleEngine()

    def test_match_accept_prepaid(self, engine):
        result = engine.match_accept("我在健身房办了会员卡要求退费")
        assert result.has_match
        assert result.highest_priority_match is not None
        top = result.highest_priority_match
        assert top.rule_id == "accept_prepaid_service"
        assert "健身" in top.matched_keywords
        assert "会员卡" in top.matched_keywords

    def test_match_accept_market(self, engine):
        result = engine.match_accept("超市买到了过期食品要求退款")
        assert result.has_match
        ids = {m.rule_id for m in result.matches}
        # 两规则都会命中
        assert "accept_market_general" in ids

    def test_match_reject_agriculture(self, engine):
        result = engine.match_reject("我买的农药有问题，牛打架用了兽药")
        assert result.has_match
        top = result.highest_priority_match
        assert top is not None
        assert top.rule_id == "out_of_scope_agriculture"
        assert top.suggest_department == "农业农村局"
        assert "农药" in top.matched_keywords

    def test_match_reject_article16(self, engine):
        result = engine.match_reject("这个事情已经处理过了，是重复投诉")
        assert result.has_match
        top = result.highest_priority_match
        assert top is not None
        assert top.rule_id == "already_accepted_or_processed"
        assert top.reason_type == "ARTICLE16_2_ALREADY_ACCEPTED_OR_PROCESSED"

    def test_match_sensitive(self, engine):
        result = engine.match_sensitive("我要自杀，你们必须处理")
        assert result.has_match
        top = result.highest_priority_match
        assert top is not None
        assert top.rule_id == "sensitive_suicide"

    def test_match_sensitive_none(self, engine):
        result = engine.match_sensitive("超市买到过期食品")
        assert not result.has_match

    def test_match_dispatch_xiaoba(self, engine):
        result = engine.match_dispatch("小坝镇有商家欺诈")
        assert result.has_match
        top = result.highest_priority_match
        assert top is not None
        assert top.rule_id == "dispatch_xiaoba"

    def test_dispatch_drug_excludes_veterinary(self, engine):
        """药品规则排除兽药关键词。"""
        result = engine.match_dispatch("兽药和饲料有问题")
        drug_matches = [m for m in result.matches if m.rule_id == "dispatch_drug_bureau"]
        assert len(drug_matches) == 0

    def test_no_match(self, engine):
        result = engine.match_accept("今天天气真好")
        assert not result.has_match

    def test_highest_priority_first(self, engine):
        """高 priority 规则排在前面。"""
        result = engine.match_accept("药店卖假药要求退款")
        if result.has_match and len(result.matches) >= 2:
            assert result.matches[0].priority >= result.matches[1].priority

    def test_reload(self, engine):
        engine.reload()
        assert "accept" in engine._rules
        assert "reject" in engine._rules


# ─── Classifier 集成测试 ───────────────────────────────────────

def _req(text: str) -> ComplaintAnalyzeRequest:
    return ComplaintAnalyzeRequest(problem_text=text)


class TestClassifierWithRuleEngine:
    @pytest.fixture
    def agent(self):
        return ClassifierAgent()

    def test_invalid_input_too_short(self, agent):
        result, reasons = agent.classify(_req("投诉"))
        assert result.accept_suggestion == AcceptSuggestion.REVIEW
        assert result.decision_source == DecisionSource.RULE
        assert "invalid_input" in result.evidence_fields

    def test_sensitive_keyword(self, agent):
        result, reasons = agent.classify(_req("我要自杀也要投诉这个商家"))
        assert any("敏感词" in r for r in reasons)

    def test_agriculture_transfer(self, agent):
        """农业农村职责外 → is_market=False + REVIEW。"""
        result, reasons = agent.classify(_req("买的农药无效，牛用的饲料也有问题"))
        assert result.is_market is False
        assert result.accept_suggestion == AcceptSuggestion.REVIEW
        assert result.reason_type == ReasonType.ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY
        assert "suggest_department=农业农村局" in result.evidence_fields

    def test_housing_transfer(self, agent):
        result, reasons = agent.classify(_req("物业费太贵了，物监办不管"))
        assert result.is_market is False
        assert "suggest_department=住建部门" in result.evidence_fields

    def test_article16_already_accepted(self, agent):
        result, reasons = agent.classify(_req("这个投诉已经受理重复投诉了"))
        assert result.is_market is True
        assert result.accept_suggestion == AcceptSuggestion.REVIEW
        assert result.reason_type == ReasonType.ARTICLE16_2_ALREADY_ACCEPTED_OR_PROCESSED

    def test_article16_missing_material(self, agent):
        result, reasons = agent.classify(_req("不知道商家是谁商家不详无法提供凭证"))
        assert result.reason_type == ReasonType.ARTICLE16_5_MISSING_OR_FALSE_MATERIALS

    def test_prepaid_accept(self, agent):
        result, reasons = agent.classify(_req("健身房办了年卡，现在要退费"))
        assert result.accept_suggestion == AcceptSuggestion.ACCEPT
        assert result.decision_source == DecisionSource.RULE
        assert any("健身" in e for e in result.evidence_fields)

    def test_market_fallback(self, agent):
        """模型不可用时，市场监管关键词兜底。"""
        agent._model = None  # 阻止 model 属性重新加载
        agent.model_path = Path("/nonexistent/model.joblib")  # 确保不会从磁盘加载
        result, reasons = agent.classify(_req("超市买到了过期食品要求退款"))
        assert result.accept_suggestion == AcceptSuggestion.ACCEPT
        assert result.decision_source == DecisionSource.RULE
        assert any("食品" in e for e in result.evidence_fields)

    def test_default_review(self, agent):
        """无任何规则命中 → REVIEW。"""
        agent._model = None
        agent.model_path = Path("/nonexistent/model.joblib")
        result, reasons = agent.classify(_req("这是一个无法识别的投诉内容XYZ"))
        assert result.accept_suggestion == AcceptSuggestion.REVIEW
        assert any("未命中" in r for r in reasons)

    def test_evidence_fields_exist_on_rule_match(self, agent):
        """规则命中时 evidence_fields 包含匹配关键词。"""
        result, _ = agent.classify(_req("小区物业供暖有问题需要投诉"))
        assert len(result.evidence_fields) > 0
        # "供暖" 命中 out_of_scope_housing（transfer 分支，有 suggest_department）
        assert any("供暖" in e for e in result.evidence_fields)

    def test_transfer_before_article16(self, agent):
        """
        转办规则（带 suggest_department）优先于 Article16 规则。
        同时命中 '物业'（out_of_scope_housing → transfer）和 '物业'（out_of_scope_general → article16）。
        但 housing 规则有 suggest_department，应走 transfer 分支。
        """
        result, _ = agent.classify(_req("物业费太贵物业供暖也有问题"))
        assert result.is_market is False
        assert "suggest_department" in " ".join(result.evidence_fields)


# ─── RuleMatch dataclass ───────────────────────────────────────

class TestRuleMatch:
    def test_dataclass_defaults(self):
        match = RuleMatch(
            rule_id="test",
            rule_name="测试",
            decision="REVIEW",
            reason_type="UNKNOWN",
            priority=50,
            matched_keywords=["测试"],
            review_required=True,
            source="test",
        )
        assert match.suggest_department is None
        assert match.note is None


# ─── RuleEngineResult dataclass ─────────────────────────────────

class TestRuleEngineResult:
    def test_empty(self):
        result = RuleEngineResult()
        assert not result.has_match
        assert result.highest_priority_match is None

    def test_single_match(self):
        match = RuleMatch(
            rule_id="t", rule_name="T", decision="ACCEPT",
            reason_type="UNKNOWN", priority=50, matched_keywords=["x"],
            review_required=False, source="test",
        )
        result = RuleEngineResult(matches=[match], highest_priority_match=match)
        assert result.has_match
        assert result.highest_priority_match is match
