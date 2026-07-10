"""节点级 Trace 结构测试（Wave 5.1）。"""
from __future__ import annotations

from app.core.schemas import AgentStep


class TestAgentStepDecisionSource:
    def test_decision_source_field(self):
        step = AgentStep(
            name="classify",
            input_summary={"text": "test"},
            output_summary={"accept_suggestion": "ACCEPT"},
            confidence=0.9,
            duration_ms=10.5,
            decision_source="RULE",
        )
        assert step.decision_source == "RULE"
        assert step.name == "classify"

    def test_decision_source_none_by_default(self):
        step = AgentStep(
            name="validate",
            input_summary={},
            output_summary={},
            duration_ms=1.0,
        )
        assert step.decision_source is None

    def test_serialization(self):
        step = AgentStep(
            name="classify",
            input_summary={},
            output_summary={},
            duration_ms=5.0,
            decision_source="MODEL",
        )
        d = step.model_dump(mode="json")
        assert d["decision_source"] == "MODEL"
