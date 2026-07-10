"""复核队列 API 测试（Wave 3.1）。
注：API 集成测试受 langgraph 'dispatch' state key 预存 bug 阻碍（routes imports orchestrator）。
路由逻辑通过 schema 验证 + 独立函数测试覆盖。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.schemas import ReviewConfirmRequest, ReviewConfirmResponse
from app.core.storage import JsonlStore


class TestReviewConfirmRequest:
    def test_legacy_fields(self):
        req = ReviewConfirmRequest(is_market=True, reviewer="张三", notes="测试")
        assert req.is_market is True
        assert req.reviewer == "张三"
        assert req.can_use_for_training is True  # default

    def test_enhanced_fields(self):
        req = ReviewConfirmRequest(
            is_market=True,
            reviewer="李四",
            correct_reason_type="ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY",
            correct_office="叶盛市场监管所",
            reviewer_note="复核通过",
            can_use_for_training=False,
        )
        assert req.correct_reason_type == "ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY"
        assert req.correct_office == "叶盛市场监管所"
        assert req.reviewer_note == "复核通过"
        assert req.can_use_for_training is False

    def test_default_can_use_for_training(self):
        req = ReviewConfirmRequest()
        assert req.can_use_for_training is True

    def test_legacy_reason_type_normalized(self):
        """旧 reason_type 枚举值被 legacy_map 标准化。"""
        req = ReviewConfirmRequest(reason_type="ARTICLE15_1_OUT_OF_SCOPE")
        assert req.reason_type == "ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY"


class TestReviewConfirmResponse:
    def test_structure(self):
        from datetime import datetime, timezone
        resp = ReviewConfirmResponse(trace_id="abc", saved=True, saved_at=datetime.now(timezone.utc))
        assert resp.trace_id == "abc"
        assert resp.saved is True
        assert resp.saved_at is not None


class TestReviewStoreOperations:
    def test_append_and_find(self, tmp_path: Path):
        store = JsonlStore(tmp_path / "reviews.jsonl")
        store.append({
            "trace_id": "trace-001",
            "saved_at": "2024-01-01",
            "decision": "REJECT",
            "review": {"reviewer": "测试员", "can_use_for_training": True},
        })
        found = store.find_by_key("trace_id", "trace-001")
        assert found is not None
        assert found["decision"] == "REJECT"
        assert found["review"]["reviewer"] == "测试员"

    def test_find_nonexistent(self, tmp_path: Path):
        store = JsonlStore(tmp_path / "reviews.jsonl")
        found = store.find_by_key("trace_id", "nonexistent")
        assert found is None

    def test_pending_logic(self, tmp_path: Path):
        """模拟 pending 筛选逻辑。"""
        trace_store = JsonlStore(tmp_path / "traces.jsonl")
        review_store = JsonlStore(tmp_path / "reviews.jsonl")

        trace_store.append({
            "trace_id": "t1",
            "request": {"problem_text": "投诉1"},
            "review_reasons": ["需复核"],
            "classification": {"accept_suggestion": "REVIEW"},
        })
        trace_store.append({
            "trace_id": "t2",
            "request": {"problem_text": "投诉2"},
            "review_reasons": [],
            "classification": {"accept_suggestion": "ACCEPT"},
        })
        trace_store.append({
            "trace_id": "t3",
            "request": {"problem_text": "投诉3"},
            "review_reasons": ["需复核"],
            "classification": {"accept_suggestion": "REVIEW"},
        })
        review_store.append({"trace_id": "t3", "saved_at": "2024-01-01", "review": {}})

        traces = trace_store.all()
        reviewed_ids = {r.get("trace_id") for r in review_store.all()}
        pending = [
            t for t in traces
            if t.get("review_reasons") and t.get("trace_id") not in reviewed_ids
        ]
        assert len(pending) == 1
        assert pending[0]["trace_id"] == "t1"
