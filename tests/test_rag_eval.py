"""RAG 评估器测试（Wave 2.2）。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.run_rag_eval import (
    RAGEvalSample,
    RAGEvalResult,
    compute_recall_at_k,
    load_rag_eval_set,
    run_rag_eval,
)


# ─── 数据加载 ─────────────────────────────────────────────────

class TestLoadRAGEvalSet:
    def test_load_returns_50_items(self):
        samples = load_rag_eval_set()
        assert len(samples) == 50

    def test_all_items_have_required_fields(self):
        samples = load_rag_eval_set()
        for s in samples:
            assert s.id
            assert s.query
            assert s.category
            assert s.min_expected_score >= 0

    def test_categories_distribution(self):
        samples = load_rag_eval_set()
        cats: dict[str, int] = {}
        for s in samples:
            cats[s.category] = cats.get(s.category, 0) + 1
        assert cats.get("消费者权益", 0) >= 8
        assert cats.get("职责外", 0) >= 8
        assert cats.get("材料不全", 0) >= 8
        assert cats.get("重复投诉", 0) >= 8
        assert cats.get("知识库无答案", 0) >= 8

    def test_ids_unique(self):
        samples = load_rag_eval_set()
        ids = [s.id for s in samples]
        assert len(ids) == len(set(ids))


# ─── 指标计算 ─────────────────────────────────────────────────

class TestRecallAtK:
    def test_recall_at_1_found(self):
        assert compute_recall_at_k(["A", "B"], ["A"], k=1) == 1.0

    def test_recall_at_1_not_found(self):
        assert compute_recall_at_k(["A", "B"], ["C"], k=1) == 0.0

    def test_recall_at_3_partial(self):
        assert compute_recall_at_k(["A", "B"], ["A", "C", "D"], k=3) == 1.0

    def test_recall_at_3_none(self):
        assert compute_recall_at_k(["A", "B"], [], k=3) == 0.0

    def test_recall_empty_expected(self):
        assert compute_recall_at_k([], ["A"], k=3) == 1.0


# ─── 评估运行 ─────────────────────────────────────────────────

class TestRAGEvalRun:
    def test_run_returns_report(self, tmp_path: Path):
        """评估可运行并返回结构化的报告。"""
        report = run_rag_eval(output_dir=tmp_path, top_k=3)
        assert report.total > 0
        assert report.recall_at_1 >= 0
        assert report.recall_at_3 >= 0
        assert report.mrr >= 0
        assert report.avg_score >= 0
        # 总数一致性
        assert report.total == report.found_results + report.no_results

    def test_output_file_written(self, tmp_path: Path):
        run_rag_eval(output_dir=tmp_path, top_k=3)
        result_file = tmp_path / "rag_eval_result.json"
        assert result_file.exists()
        data = json.loads(result_file.read_text(encoding="utf-8"))
        assert "total" in data
        assert "recall_at_1" in data


# ─── 数据结构 ─────────────────────────────────────────────────

class TestRAGEvalSample:
    def test_from_dict(self):
        d = {
            "id": "rag_001",
            "query": "测试查询",
            "reason_type": "UNKNOWN",
            "expected_title": "测试法规",
            "expected_article": "第一条",
            "min_expected_score": 0.5,
            "category": "测试类",
        }
        s = RAGEvalSample(**d)
        assert s.id == "rag_001"
        assert s.query == "测试查询"


class TestRAGEvalResult:
    def test_defaults(self):
        r = RAGEvalResult(sample_id="test")
        assert r.sample_id == "test"
        assert r.hits == []
        assert r.recall_hit is False
