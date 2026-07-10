"""训练数据冲突检测测试（Wave 3.2）。"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from app.tools.detect_training_conflicts import (
    ConflictReport,
    detect_label_conflicts,
    detect_decision_overrides,
    detect_model_disagreement,
    run_conflict_detection,
)


class TestLabelConflicts:
    def test_no_conflict(self):
        samples = [
            {"text": "投诉A", "label": "ACCEPT"},
            {"text": "投诉B", "label": "REJECT"},
        ]
        conflicts = detect_label_conflicts(samples)
        assert len(conflicts) == 0

    def test_single_conflict(self):
        samples = [
            {"text": "投诉A", "label": "ACCEPT"},
            {"text": "投诉A", "label": "REJECT"},
        ]
        conflicts = detect_label_conflicts(samples)
        assert len(conflicts) == 1
        assert conflicts[0]["conflict_type"] == "label_conflict"
        assert set(conflicts[0]["labels"]) == {"ACCEPT", "REJECT"}

    def test_empty(self):
        assert detect_label_conflicts([]) == []


class TestDecisionOverrides:
    def test_override_detected(self):
        traces = [
            {
                "trace_id": "t1",
                "classification": {"is_market": True},
            },
            {
                "trace_id": "t2",
                "classification": {"is_market": False},
            },
        ]
        reviews = [
            {"trace_id": "t1", "review": {"is_market": False}},
            {"trace_id": "t2", "review": {"is_market": False}},
        ]
        overrides = detect_decision_overrides(traces, reviews)
        assert len(overrides) == 1
        assert overrides[0]["trace_id"] == "t1"
        assert overrides[0]["original_is_market"] is True
        assert overrides[0]["review_is_market"] is False

    def test_no_review_match(self):
        traces = [{"trace_id": "t1", "classification": {"is_market": True}}]
        reviews = [{"trace_id": "t2", "review": {"is_market": False}}]
        assert detect_decision_overrides(traces, reviews) == []


class TestModelDisagreement:
    def test_disagreement(self):
        samples = [{"text": "测试", "label": "ACCEPT"}]
        disagreements = detect_model_disagreement(samples, lambda t: "REJECT")
        assert len(disagreements) == 1
        assert disagreements[0]["conflict_type"] == "model_disagreement"

    def test_agreement(self):
        samples = [{"text": "测试", "label": "ACCEPT"}]
        disagreements = detect_model_disagreement(samples, lambda t: "ACCEPT")
        assert len(disagreements) == 0

    def test_exception_handled(self):
        samples = [{"text": "测试", "label": "ACCEPT"}]
        disagreements = detect_model_disagreement(samples, lambda t: (_ for _ in ()).throw(Exception("err")))
        assert disagreements == []


class TestConflictReport:
    def test_to_dict(self):
        report = ConflictReport(total_samples=10, label_conflicts=[{"text": "x", "labels": ["A", "B"]}])
        d = report.to_dict()
        assert d["total_samples"] == 10
        assert d["label_conflict_count"] == 1


class TestRunConflictDetection:
    def test_with_csv(self, tmp_path: Path):
        csv_path = tmp_path / "train.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "label"])
            writer.writerow(["投诉A", "ACCEPT"])
            writer.writerow(["投诉A", "REJECT"])
        output = tmp_path / "conflicts.json"
        report = run_conflict_detection(training_csv=csv_path, output_path=output)
        assert report.total_samples == 2
        assert len(report.label_conflicts) == 1
        assert output.exists()

    def test_with_traces_reviews(self, tmp_path: Path):
        traces_path = tmp_path / "traces.jsonl"
        reviews_path = tmp_path / "reviews.jsonl"
        traces_path.write_text(
            json.dumps({"trace_id": "t1", "classification": {"is_market": True}}) + "\n",
            encoding="utf-8",
        )
        reviews_path.write_text(
            json.dumps({"trace_id": "t1", "review": {"is_market": False}}) + "\n",
            encoding="utf-8",
        )
        report = run_conflict_detection(traces_path=traces_path, reviews_path=reviews_path)
        assert len(report.decision_overrides) == 1
