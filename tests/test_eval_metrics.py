"""评估指标扩展测试（Wave 6.2）。"""
from __future__ import annotations

from eval.run_evaluation import EvalReport, EvalMetrics


class TestEvalReportCompliance:
    def test_false_rejection_rate(self):
        report = EvalReport()
        report.classification.total = 10
        report.false_rejection_count = 3
        assert report.false_rejection_rate == 0.3

    def test_false_rejection_rate_zero(self):
        report = EvalReport()
        assert report.false_rejection_rate == 0.0

    def test_review_rate(self):
        report = EvalReport()
        report.classification.total = 10
        report.review_count = 5
        assert report.review_rate == 0.5

    def test_reply_compliance_rate(self):
        report = EvalReport()
        report.classification.total = 10
        report.reply_compliance_pass = 8
        assert report.reply_compliance_rate == 0.8

    def test_to_dict_includes_compliance(self):
        report = EvalReport()
        report.classification.total = 10
        report.false_rejection_count = 2
        report.review_count = 4
        report.reply_compliance_pass = 9
        d = report.to_dict()
        assert "false_rejection_rate" in d
        assert "review_rate" in d
        assert "reply_compliance_rate" in d
        assert d["false_rejection_rate"] == 0.2
        assert d["review_rate"] == 0.4
        assert d["reply_compliance_rate"] == 0.9
