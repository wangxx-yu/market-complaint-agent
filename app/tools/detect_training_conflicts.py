"""训练数据冲突检测 — 检测标注冲突、决策覆盖、模型不一致。"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ConflictReport:
    total_samples: int = 0
    label_conflicts: list[dict[str, Any]] = field(default_factory=list)
    decision_overrides: list[dict[str, Any]] = field(default_factory=list)
    model_disagreements: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "label_conflict_count": len(self.label_conflicts),
            "decision_override_count": len(self.decision_overrides),
            "model_disagreement_count": len(self.model_disagreements),
            "label_conflicts": self.label_conflicts[:50],
            "decision_overrides": self.decision_overrides[:50],
            "model_disagreements": self.model_disagreements[:50],
        }


def detect_label_conflicts(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """检测同一文本的不同标注（label 冲突）。"""
    text_groups: dict[str, list[dict[str, Any]]] = {}
    for s in samples:
        text = s.get("text", "")
        if text not in text_groups:
            text_groups[text] = []
        text_groups[text].append(s)

    conflicts: list[dict[str, Any]] = []
    for text, group in text_groups.items():
        labels = {s.get("label") for s in group}
        if len(labels) > 1:
            conflicts.append({
                "text": text[:100],
                "labels": list(labels),
                "count": len(group),
                "conflict_type": "label_conflict",
            })
    return conflicts


def detect_decision_overrides(
    traces: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """检测复核结果与系统原决策不一致的情况。"""
    review_map: dict[str, dict[str, Any]] = {}
    for r in reviews:
        tid = r.get("trace_id")
        if tid:
            review_map[tid] = r

    overrides: list[dict[str, Any]] = []
    for trace in traces:
        tid = trace.get("trace_id")
        if tid not in review_map:
            continue
        review = review_map[tid]
        original_is_market = trace.get("classification", {}).get("is_market")
        review_is_market = review.get("review", {}).get("is_market")
        if original_is_market is not None and review_is_market is not None:
            if original_is_market != review_is_market:
                overrides.append({
                    "trace_id": tid,
                    "original_is_market": original_is_market,
                    "review_is_market": review_is_market,
                    "conflict_type": "decision_override",
                })
    return overrides


def detect_model_disagreement(
    samples: list[dict[str, Any]],
    model_predict_fn,
) -> list[dict[str, Any]]:
    """用当前模型预测样本，标记预测与标注不一致的。"""
    disagreements: list[dict[str, Any]] = []
    for s in samples:
        text = s.get("text", "")
        label = s.get("label")
        if not text or label is None:
            continue
        try:
            pred = model_predict_fn(text)
            if pred != label:
                disagreements.append({
                    "text": text[:100],
                    "expected_label": label,
                    "predicted_label": pred,
                    "conflict_type": "model_disagreement",
                })
        except Exception:
            pass
    return disagreements


def run_conflict_detection(
    training_csv: Path | None = None,
    traces_path: Path | None = None,
    reviews_path: Path | None = None,
    output_path: Path | None = None,
) -> ConflictReport:
    report = ConflictReport()

    # 从 CSV 加载训练样本
    if training_csv and training_csv.exists():
        import csv
        samples: list[dict[str, Any]] = []
        with open(training_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                samples.append({"text": row.get("text", ""), "label": row.get("label")})
        report.total_samples = len(samples)
        report.label_conflicts = detect_label_conflicts(samples)

    # 从 JSONL 加载 traces/reviews
    if traces_path and reviews_path and traces_path.exists() and reviews_path.exists():
        traces = _read_jsonl(traces_path)
        reviews = _read_jsonl(reviews_path)
        report.decision_overrides = detect_decision_overrides(traces, reviews)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return report


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    report = run_conflict_detection()
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
