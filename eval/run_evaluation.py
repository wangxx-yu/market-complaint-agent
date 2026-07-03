"""
Agent 评估框架 — 离线评估分类/分派/检索/端到端准确率。

用法:
    python -m eval.run_evaluation              # 全部评估
    python -m eval.run_evaluation --quick       # 快速抽查
    python -m eval.run_evaluation --output eval/results.json  # 导出结果
"""

from __future__ import annotations

import json
import time
from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval.golden_dataset import GOLDEN_SAMPLES, GoldenSample


# ── 评估指标 ──

@dataclass
class EvalMetrics:
    total: int = 0
    correct: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 4),
            "error_count": len(self.errors),
            "errors": self.errors[:20],
        }


@dataclass
class EvalReport:
    classification: EvalMetrics = field(default_factory=EvalMetrics)
    dispatch: EvalMetrics = field(default_factory=EvalMetrics)
    retrieval: EvalMetrics = field(default_factory=EvalMetrics)
    end_to_end: EvalMetrics = field(default_factory=EvalMetrics)
    latency_ms: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification.to_dict(),
            "dispatch": self.dispatch.to_dict(),
            "retrieval": self.retrieval.to_dict(),
            "end_to_end": self.end_to_end.to_dict(),
            "latency": {
                "count": len(self.latency_ms),
                "avg_ms": round(sum(self.latency_ms) / len(self.latency_ms), 1) if self.latency_ms else 0,
                "p50_ms": round(sorted(self.latency_ms)[len(self.latency_ms) // 2], 1) if self.latency_ms else 0,
                "p95_ms": round(sorted(self.latency_ms)[int(len(self.latency_ms) * 0.95)], 1) if self.latency_ms else 0,
            },
        }


# ── 评估执行器 ──

def evaluate_all(samples: list[GoldenSample] | None = None, quick: bool = False) -> EvalReport:
    """Run full evaluation suite against golden dataset."""
    from app.agents.classifier import ClassifierAgent
    from app.agents.dispatch import DispatchAgent
    from app.agents.retrieval import RetrievalAgent
    from app.core.schemas import ComplaintAnalyzeRequest

    if samples is None:
        samples = GOLDEN_SAMPLES
    if quick:
        samples = samples[:5]

    report = EvalReport()
    classifier = ClassifierAgent()
    dispatch_agent = DispatchAgent()
    retrieval_agent = RetrievalAgent()

    for sample in samples:
        request = ComplaintAnalyzeRequest(
            problem_text=sample.problem_text,
            incident_location=sample.incident_location,
            enterprise_address=sample.enterprise_address,
        )

        # ── Classification ──
        report.classification.total += 1
        t0 = time.perf_counter()
        result, _ = classifier.classify(request)
        report.latency_ms.append((time.perf_counter() - t0) * 1000)

        cls_ok = True
        if sample.expected_is_market is not None and result.is_market != sample.expected_is_market:
            cls_ok = False
        if sample.expected_accept_suggestion and result.accept_suggestion.value != sample.expected_accept_suggestion:
            cls_ok = False
        if sample.expected_reason_type and result.reason_type.value != sample.expected_reason_type:
            cls_ok = False

        if cls_ok:
            report.classification.correct += 1
        else:
            report.classification.errors.append({
                "sample_id": sample.id,
                "problem_text": sample.problem_text[:100],
                "expected": {
                    "is_market": sample.expected_is_market,
                    "accept": sample.expected_accept_suggestion,
                    "reason": sample.expected_reason_type,
                },
                "actual": {
                    "is_market": result.is_market,
                    "accept": result.accept_suggestion.value,
                    "reason": result.reason_type.value,
                    "confidence": result.confidence,
                },
            })

        # ── Dispatch (only for market-regulation samples) ──
        if sample.expected_office_name:
            report.dispatch.total += 1
            dispatch_result = dispatch_agent.dispatch(request)
            if dispatch_result.office_name == sample.expected_office_name:
                report.dispatch.correct += 1
            else:
                report.dispatch.errors.append({
                    "sample_id": sample.id,
                    "expected_office": sample.expected_office_name,
                    "actual_office": dispatch_result.office_name,
                    "actual_confidence": dispatch_result.confidence,
                })

        # ── Retrieval ──
        if sample.expected_department:
            report.retrieval.total += 1
            from app.core.enums import ReasonType
            reason_type = ReasonType(sample.expected_reason_type) if sample.expected_reason_type else ReasonType.ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY
            hits = retrieval_agent.retrieve(reason_type, sample.problem_text, top_k=3)
            if hits and hits[0].suggested_department == sample.expected_department:
                report.retrieval.correct += 1
            else:
                report.retrieval.errors.append({
                    "sample_id": sample.id,
                    "expected_department": sample.expected_department,
                    "actual_top_hit": {
                        "title": hits[0].title if hits else None,
                        "department": hits[0].suggested_department if hits else None,
                        "score": hits[0].score if hits else 0,
                    },
                })

        # ── End-to-end: passes if classification + dispatch (when applicable) + retrieval are all correct ──
        report.end_to_end.total += 1
        all_ok = cls_ok
        if sample.expected_office_name:
            if not (report.dispatch.errors and report.dispatch.errors[-1].get("sample_id") == sample.id):
                pass  # dispatch was correct
            else:
                all_ok = False
        if sample.expected_department:
            if not (report.retrieval.errors and report.retrieval.errors[-1].get("sample_id") == sample.id):
                pass  # retrieval was correct
            else:
                all_ok = False
        if all_ok:
            report.end_to_end.correct += 1
        else:
            report.end_to_end.errors.append({"sample_id": sample.id, "problem_text": sample.problem_text[:100]})

    return report


# ── CLI ──

def main() -> None:
    parser = ArgumentParser(description="Agent 评估框架")
    parser.add_argument("--quick", action="store_true", help="快速抽查（仅 5 条样本）")
    parser.add_argument("--output", type=str, default=None, help="结果导出路径 (JSON)")
    args = parser.parse_args()

    print("=" * 60)
    print("  市场监管投诉智能处理系统 — Agent 评估报告")
    print("=" * 60)

    report = evaluate_all(quick=args.quick)

    def print_metrics(name: str, m: EvalMetrics) -> None:
        print(f"\n  [{name}]")
        print(f"    样本数: {m.total}")
        print(f"    正确数: {m.correct}")
        print(f"    准确率: {m.accuracy:.2%}")
        if m.errors:
            print(f"    错误数: {len(m.errors)}")
            for err in m.errors[:3]:
                print(f"      ✗ {err.get('sample_id', '?')}: {err.get('problem_text', '')[:60]}...")

    print_metrics("分类 (Classification)", report.classification)
    print_metrics("分派 (Dispatch)     ", report.dispatch)
    print_metrics("检索 (Retrieval)    ", report.retrieval)
    print_metrics("端到端 (End-to-End) ", report.end_to_end)

    lat = report.to_dict()["latency"]
    print(f"\n  [延迟]")
    print(f"    样本数: {lat['count']}")
    print(f"    平均: {lat['avg_ms']}ms  P50: {lat['p50_ms']}ms  P95: {lat['p95_ms']}ms")

    print("\n" + "=" * 60)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果已导出: {output_path}")


if __name__ == "__main__":
    main()
