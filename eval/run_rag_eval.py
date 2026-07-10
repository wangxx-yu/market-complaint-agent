# -*- coding: utf-8 -*-
"""RAG 检索评估器 — 评估法规检索的召回率和准确率。

用法:
    python -m eval.run_rag_eval
    python -m eval.run_rag_eval --output eval/results/rag_eval_result.json
"""
from __future__ import annotations

import json
import time
from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.agents.retrieval import RetrievalAgent
from app.core.enums import ReasonType


@dataclass
class RAGEvalSample:
    id: str
    query: str
    reason_type: str = "UNKNOWN"
    expected_title: str = ""
    expected_article: str = ""
    min_expected_score: float = 0.0
    category: str = ""


@dataclass
class RAGEvalResult:
    sample_id: str
    hits: list[dict[str, Any]] = field(default_factory=list)
    recall_hit: bool = False
    recall_at_1: bool = False
    recall_at_3: bool = False
    mrr_rank: int | None = None
    top_score: float = 0.0


@dataclass
class RAGEvalReport:
    total: int = 0
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    mrr: float = 0.0
    avg_score: float = 0.0
    found_results: int = 0
    no_results: int = 0
    per_category: dict[str, dict[str, Any]] = field(default_factory=dict)
    details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "recall_at_1": round(self.recall_at_1, 4),
            "recall_at_3": round(self.recall_at_3, 4),
            "mrr": round(self.mrr, 4),
            "avg_score": round(self.avg_score, 4),
            "found_results": self.found_results,
            "no_results": self.no_results,
            "per_category": self.per_category,
        }


def load_rag_eval_set(path: Path | None = None) -> list[RAGEvalSample]:
    path = path or Path("data/evaluation/rag_eval_set.json")
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [RAGEvalSample(**item) for item in data]


def compute_recall_at_k(expected: list[str], retrieved: list[str], k: int = 1) -> float:
    """Recall@K: 期望相关项中至少有一个出现在 top-K 检索结果中则返回 1.0。"""
    if not expected:
        return 1.0
    if not retrieved:
        return 0.0
    top_k = retrieved[:k]
    for e in expected:
        if any(e in r for r in top_k):
            return 1.0
    return 0.0


def _match_hit(expected_title: str, expected_article: str, hit_dict: dict[str, Any]) -> bool:
    title = hit_dict.get("title", "") or ""
    content = hit_dict.get("content", "") or ""
    if expected_title and expected_title in title:
        return True
    if expected_article and (expected_article in title or expected_article in content):
        return True
    return False


def run_rag_eval(output_dir: Path | None = None, top_k: int = 3) -> RAGEvalReport:
    samples = load_rag_eval_set()
    if not samples:
        return RAGEvalReport()

    retriever = RetrievalAgent()
    report = RAGEvalReport(total=len(samples))
    reciprocal_ranks: list[float] = []
    scores: list[float] = []
    cat_stats: dict[str, dict[str, Any]] = {}

    for sample in samples:
        reason_type = ReasonType(sample.reason_type)
        try:
            hits = retriever.retrieve(reason_type=reason_type, query=sample.query, top_k=top_k)
        except Exception:
            hits = []

        hit_dicts = [{"title": h.title, "content": h.content, "score": h.score} for h in hits]

        recall_at_1 = False
        recall_at_3 = False
        mrr_rank = None

        if hits:
            report.found_results += 1
            recall_at_1 = _match_hit(sample.expected_title, sample.expected_article, hit_dicts[0]) if hit_dicts else False
            recall_at_3 = any(_match_hit(sample.expected_title, sample.expected_article, h) for h in hit_dicts[:3])
            for rank, h in enumerate(hit_dicts, start=1):
                if _match_hit(sample.expected_title, sample.expected_article, h):
                    mrr_rank = rank
                    break
            scores.extend(h.score for h in hits[:top_k])
            if recall_at_1:
                report.recall_at_1 += 1
            if recall_at_3:
                report.recall_at_3 += 1
        else:
            report.no_results += 1

        reciprocal_ranks.append(1.0 / mrr_rank if mrr_rank else 0.0)

        # Per-category aggregation
        cat = sample.category
        if cat not in cat_stats:
            cat_stats[cat] = {"total": 0, "recall_at_1": 0, "recall_at_3": 0, "mrr_sum": 0.0, "no_results": 0}
        cat_stats[cat]["total"] += 1
        if recall_at_1:
            cat_stats[cat]["recall_at_1"] += 1
        if recall_at_3:
            cat_stats[cat]["recall_at_3"] += 1
        cat_stats[cat]["mrr_sum"] += reciprocal_ranks[-1]
        if not hits:
            cat_stats[cat]["no_results"] += 1

    if report.total > 0:
        report.recall_at_1 /= report.total
        report.recall_at_3 /= report.total
        report.mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
        report.avg_score = sum(scores) / len(scores) if scores else 0.0

    for cat, stats in cat_stats.items():
        total = stats["total"]
        stats["recall_at_1"] = round(stats["recall_at_1"] / total, 4) if total else 0.0
        stats["recall_at_3"] = round(stats["recall_at_3"] / total, 4) if total else 0.0
        stats["mrr"] = round(stats["mrr_sum"] / total, 4) if total else 0.0
        del stats["mrr_sum"]
    report.per_category = cat_stats

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / "rag_eval_result.json"
        result_path.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return report


def main() -> None:
    parser = ArgumentParser(description="RAG 检索评估")
    parser.add_argument("--eval-set", default="data/evaluation/rag_eval_set.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--category", default=None, help="只评估指定分类")
    args = parser.parse_args()

    report = run_rag_eval(output_dir=Path(args.output) if args.output else None)
    if report.total == 0:
        print("无评估数据，退出。")
        return

    print(f"评估集: {report.total} 条")
    print(f"  Recall@1:        {report.recall_at_1:.2%}")
    print(f"  Recall@3:        {report.recall_at_3:.2%}")
    print(f"  MRR:             {report.mrr:.4f}")
    print(f"  Avg Score:       {report.avg_score:.4f}")
    print(f"  Found: {report.found_results}  NoResult: {report.no_results}")


if __name__ == "__main__":
    main()