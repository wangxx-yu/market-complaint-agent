from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from app.tools.export_review_training_data import extract_problem_text, read_jsonl_lenient


def get_system_dispatch(trace: dict[str, Any]) -> dict[str, Any]:
    dispatch = trace.get("dispatch") or {}
    return {
        "system_office_code": dispatch.get("office_code") or "",
        "system_office_name": dispatch.get("office_name") or "",
        "matched_rule": dispatch.get("matched_rule") or "",
        "confidence": dispatch.get("confidence") if dispatch.get("confidence") is not None else "",
        "needs_review": dispatch.get("needs_review") if dispatch.get("needs_review") is not None else "",
    }


def get_review_dispatch(review: dict[str, Any]) -> dict[str, str]:
    payload = review.get("review") or {}
    return {
        "review_office_code": payload.get("office_code") or "",
        "review_office_name": payload.get("office_name") or "",
    }


def extract_address_hint(trace: dict[str, Any]) -> str:
    for step in trace.get("agent_steps", []):
        if step.get("name") == "dispatch":
            return str(step.get("input_summary", {}).get("address") or "")
    return ""


def export(args: argparse.Namespace) -> dict[str, Any]:
    traces, bad_trace_rows = read_jsonl_lenient(Path(args.traces))
    reviews, bad_review_rows = read_jsonl_lenient(Path(args.reviews))
    trace_by_id = {trace.get("trace_id"): trace for trace in traces if trace.get("trace_id")}

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for review in reviews:
        trace_id = review.get("trace_id")
        trace = trace_by_id.get(trace_id)
        if not trace:
            skipped.append({"trace_id": trace_id, "reason": "trace_not_found"})
            continue
        review_dispatch = get_review_dispatch(review)
        if not review_dispatch["review_office_code"] and not review_dispatch["review_office_name"]:
            skipped.append({"trace_id": trace_id, "reason": "missing_review_office"})
            continue
        system_dispatch = get_system_dispatch(trace)
        changed = bool(
            review_dispatch["review_office_code"]
            and system_dispatch["system_office_code"]
            and review_dispatch["review_office_code"] != system_dispatch["system_office_code"]
        )
        rows.append(
            {
                "trace_id": trace_id,
                "problem_text": extract_problem_text(trace),
                "address_hint": extract_address_hint(trace),
                **system_dispatch,
                **review_dispatch,
                "changed": changed,
            }
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "trace_id",
        "problem_text",
        "address_hint",
        "system_office_code",
        "system_office_name",
        "review_office_code",
        "review_office_name",
        "matched_rule",
        "confidence",
        "needs_review",
        "changed",
    ]
    with (out_dir / "dispatch_review_data.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "dispatch_review_changes.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows([row for row in rows if row["changed"]])
    with (out_dir / "dispatch_review_skipped.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["trace_id", "reason"])
        writer.writeheader()
        writer.writerows(skipped)

    system_counts = Counter(row["system_office_name"] or "<空>" for row in rows)
    review_counts = Counter(row["review_office_name"] or "<空>" for row in rows)
    changed_counts = Counter(
        f'{row["system_office_name"] or "<空>"} -> {row["review_office_name"] or "<空>"}'
        for row in rows
        if row["changed"]
    )
    summary = {
        "reviews_read": len(reviews),
        "rows_exported": len(rows),
        "changed_rows": sum(1 for row in rows if row["changed"]),
        "skipped_rows": len(skipped),
        "bad_json_rows": len(bad_trace_rows) + len(bad_review_rows),
        "system_office_counts": dict(system_counts.most_common()),
        "review_office_counts": dict(review_counts.most_common()),
        "changed_counts": dict(changed_counts.most_common(20)),
    }
    (out_dir / "dispatch_review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="导出系统分派与人工确认分派的对比表。")
    parser.add_argument("--traces", default="data/runtime/traces.jsonl")
    parser.add_argument("--reviews", default="data/runtime/reviews.jsonl")
    parser.add_argument("--out-dir", default="data/training")
    args = parser.parse_args()
    print(json.dumps(export(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

