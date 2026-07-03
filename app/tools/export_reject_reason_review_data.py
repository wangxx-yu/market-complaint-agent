from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from app.core.enums import ReasonType
from app.tools.export_review_training_data import extract_problem_text, read_jsonl_lenient
from app.tools.train_reject_reason_model import clean_text, normalize_reason_type


DEFAULT_BASE_CSV = Path("data/training/reject_reason_trainable_only.csv")


def read_base_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or "text" not in reader.fieldnames or "reason_type" not in reader.fieldnames:
            raise ValueError(f"基础训练集必须包含 text,reason_type 两列: {path}")
        for row_number, row in enumerate(reader, start=2):
            reason_type = normalize_reason_type(row.get("reason_type"))
            text = clean_text(row.get("text"))
            feedback = clean_text(row.get("feedback"))
            if not text or not reason_type:
                continue
            rows.append(
                {
                    "source": row.get("source") or "base",
                    "source_row": row.get("source_row") or row_number,
                    "text": text,
                    "feedback": feedback,
                    "reason_type": reason_type,
                    "suggested_department": row.get("suggested_department", ""),
                    "trainable": 1,
                    "notes": row.get("note", ""),
                }
            )
    return rows


def extract_feedback(trace: dict[str, Any], review: dict[str, Any]) -> str:
    reject_detail = clean_text(review.get("review", {}).get("reject_detail"))
    if reject_detail:
        return reject_detail
    review_text = clean_text(review.get("review", {}).get("reply_text"))
    if review_text:
        return review_text
    reply = trace.get("reply_draft", {})
    if isinstance(reply, dict):
        return clean_text(reply.get("text"))
    return ""


def extract_suggested_department(review: dict[str, Any], trace: dict[str, Any]) -> str:
    classification = trace.get("classification", {})
    for field in classification.get("evidence_fields", []):
        if isinstance(field, str) and field.startswith("suggest_department="):
            return field.removeprefix("suggest_department=")
    return ""


def review_to_reject_reason_row(review: dict[str, Any], trace: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    payload = review.get("review", {})
    if payload.get("is_market") is not False:
        return None, "not_reject_review"
    reason_type = normalize_reason_type(payload.get("reason_type"))
    if reason_type is None:
        return None, "missing_or_unknown_reason_type"
    text = extract_problem_text(trace)
    if not text:
        return None, "missing_problem_text"
    return (
        {
            "source": "review",
            "source_row": review.get("trace_id", ""),
            "text": text,
            "feedback": extract_feedback(trace, review),
            "reason_type": reason_type,
            "suggested_department": extract_suggested_department(review, trace),
            "trainable": 1,
            "notes": f"reviewer={payload.get('reviewer') or ''}",
        },
        None,
    )


def dedupe_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels_by_text: dict[str, set[str]] = {}
    for row in rows:
        labels_by_text.setdefault(row["text"], set()).add(row["reason_type"])

    clean: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        labels = labels_by_text[row["text"]]
        if len(labels) > 1:
            conflicts.append({**row, "conflict_reason_types": "|".join(sorted(labels))})
            continue
        key = (row["text"], row["reason_type"])
        if key in seen:
            continue
        seen.add(key)
        clean.append(row)
    return clean, conflicts


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export(args: argparse.Namespace) -> dict[str, Any]:
    traces, bad_trace_rows = read_jsonl_lenient(Path(args.traces))
    reviews, bad_review_rows = read_jsonl_lenient(Path(args.reviews))
    trace_by_id = {trace.get("trace_id"): trace for trace in traces if trace.get("trace_id")}

    review_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    for review in reviews:
        trace_id = review.get("trace_id")
        trace = trace_by_id.get(trace_id)
        if trace is None:
            skipped_rows.append({"trace_id": trace_id, "reason": "trace_not_found"})
            continue
        row, skip_reason = review_to_reject_reason_row(review, trace)
        if skip_reason:
            skipped_rows.append({"trace_id": trace_id, "reason": skip_reason})
            continue
        if row:
            review_rows.append(row)

    base_rows = read_base_rows(Path(args.base_csv)) if args.include_base else []
    merged_rows, conflict_rows = dedupe_rows([*base_rows, *review_rows])
    reason_counts = Counter(row["reason_type"] for row in merged_rows)
    source_counts = Counter(row["source"] for row in merged_rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_csv = out_dir / args.output_name
    fieldnames = ["source", "source_row", "text", "feedback", "reason_type", "suggested_department", "trainable", "notes"]
    write_csv(output_csv, merged_rows, fieldnames)
    write_csv(out_dir / "reject_reason_review_only.csv", review_rows, fieldnames)
    write_csv(out_dir / "reject_reason_review_skipped.csv", skipped_rows, ["trace_id", "reason"])
    write_csv(out_dir / "reject_reason_review_conflicts.csv", conflict_rows, [*fieldnames, "conflict_reason_types"])
    write_csv(out_dir / "reject_reason_review_bad_json.csv", [*bad_trace_rows, *bad_review_rows], ["line_number", "reason", "raw"])

    summary = {
        "output_csv": str(output_csv),
        "base_rows": len(base_rows),
        "reviews_read": len(reviews),
        "review_rows_exported": len(review_rows),
        "merged_rows": len(merged_rows),
        "conflict_rows": len(conflict_rows),
        "skipped_rows": len(skipped_rows),
        "bad_json_rows": len(bad_trace_rows) + len(bad_review_rows),
        "reason_counts": dict(reason_counts),
        "source_counts": dict(source_counts),
    }
    (out_dir / "reject_reason_review_export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="导出人工复核的不受理原因样本，并可合并基础不受理原因训练集。")
    parser.add_argument("--base-csv", default=str(DEFAULT_BASE_CSV))
    parser.add_argument("--traces", default="data/runtime/traces.jsonl")
    parser.add_argument("--reviews", default="data/runtime/reviews.jsonl")
    parser.add_argument("--out-dir", default="data/training")
    parser.add_argument("--output-name", default="reject_reason_training_v2.csv")
    parser.add_argument("--no-base", action="store_false", dest="include_base", help="只导出人工复核样本，不合并基础训练集")
    parser.set_defaults(include_base=True)
    args = parser.parse_args()

    summary = export(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
