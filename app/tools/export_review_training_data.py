from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from app.core.training_config import ACCEPT_TRAINING_CSV
from app.tools.train_accept_model import clean_text, normalize_label


def read_jsonl_lenient(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    bad_rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows, [{"line_number": 0, "reason": "file_not_found", "raw": str(path)}]
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                bad_rows.append({"line_number": line_number, "reason": str(exc), "raw": raw[:500]})
    return rows, bad_rows


def extract_problem_text(trace: dict[str, Any]) -> str:
    for step in trace.get("agent_steps", []):
        if step.get("name") == "preprocess":
            text = step.get("output_summary", {}).get("problem_text")
            if text:
                return clean_text(text)
        if step.get("name") == "classify":
            text = step.get("input_summary", {}).get("problem_text")
            if text:
                return clean_text(text)
    return ""


def review_to_label(review: dict[str, Any]) -> int | None:
    payload = review.get("review", {})
    is_market = payload.get("is_market")
    if isinstance(is_market, bool):
        return 1 if is_market else 0
    return None


def read_base_training_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or "text" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(f"基础训练集必须包含 text,label 两列: {path}")
        for row_number, row in enumerate(reader, start=2):
            text = clean_text(row.get("text"))
            label = normalize_label(row.get("label"))
            if text and label is not None:
                rows.append({"text": text, "label": label, "source": "base", "source_row": row_number})
    return rows


def dedupe_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels_by_text: dict[str, set[int]] = {}
    for row in rows:
        labels_by_text.setdefault(row["text"], set()).add(int(row["label"]))

    clean: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for row in rows:
        labels = labels_by_text[row["text"]]
        if len(labels) > 1:
            conflicts.append({**row, "labels": ",".join(str(label) for label in sorted(labels))})
            continue
        key = (row["text"], int(row["label"]))
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
    skipped_reviews: list[dict[str, Any]] = []
    for review in reviews:
        trace_id = review.get("trace_id")
        label = review_to_label(review)
        trace = trace_by_id.get(trace_id)
        text = extract_problem_text(trace) if trace else ""
        if not trace:
            skipped_reviews.append({"trace_id": trace_id, "reason": "trace_not_found"})
            continue
        if label is None:
            skipped_reviews.append({"trace_id": trace_id, "reason": "missing_is_market"})
            continue
        if not text:
            skipped_reviews.append({"trace_id": trace_id, "reason": "missing_problem_text"})
            continue
        review_rows.append(
            {
                "text": text,
                "label": label,
                "source": "review",
                "source_row": trace_id,
            }
        )

    base_rows = read_base_training_csv(Path(args.base_csv)) if args.include_base else []
    merged_rows, conflict_rows = dedupe_rows([*base_rows, *review_rows])
    label_counts = Counter(int(row["label"]) for row in merged_rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_csv = out_dir / args.output_name
    write_csv(output_csv, merged_rows, ["text", "label", "source", "source_row"])
    write_csv(out_dir / "review_training_only.csv", review_rows, ["text", "label", "source", "source_row"])
    write_csv(out_dir / "review_export_conflicts.csv", conflict_rows, ["text", "label", "source", "source_row", "labels"])
    write_csv(out_dir / "review_export_skipped.csv", skipped_reviews, ["trace_id", "reason"])
    write_csv(out_dir / "review_export_bad_json.csv", [*bad_trace_rows, *bad_review_rows], ["line_number", "reason", "raw"])

    summary = {
        "output_csv": str(output_csv),
        "base_rows": len(base_rows),
        "reviews_read": len(reviews),
        "review_rows_exported": len(review_rows),
        "merged_rows": len(merged_rows),
        "conflict_rows": len(conflict_rows),
        "skipped_reviews": len(skipped_reviews),
        "bad_json_rows": len(bad_trace_rows) + len(bad_review_rows),
        "label_counts": dict(label_counts),
    }
    (out_dir / "review_export_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="把人工复核结果导出为新的是否受理训练集。")
    parser.add_argument("--base-csv", default=str(ACCEPT_TRAINING_CSV), help="原始基础训练集 text,label")
    parser.add_argument("--traces", default="data/runtime/traces.jsonl")
    parser.add_argument("--reviews", default="data/runtime/reviews.jsonl")
    parser.add_argument("--out-dir", default="data/training")
    parser.add_argument("--output-name", default="accept_training_v2.csv")
    parser.add_argument("--no-base", action="store_false", dest="include_base", help="只导出人工复核样本，不合并基础训练集")
    parser.set_defaults(include_base=True)
    args = parser.parse_args()

    summary = export(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

