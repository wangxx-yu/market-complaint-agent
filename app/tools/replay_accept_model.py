from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from app.agents.classifier import ClassifierAgent
from app.core.enums import AcceptSuggestion
from app.core.schemas import ComplaintAnalyzeRequest
from app.core.training_config import ACCEPT_TRAINING_CSV
from app.tools.train_accept_model import clean_text, load_clean_rows


DEFAULT_REPLAY_CSV = Path("data/training/accept_training_from_reviews.csv")
DEFAULT_OUT_DIR = Path("data/evaluation")


def choose_default_csv() -> Path:
    if DEFAULT_REPLAY_CSV.exists():
        return DEFAULT_REPLAY_CSV
    return ACCEPT_TRAINING_CSV


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def prediction_label(accept_suggestion: AcceptSuggestion) -> int | None:
    if accept_suggestion == AcceptSuggestion.ACCEPT:
        return 1
    if accept_suggestion == AcceptSuggestion.REJECT:
        return 0
    return None


def replay(args: argparse.Namespace) -> dict[str, Any]:
    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_rows, conflict_rows, dropped_rows = load_clean_rows(
        csv_path,
        args.text_col,
        args.label_col,
        args.min_chars,
    )
    if not clean_rows:
        raise ValueError("没有可回放样本，请检查 CSV")

    classifier = ClassifierAgent()
    y_true: list[int] = []
    y_pred: list[int] = []
    predictions: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    decision_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    review_reason_counts: Counter[str] = Counter()

    for row in clean_rows[: args.limit or None]:
        text = clean_text(row["text"])
        actual = int(row["label"])
        classification, review_reasons = classifier.classify(ComplaintAnalyzeRequest(problem_text=text))
        predicted = prediction_label(classification.accept_suggestion)
        decision = classification.accept_suggestion.value
        decision_counts[decision] += 1
        source_counts[classification.decision_source.value] += 1
        reason_counts[classification.reason_type.value] += 1
        for reason in review_reasons:
            review_reason_counts[reason] += 1

        item = {
            "row_number": row["row_number"],
            "actual": actual,
            "predicted": "" if predicted is None else predicted,
            "decision": decision,
            "is_market": classification.is_market,
            "reason_type": classification.reason_type.value,
            "confidence": round(classification.confidence, 4),
            "decision_source": classification.decision_source.value,
            "review_reasons": "；".join(review_reasons),
            "evidence_fields": "；".join(classification.evidence_fields),
            "text": text,
        }
        predictions.append(item)
        if predicted is None:
            review_rows.append(item)
            continue
        y_true.append(actual)
        y_pred.append(predicted)
        if actual != predicted:
            errors.append(item)

    auto_metrics: dict[str, Any]
    if y_true:
        auto_metrics = {
            "auto_accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "auto_confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
            "auto_classification_report": classification_report(
                y_true,
                y_pred,
                labels=[0, 1],
                target_names=["REJECT", "ACCEPT"],
                output_dict=True,
                zero_division=0,
            ),
        }
    else:
        auto_metrics = {
            "auto_accuracy": None,
            "auto_confusion_matrix": [[0, 0], [0, 0]],
            "auto_classification_report": {},
        }

    evaluated_rows = len(predictions)
    summary = {
        "source_csv": str(csv_path),
        "evaluated_rows": evaluated_rows,
        "clean_rows_available": len(clean_rows),
        "conflict_rows": len(conflict_rows),
        "dropped_rows": len(dropped_rows),
        "label_counts": dict(Counter(row["label"] for row in clean_rows)),
        "decision_counts": dict(decision_counts),
        "review_rows": len(review_rows),
        "auto_decision_rows": len(y_true),
        "error_rows": len(errors),
        "review_rate": round(len(review_rows) / evaluated_rows, 4) if evaluated_rows else 0,
        "source_counts": dict(source_counts),
        "reason_counts": dict(reason_counts),
        "top_review_reasons": dict(review_reason_counts.most_common(10)),
        "outputs": {
            "summary_json": (out_dir / "accept_replay_summary.json").as_posix(),
            "predictions_csv": (out_dir / "accept_replay_predictions.csv").as_posix(),
            "errors_csv": (out_dir / "accept_replay_errors.csv").as_posix(),
            "review_csv": (out_dir / "accept_replay_review.csv").as_posix(),
        },
        **auto_metrics,
    }

    (out_dir / "accept_replay_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    fieldnames = [
        "row_number",
        "actual",
        "predicted",
        "decision",
        "is_market",
        "reason_type",
        "confidence",
        "decision_source",
        "review_reasons",
        "evidence_fields",
        "text",
    ]
    write_csv(out_dir / "accept_replay_predictions.csv", predictions, fieldnames)
    write_csv(out_dir / "accept_replay_errors.csv", errors, fieldnames)
    write_csv(out_dir / "accept_replay_review.csv", review_rows, fieldnames)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="批量回放是否受理模型，统计建议受理/不受理/人工复核和错判样本。")
    parser.add_argument("--csv", default=str(choose_default_csv()), help="回放 CSV，默认优先使用 data/training/accept_training_from_reviews.csv")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--min-chars", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="只回放前 N 条；0 表示全部")
    args = parser.parse_args()

    print(json.dumps(replay(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
