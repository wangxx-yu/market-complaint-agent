from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.core.enums import ReasonType


DEFAULT_CSV = Path("data/training/reject_reason_trainable_only.csv")
DEFAULT_MODEL_DIR = Path("models/reject_reason_v1")

PHONE_RE = re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)")
ID_RE = re.compile(r"(?<!\d)(\d{17}[\dXx])(?!\d)")


def clean_text(value: str | None, max_chars: int = 2000) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    text = PHONE_RE.sub(lambda match: match.group(1)[:3] + "****" + match.group(1)[7:], text)
    text = ID_RE.sub(lambda match: match.group(1)[:6] + "********" + match.group(1)[14:], text)
    return text[:max_chars]


def normalize_reason_type(value: str | None) -> str | None:
    label = "" if value is None else str(value).strip()
    if not label or label == ReasonType.UNKNOWN.value:
        return None
    valid = {reason.value for reason in ReasonType if reason != ReasonType.UNKNOWN}
    return label if label in valid else None


def read_rows(csv_path: Path, text_col: str, feedback_col: str, label_col: str, min_chars: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid_rows: list[dict[str, Any]] = []
    dropped_rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError("CSV 没有表头")
        missing = [name for name in [text_col, label_col] if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV 缺少列: {', '.join(missing)}")
        for row_number, row in enumerate(reader, start=2):
            text = clean_text(row.get(text_col))
            feedback = clean_text(row.get(feedback_col)) if feedback_col in reader.fieldnames else ""
            label = normalize_reason_type(row.get(label_col))
            trainable = str(row.get("trainable", "1")).strip()
            if trainable not in {"1", "True", "true"}:
                dropped_rows.append({"row_number": row_number, "reason": "not_trainable", **row})
                continue
            if label is None:
                dropped_rows.append({"row_number": row_number, "reason": "invalid_reason_type", **row})
                continue
            combined_text = clean_text(f"{text} {feedback}")
            if len(combined_text) < min_chars:
                dropped_rows.append({"row_number": row_number, "reason": "text_empty_or_too_short", **row})
                continue
            valid_rows.append(
                {
                    "row_number": row_number,
                    "text": combined_text,
                    "problem_text": text,
                    "feedback": feedback,
                    "label": label,
                }
            )

    by_text: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid_rows:
        by_text[row["text"]].append(row)

    clean_rows: list[dict[str, Any]] = []
    for text, rows in by_text.items():
        labels = {row["label"] for row in rows}
        if len(labels) > 1:
            for row in rows:
                dropped_rows.append({**row, "reason": "same_text_conflicting_reason_type"})
            continue
        clean_rows.append(rows[0])

    return clean_rows, dropped_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    min_df=1,
                    max_df=0.98,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


def split_rows(rows: list[dict[str, Any]], test_size: float, random_state: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels = [row["label"] for row in rows]
    counts = Counter(labels)
    if len(counts) < 2:
        raise ValueError(f"至少需要两个原因类型才能训练，当前分布: {dict(counts)}")
    if min(counts.values()) < 2:
        raise ValueError(f"每个原因类型至少需要 2 条才能分层切分，当前分布: {dict(counts)}")
    return train_test_split(rows, test_size=test_size, random_state=random_state, stratify=labels)


def train(args: argparse.Namespace) -> dict[str, Any]:
    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_rows, dropped_rows = read_rows(csv_path, args.text_col, args.feedback_col, args.label_col, args.min_chars)
    if len(clean_rows) < args.min_samples:
        raise ValueError(f"可训练样本少于 {args.min_samples} 条，请先补充或检查不受理原因样本")

    label_counts = Counter(row["label"] for row in clean_rows)
    train_rows, test_rows = split_rows(clean_rows, args.test_size, args.random_state)

    x_train = [row["text"] for row in train_rows]
    y_train = [row["label"] for row in train_rows]
    x_test = [row["text"] for row in test_rows]
    y_test = [row["label"] for row in test_rows]

    model = build_pipeline()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    probabilities = model.predict_proba(x_test)
    classes = list(model.classes_)

    errors: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for row, actual, predicted, row_probs in zip(test_rows, y_test, y_pred, probabilities):
        prob_by_class = {label: round(float(row_probs[index]), 4) for index, label in enumerate(classes)}
        confidence = max(prob_by_class.values())
        payload = {
            "row_number": row["row_number"],
            "actual": actual,
            "predicted": predicted,
            "confidence": confidence,
            "probabilities": json.dumps(prob_by_class, ensure_ascii=False),
            "problem_text": row["problem_text"],
            "feedback": row["feedback"],
        }
        predictions.append(payload)
        if actual != predicted:
            errors.append(payload)

    labels_sorted = sorted(label_counts)
    metrics = {
        "source_csv": str(csv_path),
        "total_rows_after_cleaning": len(clean_rows),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "label_counts": dict(label_counts),
        "dropped_rows": len(dropped_rows),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "confusion_matrix_labels": labels_sorted,
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels_sorted).tolist(),
        "classification_report": classification_report(y_test, y_pred, labels=labels_sorted, output_dict=True, zero_division=0),
        "warning": "样本量很小，该模型仅作为基线；生产判断仍应优先使用规则和人工复核。",
    }

    joblib.dump(model, out_dir / "reject_reason_model.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(out_dir / "reject_reason_clean.csv", clean_rows, ["row_number", "text", "problem_text", "feedback", "label"])
    write_csv(out_dir / "test_predictions.csv", predictions, ["row_number", "actual", "predicted", "confidence", "probabilities", "problem_text", "feedback"])
    write_csv(out_dir / "test_errors.csv", errors, ["row_number", "actual", "predicted", "confidence", "probabilities", "problem_text", "feedback"])
    write_csv(out_dir / "dropped_rows.csv", dropped_rows, ["row_number", "reason", args.text_col, args.feedback_col, args.label_col, "trainable"])
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "label_meaning": {reason.value: reason.value for reason in ReasonType if reason != ReasonType.UNKNOWN},
                "model_file": "reject_reason_model.joblib",
                "metrics_file": "metrics.json",
                "input_text": f"{args.text_col} + {args.feedback_col}",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="训练不受理原因多分类模型：输入投诉文本和反馈内容，输出第16条原因类型。")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help=f"默认: {DEFAULT_CSV}")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--feedback-col", default="feedback")
    parser.add_argument("--label-col", default="reason_type")
    parser.add_argument("--out-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-chars", type=int, default=5)
    parser.add_argument("--min-samples", type=int, default=20)
    args = parser.parse_args()

    metrics = train(args)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
