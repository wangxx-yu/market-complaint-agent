


from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
from app.core.training_config import ACCEPT_MODEL_DIR, ACCEPT_TRAINING_CSV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

PHONE_RE = re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)")
ID_RE = re.compile(r"(?<!\d)(\d{17}[\dXx])(?!\d)")

def clean_text(value: str | None, max_chars: int = 2000) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    text = PHONE_RE.sub(lambda m: m.group(1)[:3] + "****" + m.group(1)[7:], text)
    text = ID_RE.sub(lambda m: m.group(1)[:6] + "********" + m.group(1)[14:], text)
    return text[:max_chars]


def normalize_label(value: str | None) -> int | None:
    label = "" if value is None else str(value).strip()
    if label in {"1", "1.0", "受理", "ACCEPT", "accept", "已受理", "已立案"}:
        return 1
    if label in {"0", "0.0", "不受理", "REJECT", "reject"}:
        return 0
    return None


def load_clean_rows(csv_path: Path, text_col: str, label_col: str, min_chars: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
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
            label = normalize_label(row.get(label_col))
            if not text or len(text) < min_chars:
                dropped_rows.append({"row_number": row_number, "reason": "text_empty_or_too_short", **row})
                continue
            if set(text) <= {"*"}:
                dropped_rows.append({"row_number": row_number, "reason": "text_only_stars", **row})
                continue
            if label is None:
                dropped_rows.append({"row_number": row_number, "reason": "invalid_label", **row})
                continue
            valid_rows.append({"row_number": row_number, "text": text, "label": label})

    by_text: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid_rows:
        by_text[row["text"]].append(row)

    clean_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    for text, rows in by_text.items():
        labels = {row["label"] for row in rows}
        if len(labels) > 1:
            for row in rows:
                conflict_rows.append({**row, "reason": "same_text_conflicting_labels"})
            continue
        clean_rows.append(rows[0])

    return clean_rows, conflict_rows, dropped_rows


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
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )


def decision_from_probability(prob_accept: float, accept_threshold: float, reject_threshold: float) -> str:
    if prob_accept >= accept_threshold:
        return "ACCEPT"
    if prob_accept <= reject_threshold:
        return "REJECT"
    return "REVIEW"


def train(args: argparse.Namespace) -> dict[str, Any]:
    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_rows, conflict_rows, dropped_rows = load_clean_rows(csv_path, args.text_col, args.label_col, args.min_chars)
    if len(clean_rows) < 20:
        raise ValueError("可训练样本少于 20 条，请先检查 CSV 或清洗规则")

    labels = [row["label"] for row in clean_rows]
    label_counts = Counter(labels)
    if len(label_counts) != 2:
        raise ValueError(f"训练需要 0/1 两类标签，当前标签分布: {dict(label_counts)}")

    train_rows, test_rows = train_test_split(
        clean_rows,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=labels,
    )
    x_train = [row["text"] for row in train_rows]
    y_train = [row["label"] for row in train_rows]
    x_test = [row["text"] for row in test_rows]
    y_test = [row["label"] for row in test_rows]

    model = build_pipeline()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, list(model.classes_).index(1)]

    errors: list[dict[str, Any]] = []
    for row, actual, predicted, prob_accept in zip(test_rows, y_test, y_pred, y_prob):
        decision = decision_from_probability(float(prob_accept), args.accept_threshold, args.reject_threshold)
        if int(actual) != int(predicted):
            errors.append(
                {
                    "row_number": row["row_number"],
                    "actual": actual,
                    "predicted": int(predicted),
                    "prob_accept": round(float(prob_accept), 4),
                    "threshold_decision": decision,
                    "text": row["text"],
                }
            )

    metrics = {
        "source_csv": str(csv_path),
        "text_col": args.text_col,
        "label_col": args.label_col,
        "total_rows_after_cleaning": len(clean_rows),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "label_counts": dict(label_counts),
        "conflict_rows": len(conflict_rows),
        "dropped_rows": len(dropped_rows),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y_test, y_pred, labels=[0, 1], target_names=["REJECT", "ACCEPT"], output_dict=True, zero_division=0),
        "thresholds": {
            "accept_threshold": args.accept_threshold,
            "reject_threshold": args.reject_threshold,
        },
    }

    # 保存训练好的模型管道（包含 TF-IDF 和分类器）
    model_path = out_dir / "accept_model.joblib"
    joblib.dump(model, model_path)

    # 保存评估指标报告
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # 保存测试集中的错误预测样本，用于后续分析
    write_csv(out_dir / "test_errors.csv", errors, ["row_number", "actual", "predicted", "prob_accept", "threshold_decision", "text"])

    # 保存清洗后用于训练的有效数据
    write_csv(out_dir / "accept_clean.csv", clean_rows, ["row_number", "text", "label"])

    # 保存因同一文本存在冲突标签而被排除的数据
    write_csv(out_dir / "accept_conflicts.csv", conflict_rows, ["row_number", "reason", "text", "label"])

    # 保存因文本过短、为空或标签无效而被丢弃的数据
    write_csv(out_dir / "accept_dropped.csv", dropped_rows, ["row_number", "reason", args.text_col, args.label_col])

    metadata = {
        "label_meaning": {"1": "受理", "0": "不受理"},
        "model_file": "accept_model.joblib",
        "metrics_file": "metrics.json",
        "thresholds": metrics["thresholds"],
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="训练是否受理二分类模型：输入具体问题 text，输出 1=受理 / 0=不受理。")
    parser.add_argument(
        "--csv",
        default=str(ACCEPT_TRAINING_CSV),
        help=f"清洗后的 CSV 路径，至少包含 text,label 两列。默认: {ACCEPT_TRAINING_CSV}",
    )
    parser.add_argument("--text-col", default="text", help="文本列名，默认 text")
    parser.add_argument("--label-col", default="label", help="标签列名，默认 label")
    parser.add_argument("--out-dir", default=str(ACCEPT_MODEL_DIR), help="模型输出目录")
    parser.add_argument("--test-size", type=float, default=0.2, help="测试集比例，默认 0.2")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-chars", type=int, default=5, help="最短文本长度")
    parser.add_argument("--accept-threshold", type=float, default=0.75, help="受理概率达到该值才自动建议受理")
    parser.add_argument("--reject-threshold", type=float, default=0.35, help="受理概率低于该值才自动建议不受理")
    args = parser.parse_args()

    metrics = train(args)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
