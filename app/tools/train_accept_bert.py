from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from app.core.training_config import ACCEPT_TRAINING_CSV
from app.tools.train_accept_model import clean_text, decision_from_probability, load_clean_rows


DEFAULT_BERT_MODEL = "bert-base-chinese"
DEFAULT_OUT_DIR = Path("models/accept_bert_v1")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_train_test_rows(
    clean_rows: list[dict[str, Any]],
    test_size: float,
    random_state: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels = [row["label"] for row in clean_rows]
    return train_test_split(
        clean_rows,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )


def evaluate_predictions(
    y_true: list[int],
    y_pred: list[int],
    prob_accept: list[float],
    accept_threshold: float,
    reject_threshold: float,
) -> dict[str, Any]:
    decisions = [
        decision_from_probability(probability, accept_threshold, reject_threshold)
        for probability in prob_accept
    ]
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["REJECT", "ACCEPT"],
            output_dict=True,
            zero_division=0,
        ),
        "threshold_decisions": dict(Counter(decisions)),
    }


def compute_class_weights(labels: list[int]) -> list[float]:
    label_counts = Counter(labels)
    total = len(labels)
    class_count = 2
    return [
        total / (class_count * label_counts.get(0, 1)),
        total / (class_count * label_counts.get(1, 1)),
    ]


class AcceptBertDataset:
    def __init__(self, texts: list[str], labels: list[int], tokenizer: Any, max_length: int) -> None:
        self.labels = labels
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(args.random_state)
    torch.manual_seed(args.random_state)

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] 读取并清洗训练数据: {csv_path}", flush=True)
    clean_rows, conflict_rows, dropped_rows = load_clean_rows(
        csv_path,
        args.text_col,
        args.label_col,
        args.min_chars,
    )
    if len(clean_rows) < 20:
        raise ValueError("可训练样本少于 20 条，请先检查 CSV 或清洗规则")

    label_counts = Counter(row["label"] for row in clean_rows)
    if len(label_counts) != 2:
        raise ValueError(f"BERT 训练需要 0/1 两类标签，当前标签分布: {dict(label_counts)}")

    print(f"[2/6] 有效样本 {len(clean_rows)} 条，标签分布: {dict(label_counts)}", flush=True)
    train_rows, test_rows = build_train_test_rows(clean_rows, args.test_size, args.random_state)
    x_train = [row["text"] for row in train_rows]
    y_train = [int(row["label"]) for row in train_rows]
    x_test = [row["text"] for row in test_rows]
    y_test = [int(row["label"]) for row in test_rows]

    print(f"[3/6] 加载 BERT 模型和分词器: {args.base_model}", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=args.local_files_only)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model,
            num_labels=2,
            local_files_only=args.local_files_only,
        )
    except OSError as exc:
        endpoint_tip = (
            "也可以加参数 --hf-endpoint https://hf-mirror.com 使用镜像下载。"
            if not args.hf_endpoint
            else f"当前镜像地址为 {args.hf_endpoint}，仍未下载成功，请检查网络或改用本地模型目录。"
        )
        raise RuntimeError(
            "BERT 基座模型加载失败。常见原因是无法连接 Hugging Face，且本机缓存里没有模型文件。\n"
            "处理办法：\n"
            "1. 网络可用时重新运行训练命令，等待 bert-base-chinese 下载完成；\n"
            f"2. {endpoint_tip}\n"
            "3. 如果已手动下载模型，把 --base-model 改成本地模型文件夹路径。"
        ) from exc

    print("[4/6] 构造训练集和测试集", flush=True)
    train_dataset = AcceptBertDataset(x_train, y_train, tokenizer, args.max_length)
    test_dataset = AcceptBertDataset(x_test, y_test, tokenizer, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device)

    class_weights = compute_class_weights(y_train) if args.class_weight == "balanced" else [1.0, 1.0]
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    print(
        f"[5/6] 开始训练，device={device}, epochs={args.epochs}, batch_size={args.batch_size}, "
        f"class_weight={args.class_weight}, weights={class_weights}",
        flush=True,
    )
    model.train()
    train_losses: list[float] = []
    for epoch in range(args.epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.pop("labels")
            output = model(**batch)
            loss = loss_fn(output.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            train_losses.append(float(loss.detach().cpu()))
        print(f"  epoch {epoch + 1}/{args.epochs} 完成，loss={train_losses[-1]:.6f}", flush=True)

    print("[6/6] 评估并保存模型", flush=True)
    model.eval()
    y_pred: list[int] = []
    prob_accept: list[float] = []
    with torch.no_grad():
        for batch in test_loader:
            labels = batch.pop("labels")
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            probabilities = torch.softmax(logits, dim=1).detach().cpu()
            y_pred.extend(probabilities.argmax(dim=1).tolist())
            prob_accept.extend(probabilities[:, 1].tolist())
            batch["labels"] = labels

    metrics = {
        "source_csv": str(csv_path),
        "text_col": args.text_col,
        "label_col": args.label_col,
        "base_model": args.base_model,
        "device": str(device),
        "total_rows_after_cleaning": len(clean_rows),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "label_counts": dict(label_counts),
        "conflict_rows": len(conflict_rows),
        "dropped_rows": len(dropped_rows),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "class_weight": args.class_weight,
        "class_weights": {"0": round(class_weights[0], 6), "1": round(class_weights[1], 6)},
        "train_loss_last": round(train_losses[-1], 6) if train_losses else None,
        "thresholds": {
            "accept_threshold": args.accept_threshold,
            "reject_threshold": args.reject_threshold,
        },
        **evaluate_predictions(y_test, y_pred, prob_accept, args.accept_threshold, args.reject_threshold),
    }

    predictions: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for row, actual, predicted, probability in zip(test_rows, y_test, y_pred, prob_accept):
        item = {
            "row_number": row["row_number"],
            "actual": actual,
            "predicted": int(predicted),
            "prob_accept": round(float(probability), 4),
            "threshold_decision": decision_from_probability(
                float(probability),
                args.accept_threshold,
                args.reject_threshold,
            ),
            "text": row["text"],
        }
        predictions.append(item)
        if int(actual) != int(predicted):
            errors.append(item)

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "label_meaning": {"1": "受理", "0": "不受理"},
                "model_type": "bert_sequence_classification",
                "base_model": args.base_model,
                "class_weight": args.class_weight,
                "class_weights": {"0": round(class_weights[0], 6), "1": round(class_weights[1], 6)},
                "thresholds": metrics["thresholds"],
                "metrics_file": "metrics.json",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_csv(out_dir / "accept_clean.csv", clean_rows, ["row_number", "text", "label"])
    write_csv(out_dir / "accept_conflicts.csv", conflict_rows, ["row_number", "reason", "text", "label"])
    write_csv(out_dir / "accept_dropped.csv", dropped_rows, ["row_number", "reason", args.text_col, args.label_col])
    write_csv(out_dir / "test_predictions.csv", predictions, ["row_number", "actual", "predicted", "prob_accept", "threshold_decision", "text"])
    write_csv(out_dir / "test_errors.csv", errors, ["row_number", "actual", "predicted", "prob_accept", "threshold_decision", "text"])

    joblib.dump(
        {
            "accept_threshold": args.accept_threshold,
            "reject_threshold": args.reject_threshold,
            "max_length": args.max_length,
            "base_model": args.base_model,
            "class_weight": args.class_weight,
            "class_weights": {"0": round(class_weights[0], 6), "1": round(class_weights[1], 6)},
        },
        out_dir / "runtime_config.joblib",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 BERT 是否受理二分类模型：输入具体问题 text，输出 1=受理 / 0=不受理。")
    parser.add_argument("--csv", default=str(ACCEPT_TRAINING_CSV), help=f"训练 CSV 路径，默认: {ACCEPT_TRAINING_CSV}")
    parser.add_argument("--text-col", default="text", help="文本列名，默认 text")
    parser.add_argument("--label-col", default="label", help="标签列名，默认 label")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="模型输出目录")
    parser.add_argument("--base-model", default=DEFAULT_BERT_MODEL, help="HuggingFace 模型名或本地模型目录")
    parser.add_argument("--hf-endpoint", default=None, help="可选 Hugging Face 镜像地址，例如 https://hf-mirror.com")
    parser.add_argument("--local-files-only", action="store_true", help="只从本机缓存或本地模型目录加载，不联网下载")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-chars", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default="balanced",
        help="balanced 会提高少数类训练权重；none 表示不加权。默认 balanced",
    )
    parser.add_argument("--accept-threshold", type=float, default=0.75)
    parser.add_argument("--reject-threshold", type=float, default=0.35)
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU 训练")
    args = parser.parse_args()

    metrics = train(args)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
