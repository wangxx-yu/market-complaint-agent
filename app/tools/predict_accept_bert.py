from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from app.tools.train_accept_bert import DEFAULT_OUT_DIR
from app.tools.train_accept_model import clean_text, decision_from_probability


def predict(model_dir: Path, text: str) -> dict:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not model_dir.exists():
        raise FileNotFoundError(f"找不到 BERT 模型目录: {model_dir}")

    metadata_path = model_dir / "metadata.json"
    runtime_config_path = model_dir / "runtime_config.joblib"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    runtime_config = joblib.load(runtime_config_path) if runtime_config_path.exists() else {}

    thresholds = metadata.get("thresholds", {})
    accept_threshold = float(runtime_config.get("accept_threshold", thresholds.get("accept_threshold", 0.75)))
    reject_threshold = float(runtime_config.get("reject_threshold", thresholds.get("reject_threshold", 0.35)))
    max_length = int(runtime_config.get("max_length", 256))

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    cleaned = clean_text(text)
    inputs = tokenizer(
        [cleaned],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=1).detach().cpu()[0]

    prob_accept = float(probabilities[1])
    raw_label = int(probabilities.argmax().item())
    decision = decision_from_probability(prob_accept, accept_threshold, reject_threshold)
    return {
        "text": cleaned,
        "raw_label": raw_label,
        "raw_label_name": "受理" if raw_label == 1 else "不受理",
        "prob_accept": round(prob_accept, 4),
        "decision": decision,
        "decision_name": {"ACCEPT": "建议受理", "REJECT": "建议不受理", "REVIEW": "建议人工复核"}[decision],
        "thresholds": {
            "accept_threshold": accept_threshold,
            "reject_threshold": reject_threshold,
        },
        "model_dir": str(model_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 BERT 是否受理模型预测一条投诉。")
    parser.add_argument("--model-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--text", required=True, help="具体问题文本")
    args = parser.parse_args()

    result = predict(Path(args.model_dir), args.text)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
