from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from app.core.training_config import ACCEPT_MODEL_DIR
from app.tools.train_accept_model import clean_text, decision_from_probability


def predict(model_dir: Path, text: str) -> dict:
    model_path = model_dir / "accept_model.joblib"
    metadata_path = model_dir / "metadata.json"
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型文件: {model_path}")

    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    thresholds = metadata.get("thresholds", {})
    accept_threshold = float(thresholds.get("accept_threshold", 0.75))
    reject_threshold = float(thresholds.get("reject_threshold", 0.35))

    model = joblib.load(model_path)
    cleaned = clean_text(text)
    prob_accept = float(model.predict_proba([cleaned])[0][list(model.classes_).index(1)])
    raw_label = int(model.predict([cleaned])[0])
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="使用是否受理模型预测一条投诉。")
    parser.add_argument("--model-dir", default=str(ACCEPT_MODEL_DIR))
    parser.add_argument("--text", required=True, help="具体问题文本")
    args = parser.parse_args()

    result = predict(Path(args.model_dir), args.text)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
