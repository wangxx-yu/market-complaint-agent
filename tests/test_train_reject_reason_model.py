from argparse import Namespace
from pathlib import Path

from app.tools.train_reject_reason_model import train


def test_train_reject_reason_model_smoke(tmp_path: Path) -> None:
    csv_path = tmp_path / "reject_reason.csv"
    rows = [
        ("物业收费问题", "物业管理事项由住建部门处理", "ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY"),
        ("公租房水费争议", "公租房水费由住建部门处理", "ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY"),
        ("经营性采购货款纠纷", "经营性消费不属于生活消费", "ARTICLE16_3_NOT_CONSUMER_DISPUTE"),
        ("投资加盟退费", "投资加盟纠纷不属于生活消费", "ARTICLE16_3_NOT_CONSUMER_DISPUTE"),
        ("商家不详无凭证", "主体不明且无法提供凭证", "ARTICLE16_5_MISSING_OR_FALSE_MATERIALS"),
        ("地址不详材料不全", "投诉材料不全", "ARTICLE16_5_MISSING_OR_FALSE_MATERIALS"),
        ("法院已受理", "同一争议法院已受理", "ARTICLE16_2_ALREADY_ACCEPTED_OR_PROCESSED"),
        ("已经起诉商家", "投诉人已提起诉讼", "ARTICLE16_2_ALREADY_ACCEPTED_OR_PROCESSED"),
    ]
    csv_path.write_text(
        "text,feedback,reason_type,trainable\n"
        + "\n".join(f"{text},{feedback},{label},1" for text, feedback, label in rows),
        encoding="utf-8-sig",
    )

    out_dir = tmp_path / "model"
    metrics = train(
        Namespace(
            csv=str(csv_path),
            text_col="text",
            feedback_col="feedback",
            label_col="reason_type",
            out_dir=str(out_dir),
            test_size=0.5,
            random_state=42,
            min_chars=2,
            min_samples=8,
        )
    )

    assert metrics["total_rows_after_cleaning"] == 8
    assert (out_dir / "reject_reason_model.joblib").exists()
    assert (out_dir / "metrics.json").exists()
