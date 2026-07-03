from argparse import Namespace
from pathlib import Path

from app.tools.predict_accept_model import predict
from app.tools.train_accept_model import load_clean_rows, train


def test_load_clean_rows_removes_conflicts(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "text,label\n"
        "购买食品过期要求退款,1\n"
        "购买食品过期要求退款,1\n"
        "物业费纠纷要求处理,0\n"
        "同一句冲突,1\n"
        "同一句冲突,0\n",
        encoding="utf-8-sig",
    )

    clean_rows, conflict_rows, dropped_rows = load_clean_rows(csv_path, "text", "label", min_chars=2)

    assert len(clean_rows) == 2
    assert len(conflict_rows) == 2
    assert dropped_rows == []


def test_train_and_predict_accept_model(tmp_path: Path) -> None:
    csv_path = tmp_path / "train.csv"
    rows = ["text,label"]
    for i in range(30):
        rows.append(f"在超市购买食品过期要求退款赔偿{i},1")
        rows.append(f"小区物业费停车位产权纠纷要求处理{i},0")
    csv_path.write_text("\n".join(rows), encoding="utf-8-sig")

    out_dir = tmp_path / "model"
    metrics = train(
        Namespace(
            csv=str(csv_path),
            text_col="text",
            label_col="label",
            out_dir=str(out_dir),
            test_size=0.25,
            random_state=42,
            min_chars=2,
            accept_threshold=0.75,
            reject_threshold=0.35,
        )
    )

    assert metrics["total_rows_after_cleaning"] == 60
    assert (out_dir / "accept_model.joblib").exists()
    result = predict(out_dir, "在超市买到过期食品，要求退款")
    assert result["prob_accept"] > 0.5

