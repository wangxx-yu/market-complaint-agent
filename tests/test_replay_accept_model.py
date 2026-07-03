from argparse import Namespace
from pathlib import Path

from app.tools.replay_accept_model import replay


def test_replay_accept_model_writes_summary_and_csvs(tmp_path: Path) -> None:
    csv_path = tmp_path / "train.csv"
    rows = ["text,label"]
    for index in range(12):
        rows.append(f"在超市购买食品过期要求退款{index},1")
        rows.append(f"物业费和停车位产权纠纷要求处理{index},0")
    csv_path.write_text("\n".join(rows), encoding="utf-8-sig")

    out_dir = tmp_path / "evaluation"
    summary = replay(
        Namespace(
            csv=str(csv_path),
            text_col="text",
            label_col="label",
            out_dir=str(out_dir),
            min_chars=2,
            limit=0,
        )
    )

    assert summary["evaluated_rows"] == 24
    assert "decision_counts" in summary
    assert (out_dir / "accept_replay_summary.json").exists()
    assert (out_dir / "accept_replay_predictions.csv").exists()
    assert (out_dir / "accept_replay_errors.csv").exists()
    assert (out_dir / "accept_replay_review.csv").exists()
