import json
from argparse import Namespace
from pathlib import Path

from app.tools.export_review_training_data import export


def test_export_review_training_data_merges_base_and_reviews(tmp_path: Path) -> None:
    base = tmp_path / "base.csv"
    base.write_text("text,label\n食品过期,1\n物业纠纷,0\n", encoding="utf-8-sig")
    traces = tmp_path / "traces.jsonl"
    trace = {
        "trace_id": "t1",
        "agent_steps": [
            {"name": "preprocess", "output_summary": {"problem_text": "新投诉内容"}},
        ],
    }
    traces.write_text(json.dumps(trace, ensure_ascii=False) + "\n", encoding="utf-8")
    reviews = tmp_path / "reviews.jsonl"
    reviews.write_text(
        json.dumps({"trace_id": "t1", "review": {"is_market": True}}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = export(
        Namespace(
            base_csv=str(base),
            traces=str(traces),
            reviews=str(reviews),
            out_dir=str(tmp_path / "out"),
            output_name="v2.csv",
            include_base=True,
        )
    )

    assert summary["review_rows_exported"] == 1
    assert summary["merged_rows"] == 3
    assert (tmp_path / "out" / "v2.csv").exists()

