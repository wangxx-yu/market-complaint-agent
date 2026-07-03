import json
from argparse import Namespace
from pathlib import Path

from app.tools.export_dispatch_review_data import export


def test_export_dispatch_review_data_marks_changed_rows(tmp_path: Path) -> None:
    traces = tmp_path / "traces.jsonl"
    trace = {
        "trace_id": "t1",
        "dispatch": {
            "office_code": "QTX_XIAOBA",
            "office_name": "小坝市场监管所",
            "matched_rule": "default_office",
            "confidence": 0.35,
            "needs_review": True,
        },
        "agent_steps": [
            {"name": "preprocess", "output_summary": {"problem_text": "投诉内容"}},
            {"name": "dispatch", "input_summary": {"address": "某地址"}},
        ],
    }
    traces.write_text(json.dumps(trace, ensure_ascii=False) + "\n", encoding="utf-8")
    reviews = tmp_path / "reviews.jsonl"
    reviews.write_text(
        json.dumps(
            {
                "trace_id": "t1",
                "review": {
                    "office_code": "QTX_HEXI",
                    "office_name": "河西市场监管所",
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = export(Namespace(traces=str(traces), reviews=str(reviews), out_dir=str(tmp_path / "out")))

    assert summary["rows_exported"] == 1
    assert summary["changed_rows"] == 1
    assert (tmp_path / "out" / "dispatch_review_changes.csv").exists()

