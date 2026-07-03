import json
from argparse import Namespace
from pathlib import Path

from app.tools.export_reject_reason_review_data import export


def test_export_reject_reason_review_data_merges_base_and_reviews(tmp_path: Path) -> None:
    base = tmp_path / "base.csv"
    base.write_text(
        "text,feedback,reason_type,suggested_department,trainable,note\n"
        "物业纠纷,住建处理,ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY,住建部门,1,base\n",
        encoding="utf-8-sig",
    )
    traces = tmp_path / "traces.jsonl"
    trace = {
        "trace_id": "t1",
        "classification": {"evidence_fields": ["suggest_department=农业农村局"]},
        "agent_steps": [
            {"name": "preprocess", "output_summary": {"problem_text": "购买牛用食品添加剂后要求退货赔偿。"}},
        ],
        "reply_draft": {"text": "建议您向农业农村局反映。"},
    }
    traces.write_text(json.dumps(trace, ensure_ascii=False) + "\n", encoding="utf-8")
    reviews = tmp_path / "reviews.jsonl"
    reviews.write_text(
        json.dumps(
            {
                "trace_id": "t1",
                "review": {
                    "is_market": False,
                    "reason_type": "ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY",
                    "reject_detail": "兽药及兽用产品由农业农村局监管。",
                    "reply_text": "建议您向农业农村局反映。",
                    "reviewer": "tester",
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = export(
        Namespace(
            base_csv=str(base),
            traces=str(traces),
            reviews=str(reviews),
            out_dir=str(tmp_path / "out"),
            output_name="reject_reason_v2.csv",
            include_base=True,
        )
    )

    assert summary["review_rows_exported"] == 1
    assert summary["merged_rows"] == 2
    assert summary["reason_counts"]["ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY"] == 2
    assert (tmp_path / "out" / "reject_reason_v2.csv").exists()
    exported = (tmp_path / "out" / "reject_reason_review_only.csv").read_text(encoding="utf-8-sig")
    assert "兽药及兽用产品由农业农村局监管。" in exported
