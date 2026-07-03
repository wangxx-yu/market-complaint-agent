from argparse import Namespace
from pathlib import Path

from app.tools.analyze_accept_review_samples import analyze


def test_analyze_accept_review_samples_groups_categories(tmp_path: Path) -> None:
    csv_path = tmp_path / "review.csv"
    csv_path.write_text(
        "row_number,actual,predicted,decision,is_market,reason_type,confidence,decision_source,review_reasons,evidence_fields,text\n"
        "2,1,,REVIEW,True,UNKNOWN,0.6,MODEL,模型受理概率处于人工复核区间,,饭店食品变质要求退款\n"
        "3,0,,REVIEW,True,UNKNOWN,0.5,MODEL,模型受理概率处于人工复核区间,,物业费停车位收费纠纷\n",
        encoding="utf-8-sig",
    )
    out_dir = tmp_path / "out"

    summary = analyze(
        Namespace(
            csv=str(csv_path),
            out_dir=str(out_dir),
            samples_per_category=10,
        )
    )

    assert summary["total_review_rows"] == 2
    assert summary["matched_rows"] == 2
    categories = {row["category"]: row for row in summary["category_summary"]}
    assert categories["食品餐饮"]["accept_count"] == 1
    assert categories["物业住建"]["reject_count"] == 1
    assert (out_dir / "accept_review_category_summary.csv").exists()
