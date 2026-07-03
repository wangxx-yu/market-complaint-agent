import json
from argparse import Namespace
from pathlib import Path

from app.tools.build_dispatch_mapping import build_mapping


def test_build_dispatch_mapping_from_history(tmp_path: Path) -> None:
    complaints = tmp_path / "complaints.jsonl"
    rows = [
        {"registration_id": "1", "enterprise_address": "青铜峡市铝厂", "incident_location": "", "enterprise_name": "A", "handling_org": "河西市场监管所"},
        {"registration_id": "2", "enterprise_address": "青铜峡市铝厂", "incident_location": "", "enterprise_name": "B", "handling_org": "河西市场监管所"},
        {"registration_id": "3", "enterprise_address": "青铜峡市小坝镇", "incident_location": "", "enterprise_name": "C", "handling_org": "小坝市场监管所"},
    ]
    complaints.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")

    summary = build_mapping(
        Namespace(
            complaints=str(complaints),
            out_dir=str(tmp_path / "out"),
            min_support=2,
            min_confidence=0.8,
            include_bureau=False,
        )
    )

    assert summary["accepted_aliases"] >= 1
    mapping = json.loads((tmp_path / "out" / "dispatch_mapping.json").read_text(encoding="utf-8"))
    assert mapping["铝厂"]["office_name"] == "河西市场监管所"

