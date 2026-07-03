from pathlib import Path

from app.tools.ingest_excel import ingest, normalize_records


def test_normalize_records_reads_platform_excel() -> None:
    records = normalize_records(Path("1780048373993业务信息.xlsx"))

    assert len(records) == 2150
    assert records[0].registration_id
    assert records[0].problem_text
    assert records[0].accept_status in {"已受理", "不受理", "不立案", "已立案", None}


def test_ingest_outputs_runtime_files(tmp_path) -> None:
    summary = ingest(Path("."), tmp_path)

    assert summary["records"] >= 3800
    assert summary["aliases"] > 0
    assert (tmp_path / "complaints.jsonl").exists()
    assert (tmp_path / "address_aliases.json").exists()
    assert (tmp_path / "reply_templates.json").exists()

