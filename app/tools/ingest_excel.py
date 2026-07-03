from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

from app.core.schemas import HistoricalComplaint
from app.core.text import mask_pii, normalize_text

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
CELL_RE = re.compile(r"([A-Z]+)([0-9]+)")

FIELD_MAP = {
    "登记编号": "registration_id",
    "类型": "complaint_type",
    "接收方式": "channel",
    "企业名称": "enterprise_name",
    "企业地址": "enterprise_address",
    "诉求内容": "appeal_text",
    "问题类别": "problem_category",
    "事发地": "incident_location",
    "具体问题": "problem_text",
    "初查受理状态": "accept_status",
    "处理机构": "handling_org",
    "处理部门": "handling_department",
    "反馈内容": "feedback",
}

OFFICE_CODES = {
    "小坝市场监管所": "QTX_XIAOBA",
    "裕民市场监管所": "QTX_YUMIN",
    "河西市场监管所": "QTX_HEXI",
    "河东市场监管所": "QTX_HEDONG",
    "瞿靖市场监管所": "QTX_QUJING",
    "叶盛市场监管所": "QTX_YESHENG",
    "大坝市场监管所": "QTX_DABA",
    "青铜峡市市场监督管理局": "QTX_BUREAU",
}


def colnum(col: str) -> int:
    number = 0
    for char in col:
        number = number * 26 + ord(char) - 64
    return number


def read_platform_xlsx(path: Path) -> list[dict[int, str]]:
    with ZipFile(path) as archive:
        xml = archive.read("xl/worksheets/sheet1.xml")
    root = ET.fromstring(xml)
    rows: dict[int, dict[int, str]] = {}
    for cell in root.findall(".//m:c", NS):
        match = CELL_RE.match(cell.attrib.get("r", ""))
        if not match:
            continue
        col, row = colnum(match.group(1)), int(match.group(2))
        texts = [node.text or "" for node in cell.findall(".//m:t", NS)]
        value_node = cell.find("m:v", NS)
        value = "".join(texts) if texts else (value_node.text if value_node is not None else "")
        if value:
            rows.setdefault(row, {})[col] = value
    return [rows[row] for row in sorted(rows)]


def normalize_records(path: Path) -> list[HistoricalComplaint]:
    rows = read_platform_xlsx(path)
    if not rows:
        return []
    header = rows[0]
    column_to_field = {column: FIELD_MAP[name] for column, name in header.items() if name in FIELD_MAP}
    records: list[HistoricalComplaint] = []
    for raw in rows[1:]:
        payload: dict[str, str | None] = {}
        for column, field in column_to_field.items():
            value = normalize_text(raw.get(column))
            if field in {"problem_text", "appeal_text", "feedback"}:
                value = mask_pii(value)
            payload[field] = value or None
        if not payload.get("registration_id") or not payload.get("problem_text"):
            continue
        records.append(HistoricalComplaint(**payload))
    return records


def extract_aliases(records: list[HistoricalComplaint]) -> dict[str, dict[str, str | float]]:
    votes: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        office = record.handling_org
        if not office or office not in OFFICE_CODES:
            continue
        for source in [record.incident_location, record.enterprise_address]:
            text = normalize_text(source)
            if not text or text in {"*****", "青铜峡市", "宁夏回族自治区吴忠市青铜峡市"}:
                continue
            if 2 <= len(text) <= 24:
                votes[text][office] += 1

    aliases: dict[str, dict[str, str | float]] = {}
    for alias, counter in votes.items():
        office, count = counter.most_common(1)[0]
        total = sum(counter.values())
        confidence = count / total
        if count >= 2 and confidence >= 0.75:
            aliases[alias] = {
                "office_code": OFFICE_CODES[office],
                "office_name": office,
                "confidence": round(min(0.95, 0.55 + confidence * 0.35), 2),
            }

    aliases.update(
        {
            "铝厂": {"office_code": "QTX_HEXI", "office_name": "河西市场监管所", "confidence": 0.95},
            "小坝镇": {"office_code": "QTX_XIAOBA", "office_name": "小坝市场监管所", "confidence": 0.88},
            "裕民街道": {"office_code": "QTX_YUMIN", "office_name": "裕民市场监管所", "confidence": 0.9},
            "河西": {"office_code": "QTX_HEXI", "office_name": "河西市场监管所", "confidence": 0.88},
            "河东": {"office_code": "QTX_HEDONG", "office_name": "河东市场监管所", "confidence": 0.88},
            "瞿靖": {"office_code": "QTX_QUJING", "office_name": "瞿靖市场监管所", "confidence": 0.88},
            "叶盛": {"office_code": "QTX_YESHENG", "office_name": "叶盛市场监管所", "confidence": 0.88},
            "大坝": {"office_code": "QTX_DABA", "office_name": "大坝市场监管所", "confidence": 0.88},
        }
    )
    return aliases


def extract_reply_templates(records: list[HistoricalComplaint]) -> dict[str, str]:
    feedback_counter = Counter(
        normalize_text(record.feedback, max_chars=240)
        for record in records
        if record.feedback and 8 <= len(record.feedback) <= 240
    )
    common = [text for text, count in feedback_counter.most_common(20) if count >= 2]
    return {
        "accept_default": "您的投诉已登记。建议由{office_name}进一步核查处理，工作人员将结合事实、证据和相关规定依法办理。",
        "reject_out_of_scope": "经初步判断，该事项可能不属于我局职责范围或本机关无处理权限。建议您向相关主管部门反映，具体部门需结合事实和属地进一步确认。",
        "reject_already_processed": "该同一消费争议可能已由人民法院、仲裁机构、其他行政机关、消费者协会等单位受理或处理。建议您通过原受理渠道查询办理进展。",
        "reject_not_consumer": "该事项可能不属于为生活消费需要购买、使用商品或者接受服务产生的消费者权益争议。建议您通过相应主管部门或司法途径反映。",
        "reject_expired": "该事项可能已超过三年投诉时效。建议您补充争议发生时间、交易凭证等材料后由工作人员复核。",
        "reject_missing_or_false_materials": "该投诉可能缺少被投诉人、具体诉求、事实理由、消费凭证等必要材料，或存在材料真实性疑问。请补充真实、完整材料后再提交。",
        "reject_identity_verify": "该投诉可能存在冒用他人名义或不配合身份核验的情形。请由投诉人本人提交并配合身份核验。",
        "reject_other_legal_reasons": "该事项可能属于法律、法规、规章规定的其他不予受理情形，建议由工作人员结合材料进一步复核。",
        "historical_common": "\n".join(common[:5]),
    }


def ingest(data_dir: Path, out_dir: Path) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[HistoricalComplaint] = []
    for path in sorted(data_dir.glob("*.xlsx")):
        records.extend(normalize_records(path))

    complaints_path = out_dir / "complaints.jsonl"
    with complaints_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(record.model_dump_json() + "\n")

    aliases = extract_aliases(records)
    (out_dir / "address_aliases.json").write_text(json.dumps(aliases, ensure_ascii=False, indent=2), encoding="utf-8")
    templates = extract_reply_templates(records)
    (out_dir / "reply_templates.json").write_text(json.dumps(templates, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"records": len(records), "aliases": len(aliases), "templates": len(templates)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("data/runtime"))
    args = parser.parse_args()
    summary = ingest(args.data_dir, args.out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
