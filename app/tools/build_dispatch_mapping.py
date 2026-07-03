from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

VALID_OFFICES = {
    "小坝市场监管所": "QTX_XIAOBA",
    "裕民市场监管所": "QTX_YUMIN",
    "河西市场监管所": "QTX_HEXI",
    "河东市场监管所": "QTX_HEDONG",
    "瞿靖市场监管所": "QTX_QUJING",
    "叶盛市场监管所": "QTX_YESHENG",
    "大坝市场监管所": "QTX_DABA",
}

BUREAU_OFFICES = {"青铜峡市市场监督管理局"}

BUILT_IN_RULES = {
    "铝厂": ("QTX_HEXI", "河西市场监管所", 0.95),
    "河西": ("QTX_HEXI", "河西市场监管所", 0.9),
    "河东": ("QTX_HEDONG", "河东市场监管所", 0.9),
    "小坝镇": ("QTX_XIAOBA", "小坝市场监管所", 0.9),
    "小坝": ("QTX_XIAOBA", "小坝市场监管所", 0.84),
    "裕民": ("QTX_YUMIN", "裕民市场监管所", 0.9),
    "瞿靖": ("QTX_QUJING", "瞿靖市场监管所", 0.9),
    "叶盛": ("QTX_YESHENG", "叶盛市场监管所", 0.9),
    "大坝": ("QTX_DABA", "大坝市场监管所", 0.9),
}

NOISE_VALUES = {
    "",
    "*****",
    "青铜峡市",
    "宁夏青铜峡市",
    "宁夏回族自治区吴忠市青铜峡市",
    "宁夏回族自治区青铜峡市",
    "吴忠市青铜峡市",
}


def normalize(value: str | None) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", "", text)
    text = text.replace("宁夏回族自治区", "").replace("吴忠市", "")
    return text.strip("，。,.;；:：、 ")


def candidate_aliases(record: dict[str, Any]) -> set[str]:
    aliases: set[str] = set()
    for key in ["incident_location", "enterprise_address"]:
        text = normalize(record.get(key))
        if text and text not in NOISE_VALUES:
            aliases.add(text)
            if text.startswith("青铜峡市") and len(text) > 4:
                aliases.add(text.removeprefix("青铜峡市"))
    enterprise_name = normalize(record.get("enterprise_name"))
    if enterprise_name and 2 <= len(enterprise_name) <= 18:
        aliases.add(enterprise_name)
    problem_text = str(record.get("problem_text") or "")
    for match in re.finditer(r"(?:商户详细位置|商户地址|详细位置|地址)[:：]([^事件问题主要诉求，。,；;]{2,30})", problem_text):
        aliases.add(normalize(match.group(1)))
    return {alias for alias in aliases if 2 <= len(alias) <= 30 and alias not in NOISE_VALUES}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_mapping(args: argparse.Namespace) -> dict[str, Any]:
    records = read_jsonl(Path(args.complaints))
    votes: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[tuple[str, str], str] = {}
    skipped_bureau = 0

    for record in records:
        office = record.get("handling_org") or ""
        if office in BUREAU_OFFICES and not args.include_bureau:
            skipped_bureau += 1
            continue
        if office not in VALID_OFFICES:
            continue
        for alias in candidate_aliases(record):
            votes[alias][office] += 1
            examples.setdefault((alias, office), record.get("registration_id", ""))

    accepted_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    for alias, counter in votes.items():
        total = sum(counter.values())
        office, count = counter.most_common(1)[0]
        confidence = count / total if total else 0
        row = {
            "alias": alias,
            "office_code": VALID_OFFICES[office],
            "office_name": office,
            "confidence": round(confidence, 4),
            "support": count,
            "total": total,
            "all_votes": json.dumps(dict(counter), ensure_ascii=False),
            "example_registration_id": examples.get((alias, office), ""),
        }
        if count >= args.min_support and confidence >= args.min_confidence:
            accepted_rows.append(row)
        else:
            conflict_rows.append(row)

    for alias, (code, office, confidence) in BUILT_IN_RULES.items():
        accepted_rows.append(
            {
                "alias": alias,
                "office_code": code,
                "office_name": office,
                "confidence": confidence,
                "support": "manual",
                "total": "manual",
                "all_votes": "{}",
                "example_registration_id": "manual_rule",
            }
        )

    accepted_rows.sort(key=lambda row: (str(row["support"]) != "manual", -float(row["confidence"]), str(row["alias"])))
    conflict_rows.sort(key=lambda row: (-int(row["total"]), str(row["alias"])))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_fields = ["alias", "office_code", "office_name", "confidence", "support", "total", "all_votes", "example_registration_id"]
    with (out_dir / "dispatch_mapping.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(accepted_rows)
    with (out_dir / "dispatch_mapping_conflicts.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(conflict_rows)

    mapping_json = {
        row["alias"]: {
            "office_code": row["office_code"],
            "office_name": row["office_name"],
            "confidence": float(row["confidence"]),
            "support": row["support"],
        }
        for row in accepted_rows
    }
    (out_dir / "dispatch_mapping.json").write_text(json.dumps(mapping_json, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "records": len(records),
        "accepted_aliases": len(accepted_rows),
        "conflict_aliases": len(conflict_rows),
        "skipped_bureau_records": skipped_bureau,
        "min_support": args.min_support,
        "min_confidence": args.min_confidence,
    }
    (out_dir / "dispatch_mapping_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="从历史投诉数据中筛选地址到市场监管所的映射表。")
    parser.add_argument("--complaints", default="data/runtime/complaints.jsonl")
    parser.add_argument("--out-dir", default="data/dispatch")
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--min-confidence", type=float, default=0.8)
    parser.add_argument("--include-bureau", action="store_true", help="包含局本级历史记录，默认排除以减少噪声")
    args = parser.parse_args()

    print(json.dumps(build_mapping(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

