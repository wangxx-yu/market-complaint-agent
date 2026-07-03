from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

from app.core.enums import ReasonType

if TYPE_CHECKING:
    import pandas as pd

PHONE_RE = re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)")
ID_RE = re.compile(r"(?<!\d)(\d{17}[\dXx])(?!\d)")


DEFAULT_INPUT = Path("C:/Users/wangxinwx/Desktop/模型构建训练项目/不受理.xlsx")
DEFAULT_OUTPUT = Path("data/training/reject_reason_training_candidates.csv")


@dataclass(frozen=True)
class RejectRule:
    reason_type: ReasonType
    keywords: tuple[str, ...]
    suggested_department: str = ""
    note: str = ""
    trainable: bool = True


RULES: tuple[RejectRule, ...] = (
    RejectRule(
        reason_type=ReasonType.ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY,
        keywords=("农业农村局", "农药", "农用机械", "农机", "农产品生产", "牛用", "兽用", "兽药", "饲料", "畜牧", "养殖"),
        suggested_department="农业农村局",
        note="农业农村、兽药兽用、农药、农机或农产品生产类事项",
    ),
    RejectRule(
        reason_type=ReasonType.ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY,
        keywords=("物业", "物业费", "物业管理", "自行车棚", "车棚", "公租房", "房管局", "水费", "供热", "供暖", "住建局", "物监办"),
        suggested_department="住建部门",
        note="物业、公租房、供水供暖等事项",
    ),
    RejectRule(
        reason_type=ReasonType.ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY,
        keywords=("派出所", "公安", "报警", "打架", "人身伤害", "故意伤害", "盗窃"),
        suggested_department="公安机关",
        note="治安、人身伤害或公安已处理事项",
    ),
    RejectRule(
        reason_type=ReasonType.ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY,
        keywords=("香烟", "卷烟", "烟草", "烟草专卖"),
        suggested_department="烟草专卖部门",
        note="烟草专卖事项",
    ),
    RejectRule(
        reason_type=ReasonType.ARTICLE16_2_ALREADY_ACCEPTED_OR_PROCESSED,
        keywords=("法院已受理", "仲裁已受理", "消协已处理", "已由法院", "已起诉", "已经起诉", "提起诉讼", "已有派出所处理", "同一事项已"),
        note="同一争议已被其他机关、组织受理或处理",
    ),
    RejectRule(
        reason_type=ReasonType.ARTICLE16_3_NOT_CONSUMER_DISPUTE,
        keywords=("经营性消费", "不属于生活消费", "投资", "加盟", "合伙", "工程款", "货款", "劳动纠纷", "拖欠工资", "借款", "贷款"),
        note="非生活消费权益争议",
    ),
    RejectRule(
        reason_type=ReasonType.ARTICLE16_4_EXPIRED,
        keywords=("超过三年", "三年前", "已过三年", "投诉时效"),
        note="可能超过三年投诉时效",
    ),
    RejectRule(
        reason_type=ReasonType.ARTICLE16_5_MISSING_OR_FALSE_MATERIALS,
        keywords=("主体不明", "地址不详", "商家不详", "不能提供", "无法提供", "未提供", "无凭证", "没有凭证", "证据不足", "材料不全", "虚假材料"),
        note="必要材料缺失、证据不足或材料真实性存疑",
    ),
    RejectRule(
        reason_type=ReasonType.ARTICLE16_6_IMPERSONATION_OR_REFUSE_VERIFY,
        keywords=("冒用他人", "冒名", "身份核验", "拒绝身份核验", "不配合身份核验"),
        note="冒名或拒不配合身份核验",
    ),
)

POST_PROCESS_ONLY_KEYWORDS = (
    "撤诉",
    "已撤诉",
    "和解",
    "自行和解",
    "协商解决",
    "双方已协商",
    "协商处理",
)

CAUTION_KEYWORDS = (
    "无质量问题",
    "市场调节价",
    "自主定价",
    "明码标价",
    "药品是特殊商品",
    "一经售出不能退换",
    "检定合格",
)


def clean_text(value: str | None, max_chars: int = 2000) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    text = PHONE_RE.sub(lambda match: match.group(1)[:3] + "****" + match.group(1)[7:], text)
    text = ID_RE.sub(lambda match: match.group(1)[:6] + "********" + match.group(1)[14:], text)
    return text[:max_chars]


def find_column(columns: list[str], candidates: tuple[str, ...], fallback_index: int) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return columns[fallback_index]


def matched_keywords(text: str, keywords: tuple[str, ...]) -> list[str]:
    return [keyword for keyword in keywords if keyword and keyword in text]


def classify_reject_reason(problem_text: str, feedback: str) -> dict[str, Any]:
    combined = f"{problem_text} {feedback}"
    post_process_hits = matched_keywords(combined, POST_PROCESS_ONLY_KEYWORDS)
    caution_hits = matched_keywords(combined, CAUTION_KEYWORDS)

    for rule in RULES:
        hits = matched_keywords(combined, rule.keywords)
        if not hits:
            continue
        trainable = rule.trainable and not post_process_hits
        note_parts = [rule.note]
        if post_process_hits:
            note_parts.append("包含撤诉/和解/协商等后续处理结果，不建议直接训练为初始不受理原因")
        if caution_hits:
            note_parts.append("包含需谨慎口径，建议人工复核")
        return {
            "reason_type": rule.reason_type.value,
            "suggested_department": rule.suggested_department,
            "matched_keywords": "|".join(hits),
            "trainable": trainable,
            "needs_manual_review": bool(post_process_hits or caution_hits),
            "note": "；".join(part for part in note_parts if part),
        }

    if post_process_hits:
        return {
            "reason_type": "UNKNOWN",
            "suggested_department": "",
            "matched_keywords": "|".join(post_process_hits),
            "trainable": False,
            "needs_manual_review": True,
            "note": "撤诉/和解/协商解决多为后续处理结果，不建议直接训练为初始不受理原因",
        }

    if caution_hits:
        return {
            "reason_type": "UNKNOWN",
            "suggested_department": "",
            "matched_keywords": "|".join(caution_hits),
            "trainable": False,
            "needs_manual_review": True,
            "note": "该类可能是处理结论或需结合事实判断，先人工复核",
        }

    return {
        "reason_type": "UNKNOWN",
        "suggested_department": "",
        "matched_keywords": "",
        "trainable": False,
        "needs_manual_review": True,
        "note": "未命中稳定预标注规则，请人工判断",
    }


def read_reject_excel(path: Path) -> pd.DataFrame:
    import pandas as pd

    df = pd.read_excel(path).fillna("")
    if df.empty:
        raise ValueError(f"Excel 没有可用数据: {path}")
    return df


def build_candidates(input_path: Path, output_path: Path) -> dict[str, Any]:
    df = read_reject_excel(input_path)
    columns = [str(column) for column in df.columns]
    problem_col = find_column(columns, ("具体问题", "problem_text", "text"), 0)
    status_col = find_column(columns, ("初查受理状态", "accept_status", "label"), min(2, len(columns) - 1))
    feedback_col = find_column(columns, ("反馈内容", "feedback"), min(3, len(columns) - 1))

    rows: list[dict[str, Any]] = []
    for row_number, row in df.iterrows():
        problem_text = clean_text(str(row.get(problem_col, "")))
        feedback = clean_text(str(row.get(feedback_col, "")))
        status = clean_text(str(row.get(status_col, "")), max_chars=100)
        label = classify_reject_reason(problem_text, feedback)
        rows.append(
            {
                "source_row": row_number + 2,
                "text": problem_text,
                "feedback": feedback,
                "accept_status": status,
                "is_reject": 1,
                "reason_type": label["reason_type"],
                "suggested_department": label["suggested_department"],
                "matched_keywords": label["matched_keywords"],
                "trainable": int(bool(label["trainable"])),
                "needs_manual_review": int(bool(label["needs_manual_review"])),
                "note": label["note"],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_row",
        "text",
        "feedback",
        "accept_status",
        "is_reject",
        "reason_type",
        "suggested_department",
        "matched_keywords",
        "trainable",
        "needs_manual_review",
        "note",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    reason_counts = Counter(row["reason_type"] for row in rows)
    trainable_counts = Counter(str(row["trainable"]) for row in rows)
    department_counts = Counter(row["suggested_department"] for row in rows if row["suggested_department"])
    summary = {
        "input": str(input_path),
        "output_csv": str(output_path),
        "rows": len(rows),
        "reason_counts": dict(reason_counts),
        "trainable_counts": dict(trainable_counts),
        "department_counts": dict(department_counts),
        "unknown_rows": reason_counts.get("UNKNOWN", 0),
        "manual_review_rows": sum(int(row["needs_manual_review"]) for row in rows),
    }
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="从不受理 Excel 生成第16条不受理原因训练候选表。")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="不受理 Excel 路径")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="输出 CSV 路径")
    args = parser.parse_args()

    summary = build_candidates(Path(args.input), Path(args.output))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
