from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_REVIEW_CSV = Path("data/evaluation/accept_replay_review.csv")
DEFAULT_OUT_DIR = Path("data/evaluation")

CATEGORY_PATTERNS: dict[str, list[str]] = {
    "食品餐饮": ["食品", "餐馆", "饭店", "火锅", "就餐", "变质", "过期", "虫", "外卖", "蛋糕", "饮料"],
    "退款退货": ["退款", "退货", "退费", "退赔", "赔偿", "更换", "售后"],
    "充值会员卡": ["充值", "会员卡", "年卡", "储值卡", "预付卡", "退卡"],
    "价格收费": ["价格", "收费", "明码标价", "多收", "涨价", "停车费", "物业费"],
    "质量三包": ["质量", "三包", "维修", "假冒", "合格产品", "瑕疵", "损坏"],
    "虚假宣传广告": ["虚假宣传", "广告", "诱导", "宣传"],
    "物业住建": ["物业", "车位", "停车位", "供暖", "供热", "燃气", "天然气", "公租房", "住建"],
    "农业畜牧": ["兽药", "兽用", "饲料", "牛用", "养殖", "畜牧", "农药", "农机"],
    "劳动工资": ["工资", "劳动", "上班", "兼职", "工钱"],
    "公安治安": ["公安", "派出所", "报警", "打架", "盗窃", "人身伤害"],
    "烟草": ["香烟", "卷烟", "烟草"],
}


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def matched_categories(text: str) -> list[str]:
    return [
        category
        for category, keywords in CATEGORY_PATTERNS.items()
        if any(keyword in text for keyword in keywords)
    ]


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(csv_path)

    category_rows: dict[str, list[dict[str, Any]]] = {category: [] for category in CATEGORY_PATTERNS}
    unmatched_rows: list[dict[str, Any]] = []
    for row in rows:
        categories = matched_categories(str(row.get("text", "")))
        if not categories:
            unmatched_rows.append(row)
            continue
        for category in categories:
            category_rows[category].append(row)

    summary_rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for category, matched in category_rows.items():
        if not matched:
            continue
        label_counts = Counter(str(row.get("actual", "")) for row in matched)
        total = len(matched)
        accept_count = int(label_counts.get("1", 0))
        reject_count = int(label_counts.get("0", 0))
        model_review_count = sum(1 for row in matched if "模型受理概率" in str(row.get("review_reasons", "")))
        rule_review_count = total - model_review_count
        summary_rows.append(
            {
                "category": category,
                "total": total,
                "accept_count": accept_count,
                "reject_count": reject_count,
                "accept_rate": round(accept_count / total, 4),
                "reject_rate": round(reject_count / total, 4),
                "model_review_count": model_review_count,
                "rule_review_count": rule_review_count,
            }
        )
        for row in matched[: args.samples_per_category]:
            samples.append({"category": category, **row})

    summary_rows.sort(key=lambda row: (row["total"], row["accept_rate"]), reverse=True)
    summary = {
        "source_csv": str(csv_path),
        "total_review_rows": len(rows),
        "matched_rows": len(rows) - len(unmatched_rows),
        "unmatched_rows": len(unmatched_rows),
        "category_summary": summary_rows,
        "outputs": {
            "summary_json": (out_dir / "accept_review_category_summary.json").as_posix(),
            "summary_csv": (out_dir / "accept_review_category_summary.csv").as_posix(),
            "samples_csv": (out_dir / "accept_review_category_samples.csv").as_posix(),
            "unmatched_csv": (out_dir / "accept_review_category_unmatched.csv").as_posix(),
        },
    }

    (out_dir / "accept_review_category_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(
        out_dir / "accept_review_category_summary.csv",
        summary_rows,
        ["category", "total", "accept_count", "reject_count", "accept_rate", "reject_rate", "model_review_count", "rule_review_count"],
    )
    if rows:
        sample_fields = ["category", *rows[0].keys()]
        write_csv(out_dir / "accept_review_category_samples.csv", samples, sample_fields)
        write_csv(out_dir / "accept_review_category_unmatched.csv", unmatched_rows, list(rows[0].keys()))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="分析批量回放中进入人工复核的样本，按业务关键词统计可优化方向。")
    parser.add_argument("--csv", default=str(DEFAULT_REVIEW_CSV))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--samples-per-category", type=int, default=20)
    args = parser.parse_args()

    print(json.dumps(analyze(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
