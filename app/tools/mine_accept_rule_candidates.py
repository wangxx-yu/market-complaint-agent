from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_REVIEW_CSV = Path("data/evaluation/accept_replay_review.csv")
DEFAULT_OUT_DIR = Path("data/evaluation")

KEYWORDS = [
    "食品",
    "餐馆",
    "饭店",
    "火锅",
    "就餐",
    "变质",
    "过期",
    "异物",
    "虫",
    "退款",
    "退货",
    "退费",
    "赔偿",
    "更换",
    "售后",
    "质量",
    "三包",
    "维修",
    "假冒",
    "虚假宣传",
    "广告",
    "价格",
    "收费",
    "明码标价",
    "多收",
    "会员卡",
    "充值",
    "储值卡",
    "年卡",
    "课程",
    "健身",
    "美容",
    "培训",
    "商家",
    "商户",
    "店铺",
    "超市",
    "药店",
    "酒店",
    "物业",
    "停车费",
    "车位",
    "供暖",
    "燃气",
    "工资",
    "劳动",
    "兽药",
    "饲料",
    "养殖",
    "烟草",
]


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def row_keywords(text: str) -> set[str]:
    return {keyword for keyword in KEYWORDS if keyword in text}


def mine_candidates(
    rows: list[dict[str, Any]],
    min_support: int,
    max_size: int,
) -> list[dict[str, Any]]:
    pattern_counts: dict[tuple[str, ...], Counter[str]] = {}
    for row in rows:
        keywords = sorted(row_keywords(str(row.get("text", ""))))
        label = str(row.get("actual", ""))
        for size in range(1, max_size + 1):
            for pattern in itertools.combinations(keywords, size):
                pattern_counts.setdefault(pattern, Counter())[label] += 1

    candidates: list[dict[str, Any]] = []
    for pattern, counts in pattern_counts.items():
        total = sum(counts.values())
        if total < min_support:
            continue
        accept_count = int(counts.get("1", 0))
        reject_count = int(counts.get("0", 0))
        accept_rate = accept_count / total if total else 0
        reject_rate = reject_count / total if total else 0
        candidates.append(
            {
                "pattern": " + ".join(pattern),
                "keyword_count": len(pattern),
                "total": total,
                "accept_count": accept_count,
                "reject_count": reject_count,
                "accept_rate": round(accept_rate, 4),
                "reject_rate": round(reject_rate, 4),
            }
        )
    candidates.sort(key=lambda item: (item["accept_rate"], item["total"], item["keyword_count"]), reverse=True)
    return candidates


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    rows = read_rows(csv_path)
    candidates = mine_candidates(rows, args.min_support, args.max_size)
    high_accept = [
        row
        for row in candidates
        if row["accept_rate"] >= args.accept_rate and row["accept_count"] >= args.min_accept_count
    ]
    high_reject = sorted(
        [
            row
            for row in candidates
            if row["reject_rate"] >= args.reject_rate and row["reject_count"] >= args.min_reject_count
        ],
        key=lambda item: (item["reject_rate"], item["total"], item["keyword_count"]),
        reverse=True,
    )

    write_csv(
        out_dir / "accept_rule_candidates_all.csv",
        candidates,
        ["pattern", "keyword_count", "total", "accept_count", "reject_count", "accept_rate", "reject_rate"],
    )
    write_csv(
        out_dir / "accept_rule_candidates_high_accept.csv",
        high_accept,
        ["pattern", "keyword_count", "total", "accept_count", "reject_count", "accept_rate", "reject_rate"],
    )
    write_csv(
        out_dir / "accept_rule_candidates_high_reject.csv",
        high_reject,
        ["pattern", "keyword_count", "total", "accept_count", "reject_count", "accept_rate", "reject_rate"],
    )
    summary = {
        "source_csv": str(csv_path),
        "rows": len(rows),
        "keywords": len(KEYWORDS),
        "candidate_count": len(candidates),
        "high_accept_count": len(high_accept),
        "high_reject_count": len(high_reject),
        "top_high_accept": high_accept[:20],
        "top_high_reject": high_reject[:20],
        "outputs": {
            "all_csv": (out_dir / "accept_rule_candidates_all.csv").as_posix(),
            "high_accept_csv": (out_dir / "accept_rule_candidates_high_accept.csv").as_posix(),
            "high_reject_csv": (out_dir / "accept_rule_candidates_high_reject.csv").as_posix(),
        },
    }
    (out_dir / "accept_rule_candidates_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="从人工复核样本中挖掘高受理率/高不受理率关键词组合候选。")
    parser.add_argument("--csv", default=str(DEFAULT_REVIEW_CSV))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--min-support", type=int, default=8)
    parser.add_argument("--max-size", type=int, default=3)
    parser.add_argument("--accept-rate", type=float, default=0.9)
    parser.add_argument("--reject-rate", type=float, default=0.75)
    parser.add_argument("--min-accept-count", type=int, default=6)
    parser.add_argument("--min-reject-count", type=int, default=6)
    args = parser.parse_args()
    print(json.dumps(analyze(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
