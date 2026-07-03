from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_REVIEW_CSV = Path("data/evaluation/accept_replay_review.csv")
DEFAULT_CANDIDATES_CSV = Path("data/evaluation/accept_rule_candidates_high_accept.csv")
DEFAULT_OUT_DIR = Path("data/evaluation")


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pattern_keywords(pattern: str) -> list[str]:
    return [part.strip() for part in pattern.split("+") if part.strip()]


def matches_pattern(text: str, pattern: str) -> bool:
    return all(keyword in text for keyword in pattern_keywords(pattern))


def simulate(args: argparse.Namespace) -> dict[str, Any]:
    review_rows = read_csv(Path(args.review_csv))
    candidates = read_csv(Path(args.candidates_csv))
    eligible_candidates = [
        candidate
        for candidate in candidates
        if int(candidate.get("total", 0) or 0) >= args.min_support
        and float(candidate.get("accept_rate", 0) or 0) >= args.min_accept_rate
    ]

    candidate_results: list[dict[str, Any]] = []
    for candidate in eligible_candidates:
        pattern = str(candidate["pattern"])
        matched = [row for row in review_rows if matches_pattern(str(row.get("text", "")), pattern)]
        accept_hits = sum(1 for row in matched if str(row.get("actual")) == "1")
        reject_hits = sum(1 for row in matched if str(row.get("actual")) == "0")
        total_hits = len(matched)
        candidate_results.append(
            {
                "pattern": pattern,
                "hit_rows": total_hits,
                "true_accept_hits": accept_hits,
                "false_accept_hits": reject_hits,
                "precision_if_auto_accept": round(accept_hits / total_hits, 4) if total_hits else 0,
                "review_reduction_if_alone": total_hits,
                "source_total": int(candidate.get("total", 0) or 0),
                "source_accept_rate": float(candidate.get("accept_rate", 0) or 0),
            }
        )

    auto_rows_by_key: dict[str, dict[str, Any]] = {}
    matched_patterns_by_key: dict[str, list[str]] = {}
    for result in candidate_results:
        pattern = result["pattern"]
        for row in review_rows:
            if not matches_pattern(str(row.get("text", "")), pattern):
                continue
            key = str(row.get("row_number") or row.get("text"))
            auto_rows_by_key[key] = row
            matched_patterns_by_key.setdefault(key, []).append(pattern)

    auto_rows: list[dict[str, Any]] = []
    false_accept_rows: list[dict[str, Any]] = []
    for key, row in auto_rows_by_key.items():
        item = {**row, "matched_patterns": "；".join(matched_patterns_by_key.get(key, []))}
        auto_rows.append(item)
        if str(row.get("actual")) == "0":
            false_accept_rows.append(item)

    review_rows_count = len(review_rows)
    review_reduction = len(auto_rows)
    false_accept_count = len(false_accept_rows)
    true_accept_count = review_reduction - false_accept_count
    summary = {
        "review_rows": review_rows_count,
        "candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible_candidates),
        "auto_accept_rows": review_reduction,
        "true_accept_rows": true_accept_count,
        "false_accept_rows": false_accept_count,
        "new_review_rows": review_rows_count - review_reduction,
        "review_reduction_rate": round(review_reduction / review_rows_count, 4) if review_rows_count else 0,
        "precision_if_auto_accept_all": round(true_accept_count / review_reduction, 4) if review_reduction else 0,
        "candidate_results": sorted(
            candidate_results,
            key=lambda row: (row["precision_if_auto_accept"], row["hit_rows"]),
            reverse=True,
        ),
        "outputs": {
            "summary_json": (Path(args.out_dir) / "accept_rule_simulation_summary.json").as_posix(),
            "auto_accept_csv": (Path(args.out_dir) / "accept_rule_simulation_auto_accept.csv").as_posix(),
            "false_accept_csv": (Path(args.out_dir) / "accept_rule_simulation_false_accept.csv").as_posix(),
        },
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "accept_rule_simulation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if review_rows:
        fieldnames = [*review_rows[0].keys(), "matched_patterns"]
        write_csv(out_dir / "accept_rule_simulation_auto_accept.csv", auto_rows, fieldnames)
        write_csv(out_dir / "accept_rule_simulation_false_accept.csv", false_accept_rows, fieldnames)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="模拟高受理率候选规则上线后的复核减少量和错判风险。")
    parser.add_argument("--review-csv", default=str(DEFAULT_REVIEW_CSV))
    parser.add_argument("--candidates-csv", default=str(DEFAULT_CANDIDATES_CSV))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--min-support", type=int, default=8)
    parser.add_argument("--min-accept-rate", type=float, default=0.9)
    args = parser.parse_args()
    print(json.dumps(simulate(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
