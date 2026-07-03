from argparse import Namespace
from pathlib import Path

from app.tools.mine_accept_rule_candidates import analyze, mine_candidates


def test_mine_candidates_finds_high_accept_pattern() -> None:
    rows = []
    for index in range(8):
        rows.append({"actual": "1", "text": f"饭店食品有虫要求退款{index}"})
    for index in range(2):
        rows.append({"actual": "0", "text": f"物业停车费纠纷{index}"})

    candidates = mine_candidates(rows, min_support=3, max_size=2)
    by_pattern = {row["pattern"]: row for row in candidates}

    assert by_pattern["虫 + 食品"]["accept_count"] == 8
    assert by_pattern["虫 + 食品"]["accept_rate"] == 1.0


def test_analyze_writes_candidate_files(tmp_path: Path) -> None:
    csv_path = tmp_path / "review.csv"
    csv_path.write_text(
        "actual,text\n"
        + "\n".join([f"1,饭店食品有虫要求退款{i}" for i in range(8)])
        + "\n"
        + "\n".join([f"0,物业停车费纠纷{i}" for i in range(8)]),
        encoding="utf-8-sig",
    )
    out_dir = tmp_path / "out"

    summary = analyze(
        Namespace(
            csv=str(csv_path),
            out_dir=str(out_dir),
            min_support=3,
            max_size=2,
            accept_rate=0.9,
            reject_rate=0.75,
            min_accept_count=3,
            min_reject_count=3,
        )
    )

    assert summary["high_accept_count"] > 0
    assert summary["high_reject_count"] > 0
    assert (out_dir / "accept_rule_candidates_high_accept.csv").exists()
    assert (out_dir / "accept_rule_candidates_high_reject.csv").exists()
