from argparse import Namespace
from pathlib import Path

from app.tools.simulate_accept_rule_candidates import simulate


def test_simulate_accept_rule_candidates_counts_false_accepts(tmp_path: Path) -> None:
    review_csv = tmp_path / "review.csv"
    review_csv.write_text(
        "row_number,actual,text\n"
        "1,1,商家价格多收要求处理\n"
        "2,1,商家价格未明码标价\n"
        "3,0,商家价格物业收费纠纷\n",
        encoding="utf-8-sig",
    )
    candidates_csv = tmp_path / "candidates.csv"
    candidates_csv.write_text(
        "pattern,total,accept_rate\n"
        "价格 + 商家,3,0.9\n",
        encoding="utf-8-sig",
    )
    out_dir = tmp_path / "out"

    summary = simulate(
        Namespace(
            review_csv=str(review_csv),
            candidates_csv=str(candidates_csv),
            out_dir=str(out_dir),
            min_support=1,
            min_accept_rate=0.9,
        )
    )

    assert summary["auto_accept_rows"] == 3
    assert summary["true_accept_rows"] == 2
    assert summary["false_accept_rows"] == 1
    assert (out_dir / "accept_rule_simulation_false_accept.csv").exists()
