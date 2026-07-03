from app.tools.train_accept_bert import build_train_test_rows, compute_class_weights, evaluate_predictions


def test_build_train_test_rows_keeps_both_labels() -> None:
    rows = []
    for index in range(20):
        rows.append({"row_number": index + 2, "text": f"食品变质要求退款{index}", "label": 1})
        rows.append({"row_number": index + 102, "text": f"物业费纠纷要求处理{index}", "label": 0})

    train_rows, test_rows = build_train_test_rows(rows, test_size=0.25, random_state=42)

    assert len(train_rows) == 30
    assert len(test_rows) == 10
    assert {row["label"] for row in train_rows} == {0, 1}
    assert {row["label"] for row in test_rows} == {0, 1}


def test_evaluate_predictions_counts_threshold_decisions() -> None:
    metrics = evaluate_predictions(
        y_true=[1, 1, 0, 0],
        y_pred=[1, 1, 0, 1],
        prob_accept=[0.91, 0.76, 0.12, 0.62],
        accept_threshold=0.75,
        reject_threshold=0.35,
    )

    assert metrics["accuracy"] == 0.75
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert metrics["threshold_decisions"] == {"ACCEPT": 2, "REJECT": 1, "REVIEW": 1}


def test_compute_class_weights_raises_minority_class_weight() -> None:
    weights = compute_class_weights([0, 1, 1, 1])

    assert weights[0] == 2.0
    assert round(weights[1], 4) == 0.6667
