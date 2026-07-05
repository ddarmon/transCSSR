import numpy as np
import pytest

from filter_data_methods import (
    compute_metrics,
    compute_precision,
    compute_recall,
    compute_tv,
)


def test_precision_and_recall_with_partial_predictions():
    truth = "011010"
    predictions = "01N110"

    assert compute_precision(truth, predictions) == pytest.approx(2 / 3)
    assert compute_recall(truth, predictions) == pytest.approx(2 / 3)


def test_accuracy_metric_matches_manual_calculation():
    truth = "0101"
    predictions = "0100"

    accuracy = compute_metrics(truth, predictions, metric="accuracy")
    assert accuracy == pytest.approx(0.75)


def test_f_score_aggregates_precision_and_recall():
    truth = "0110"
    predictions = "0111"

    f_score = compute_metrics(truth, predictions, metric="F")
    precision = compute_precision(truth, predictions)
    recall = compute_recall(truth, predictions)

    expected = 2 * precision * recall / (precision + recall)
    assert f_score == pytest.approx(expected)


def test_total_variation_distance_handles_probabilities():
    truth = "0101"
    probs = [0.2, 0.7, 0.1, 0.9]

    tv = compute_metrics(truth, probs, metric="tv")
    manual_tv = 0.5 * np.sum([
        abs(int(t) - p) + abs(1 - int(t) - (1 - p)) for t, p in zip(truth, probs)
    ]) / len(probs)

    assert tv == pytest.approx(manual_tv)


def test_precision_accepts_integer_predictions(integer_encoded_toy):
    truth = "010001"
    predictions = integer_encoded_toy["Y"]

    expected = compute_precision(truth, "".join(str(v) for v in predictions))
    assert compute_precision(truth, predictions) == pytest.approx(expected)
