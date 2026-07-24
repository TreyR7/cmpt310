from smart_livestock_gate.counting.tally import tally_predictions


def test_tally_predictions_counts_and_sorts_labels():
    predictions = ["sheep", "cow", "sheep", "chicken"]

    assert tally_predictions(predictions) == {
        "chicken": 1,
        "cow": 1,
        "sheep": 2,
    }

