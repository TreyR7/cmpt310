"""Helpers for turning predictions into inventory-style counts."""

from collections import Counter


def tally_predictions(predictions):
    """Return a stable, alphabetically ordered mapping of label counts."""
    counts = Counter(str(prediction) for prediction in predictions)
    return dict(sorted(counts.items()))
