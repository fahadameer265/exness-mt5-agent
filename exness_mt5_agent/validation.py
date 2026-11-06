"""Time-series splits and the only classification metric we act on.

Accuracy is the wrong score here. Most bars are not 1:2 winners, so a model
that always predicts a loss looks accurate and never makes money. Precision
at the live threshold answers the question we actually trade: when the model
is willing to speak, how often was it right in the past?
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def expanding_folds(
    n_samples: int, n_splits: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Expanding walk-forward folds, same geometry as sklearn TimeSeriesSplit.

    Fold k trains on the earliest block and tests on the next block. The test
    block never sits earlier than the training block.
    """

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if n_samples < n_splits + 1:
        raise ValueError("Not enough samples for the requested folds.")
    test_size = n_samples // (n_splits + 1)
    if test_size < 1:
        raise ValueError("Fold size collapsed to zero.")
    indices = np.arange(n_samples)
    for fold in range(n_splits):
        test_start = (fold + 1) * test_size
        test_end = (fold + 2) * test_size if fold < n_splits - 1 else n_samples
        if test_start >= n_samples or test_start >= test_end:
            continue
        yield indices[:test_start], indices[test_start:test_end]


def purged_folds(
    bar_index: np.ndarray, n_splits: int, embargo: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Walk-forward folds with a gap of `embargo` source bars.

    Labels look `embargo` bars into the future. Without the gap, a training
    label would be computed from prices that also build the test features.
    The cut uses the original bar index, so it still works after we keep only
    the setup rows.
    """

    bar_index = np.asarray(bar_index)
    if len(bar_index) != len(np.unique(bar_index)):
        raise ValueError("bar_index values must be unique.")
    order = np.argsort(bar_index, kind="mergesort")
    if not np.array_equal(order, np.arange(len(bar_index))):
        raise ValueError("bar_index must already be time-sorted.")
    for train_pos, test_pos in expanding_folds(len(bar_index), n_splits):
        test_start = int(bar_index[test_pos[0]])
        keep = bar_index[train_pos] < (test_start - embargo)
        purged = train_pos[keep]
        if len(purged) == 0 or len(test_pos) == 0:
            continue
        yield purged, test_pos


def precision_at_threshold(
    y_true: np.ndarray, probability: np.ndarray, threshold: float
) -> tuple[float | None, int]:
    """Return (precision, number of predicted positives).

    Precision is None when the model never clears the threshold. That silence
    is a result, not a division-by-zero error: the cutoff is doing its job.
    """

    y_true = np.asarray(y_true).astype(int)
    probability = np.asarray(probability, dtype=float)
    chosen = probability >= threshold
    count = int(chosen.sum())
    if count == 0:
        return None, 0
    return float((y_true[chosen] == 1).mean()), count
