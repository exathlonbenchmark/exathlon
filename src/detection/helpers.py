"""Final detection helpers module.
"""
import os
import numpy as np

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from data.helpers import get_numpy_from_numpy_list


def threshold_period_scores(period_scores, threshold):
    """Thresholds `period_scores` and returns their corresponding binary predictions.

    Every score above or equal to `threshold` is turned to 1, every score strictly below
    is turned to 0.

    Args:
        period_scores (ndarray): 1d-array of record-wise outlier scores for the period.
        threshold (float): outlier score value above which a record should be assigned 1.

    Returns:
        ndarray: record-wise binary predictions for the period, having the same
            shape as `period_scores`.
    """
    return np.array(period_scores >= threshold, dtype=int)


def threshold_scores(periods_scores, threshold):
    """Thresholds `periods_scores` and returns their corresponding binary predictions.

    Args:
        periods_scores (ndarray): shape (n_periods, period_length).
            Where `period_length` depends on the period.
        threshold (float): outlier score value above which a record should be assigned 1.

    Returns:
        ndarray: record-wise binary predictions for the periods, having the same
            shape as `periods_scores`.
    """
    # flatten the scores prior to thresholding to improve efficiency
    flattened_preds = threshold_period_scores(np.concatenate(periods_scores, axis=0), threshold)
    # recover periods separation
    periods_preds, cursor = [], 0
    for period_length in [len(period) for period in periods_scores]:
        periods_preds.append(flattened_preds[cursor:cursor+period_length])
        cursor += period_length
    return get_numpy_from_numpy_list(periods_preds)
