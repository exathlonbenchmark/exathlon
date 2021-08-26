"""Reconstruction-oriented utility module.
"""
import numpy as np


def get_period_windows(period, window_size, window_step):
    """Returns sliding windows of `window_size` elements extracted every `window_step` from `period`.

    Args:
        period (ndarray): input period of shape `(period_length, n_features)`.
        window_size (int): number of elements of the windows to extract.
        window_step (int): step size between two adjacent windows to extract.

    Returns:
        ndarray: windows array of shape `(n_windows, window_size, n_features)`.
    """
    windows = []
    for start_idx in range(0, period.shape[0] - window_size + 1, window_step):
        windows.append(period[start_idx:(start_idx + window_size)])
    return np.array(windows, dtype=np.float64)
