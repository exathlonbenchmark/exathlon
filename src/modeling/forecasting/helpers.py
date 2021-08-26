"""Forecasting helpers module.
"""
import numpy as np


def get_trimmed_periods(periods, n_back):
    """Returns the trimmed version of the provided periods, removing the first `n_back` records for each one.

    Args:
        periods (ndarray): periods to trim of shape `(n_periods, period_length, ...)`.
        n_back (int): amount of records to remove from the beginning of each period.

    Returns:
        ndarray: trimmed periods of shape `(n_periods, period_length - n_back, ...)`.
    """
    trimmed = []
    for period in periods:
        trimmed.append(period[n_back:, ...])
    return np.array(trimmed, dtype=object)


def get_sequence_target_pairs(periods, n_back, n_forward):
    """Converts the provided periods to `(sequence, target)` pairs for model training and inference.

    Args:
        periods (ndarray): input periods of shape `(n_periods, period_length, n_features)`.
        n_back (int): number of records used by the model to perform forecasts.
        n_forward (int): number of records to forecast forward.

    Returns:
        ndarray, ndarray: the extracted `(sequence, target)` pairs.
            `sequences`: shape `(n_pairs, n_back, features)`.
            `targets`: shape `(n_pairs, n_forward, n_features)` if `n_forward` is not 1 else `(n_pairs, n_features)`.
    """
    sequences, targets = [], []
    for period in periods:
        # get (sequence, target) pairs for the period
        p_sequences, p_targets = get_period_sequence_target_pairs(period, n_back, n_forward)
        sequences.append(p_sequences)
        targets.append(p_targets)
    # concatenate (sequence, target) pairs from all periods to form the final pair arrays
    return np.concatenate(sequences, axis=0), np.concatenate(targets, axis=0)


def get_period_sequence_target_pairs(period, n_back, n_forward):
    """Extracts `(sequence, target)` pairs from the provided period.

    Args:
        period (ndarray): input period of shape `(period_length, n_features)`.
        n_back (int): number of records used by the model to perform forecasts.
        n_forward (int): number of records to forecast forward.

    Returns:
        ndarray, ndarray: the extracted `(sequence, target)` pairs.
            `sequences`: shape `(n_pairs, n_back, features)`.
            `targets`: shape `(n_pairs, n_forward, n_features)` if `n_forward` is not 1 else `(n_pairs, n_features)`.
    """
    sequences, targets = [], []
    for i in range(period.shape[0] - n_back - n_forward + 1):
        sequences.append(period[i:i+n_back, :])
        targets.append(period[i+n_back:i+n_back+n_forward, :] if n_forward > 1 else period[i+n_back, :])
    return np.array(sequences, dtype=np.float64), np.array(targets, dtype=np.float64)
