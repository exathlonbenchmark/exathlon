"""Model-free explanation discovery helpers module.
"""
import numpy as np


def get_split_sample(sample, sample_labels):
    """Returns the original `sample` split into its "normal" and "anomalous" parts.

    Args:
        sample (ndarray): sample of shape `(sample_length, n_features)`.
        sample_labels (ndarray): sample labels of shape `(sample_length,)`.

    Returns:
        ndarray, ndarray: the normal and anomalous records of the sample, respectively.
    """
    return sample[sample_labels == 0], sample[sample_labels > 0]


def get_merged_sample(normal_records, anomalous_records, positive_class=1):
    """Returns the provided normal and anomalous records as a single explanation sample and labels.

    Args:
        normal_records (ndarray): normal records of shape `(n_normal_records, n_features)`.
        anomalous_records: anomalous records of shape `(n_anomalous_records, n_features)`.
        positive_class (int): positive class to use for the anomaly labels (default 1).

    Returns:
        ndarray, ndarray: the explanation sample and corresponding labels, respectively.
    """
    sample = np.concatenate([normal_records, anomalous_records])
    sample_labels = np.concatenate(
        [np.zeros(len(normal_records)), positive_class * np.ones(len(anomalous_records))]
    ).astype(int)
    return sample, sample_labels
