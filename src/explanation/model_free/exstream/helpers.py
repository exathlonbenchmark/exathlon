"""EXstream helpers module.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import entropy
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from sklearn.preprocessing import StandardScaler

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from data.helpers import get_sliding_windows
from metrics.ad_evaluators import extract_binary_ranges_ids


def get_feature_segments(normal_values, anomalous_values):
    """Returns the normal, anomalous and mixed "segments" for the provided feature values.

    Normal, anomalous and mixed segments are defined as intervals of values that appear only
    in normal data, only in anomalous data, and in both normal and anomalous data, respectively.

    Args:
        normal_values (ndarray): 1d-array of normal values.
        anomalous_values (ndarray): 1d-array of anomalous values.

    Returns:
        list, list, list: lists of normal, anomalous and mixed value segments. Each segment as a
            tuple of the form `(min_value, max_value, n_values)`.
    """
    common_values, labels = set(normal_values).intersection(set(anomalous_values)), []

    # label values as 1 if they appear in normal data only, -1 if anomalous data only, 0 if both
    for values, label in zip([normal_values, anomalous_values], [1, -1]):
        labels += [0 if v in common_values else label for v in values]
    labels = np.array(labels)

    # sort values in ascending order, opening lower and upper bounds (and adapting corresponding labels)
    values = np.concatenate([normal_values, anomalous_values])
    sorted_ids = np.argsort(values)
    sorted_values, sorted_labels = values[sorted_ids], labels[sorted_ids]
    for i, v in zip([0, -1], [-np.inf, np.inf]):
        sorted_values[i] = v

    # extract normal, anomalous and mixed segments
    segs_keys = ['normal', 'anomalous', 'mixed']
    segs_dict = {k: [] for k in segs_keys}
    for k, label in zip(segs_keys, [1, -1, 0]):
        for range_ in extract_binary_ranges_ids((sorted_labels == label).astype(int)):
            segs_values = sorted_values[slice(*range_)]
            segs_dict[k].append((segs_values[0], segs_values[-1], len(segs_values)))
    return segs_dict['normal'], segs_dict['anomalous'], segs_dict['mixed']


def get_single_feature_reward(normal_values, anomalous_values, normal_segs, anomalous_segs,
                              mixed_segs, n_records=None, n_normal_records=None, n_anomalous_records=None):
    """Returns the single-feature reward for the provided values and segments.

    For a given feature, this single-feature reward is defined as the ratio between its class
    entropy and regularized segmentation entropy. Here, all entropy values are computed in nats (base e).

    Args:
        normal_values (ndarray): 1d-array of normal values.
        anomalous_values (ndarray): 1d-array of anomalous values.
        normal_segs (list): list of normal segments of the form `(min_value, max_value, n_values)`.
        anomalous_segs (list): list of anomalous segments of the same form.
        mixed_segs (list): list of mixed segments of the same form.
        n_records (int|None): optional total number of records (normal + anomalous).
        n_normal_records (int|None): optional number of normal records.
        n_anomalous_records (int|None): optional number of anomalous records.

    Returns:
        float: the single-feature reward for the provided values and segments.
    """
    # compute numbers of records that were not provided
    if n_normal_records is None:
        n_normal_records = len(normal_values)
    if n_anomalous_records is None:
        n_anomalous_records = len(anomalous_values)
    if n_records is None:
        n_records = n_normal_records + n_anomalous_records

    # compute class entropy
    p_class = np.array([n_normal_records, n_anomalous_records]) / n_records
    class_entropy = entropy(p_class)

    # compute regularized segmentation entropy
    reg_seg_entropy = get_reg_segmentation_entropy(normal_segs, anomalous_segs, mixed_segs, n_records)

    # return single-feature reward
    if np.isposinf(reg_seg_entropy):
        return 0.0
    return class_entropy / reg_seg_entropy


def get_reg_segmentation_entropy(normal_segs, anomalous_segs, mixed_segs, n_records):
    """Returns the regularized segmentation entropy for the provided segments.

    Args:
        normal_segs (list): list of normal segments of the form `(min_value, max_value, n_values)`.
        anomalous_segs (list): list of anomalous segments of the same form.
        mixed_segs (list): list of mixed segments of the same form.
        n_records (int): total number of records.

    Returns:
        float: the regularized segmentation entropy for the segments (+infinity if only mixed).
    """
    if len(normal_segs) + len(anomalous_segs) == 0:
        # all values appear in both normal and anomalous data: largest entropy
        return np.inf

    # segmentation entropy
    p_segs = np.array([c for _, _, c in normal_segs + anomalous_segs + mixed_segs]) / n_records
    segmentation_entropy = entropy(p_segs)

    # add sum of entropy values for the worst-case (uniform) orderings of mixed segment (regularization)
    segmentation_entropy += sum([np.log(c) for _, _, c in mixed_segs])
    return segmentation_entropy


def get_fp_features(normal_records, scaled_std_threshold):
    """Returns the "false positive" feature ids for the provided normal records.

    Unlike in the original paper, this false positive identification is performed
    in an unsupervised way. "False positive" features are defined as features that either:
    - have an especially large standard variation (as compared to other features).
    - tend to strictly increase or decrease throughout the records.

    To assess whether a feature "tends" to strictly increase and decrease, we consider
    the evolution of its average value within jumping windows of 1/5th the number of records.

    Args:
        normal_records (ndarray): normal records of shape `(n_records, n_features)`.
        scaled_std_threshold (float): scaled standard deviation threshold above which
            to define a feature as "false positive".

    Returns:
        list: false positive feature ids for the normal records.
    """
    # features with especially large standard deviation
    scaler = StandardScaler()
    scaled_stds = scaler.fit_transform(np.std(normal_records, axis=0).reshape(-1, 1))
    false_positives = list(np.argwhere(scaled_stds >= scaled_std_threshold)[:, 0])

    # features tending to increase or decrease throughout the records
    n_records, n_features = normal_records.shape
    window_size = max(1, n_records // 5)
    for ft in range(n_features):
        if ft not in false_positives:
            # get average values within jumping windows throughout the records
            window_averages = np.array([
                np.mean(w) for w in get_sliding_windows(
                    normal_records[:, ft], window_size, window_size, True
                )
            ])
            # add as false positive if averages strictly increase or decrease
            if np.all(window_averages[1:] > window_averages[:-1]) \
                    or np.all(window_averages[1:] < window_averages[:-1]):
                false_positives.append(ft)
    return false_positives


def get_low_reward_features(feature_rewards):
    """Returns the "low-reward" feature ids based on the provided rewards.

    This corresponds to the "reward leap filtering" step of EXstream. We first sort features
    by reward, then define "low-reward" features as those ranking below the sharpest reward leap.

    Args:
        feature_rewards (ndarray): 1d-array of feature rewards, whose ids correspond to features.

    Returns:
        list: "low-reward" feature ids.
    """
    # descending feature rewards and corresponding reward leaps
    sorted_rewards = sorted(feature_rewards, reverse=True)
    sorted_rewards_leaps = -np.diff(sorted_rewards)
    # reward before the sharpest reward leap
    reward_threshold = sorted_rewards[np.argmax(sorted_rewards_leaps)]
    # return features having a reward strictly lower than the threshold
    return list(np.argwhere(feature_rewards < reward_threshold)[:, 0])


def get_clustered_features(records):
    """Returns a cluster number for each feature of the provided records.

    Features are clustered according to pairwise Pearson correlation.

    Args:
        records (ndarray): records whose features to cluster, of shape `(n_records, n_features)`.

    Returns:
        ndarray: the cluster numbers for each feature of shape `(n_features,)`, numbers start from 1.
    """
    # pairwise correlations of features (similarity matrix) (constants lead to NaNs converted to zeros)
    corr_matrix = np.nan_to_num(pd.DataFrame(records).corr())
    # self-correlations are always one (even for constant features)
    np.fill_diagonal(corr_matrix, 1)
    # deduce pairwise distances from correlations (distance matrix)
    pw_distances = 1 - np.abs(corr_matrix)
    # fix slightly negative distances due to numerical instabilities
    pw_distances[pw_distances < 0] = 0
    # cluster features using the pairwise distances
    linkage = sch.linkage(ssd.squareform(pw_distances), method='complete')
    # flatten clusters using half the maximum distance as the cophenetic distance threshold
    return sch.fcluster(linkage, 0.5 * np.max(pw_distances), 'distance')
