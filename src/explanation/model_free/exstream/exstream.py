"""EXstream explanation discovery module.
"""
import os
import time
import warnings

import numpy as np

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from explanation.model_free.model_free_explainers import ModelFreeExplainer
from explanation.model_free.exstream.helpers import (
    get_feature_segments, get_single_feature_reward, get_low_reward_features, get_fp_features,
    get_clustered_features
)


class EXstream(ModelFreeExplainer):
    """EXstream explanation discovery class.

    See http://www.lix.polytechnique.fr/Labo/Yanlei.Diao/publications/xstream-edbt2017.pdf
    for more details.
    """
    def __init__(self, args, output_path):
        super().__init__(args, output_path)
        # scaled std threshold above which to define a feature as "false positive"
        self.scaled_std_threshold = args.exstream_fp_scaled_std_threshold

    def explain_split_sample(self, normal_records, anomalous_records):
        """EXstream implementation.

        An "explanation" is defined as a set of relevant features and corresponding anomalous
        value ranges (included start, included end). Multiple anomalous ranges can be provided
        for each feature (returned as a list of ranges lists at key "anomalous_ranges").
        """
        start_t = time.time()
        n_features = normal_records.shape[1]
        n_normal_records, n_anomalous_records = len(normal_records), len(anomalous_records)
        n_records = n_normal_records + n_anomalous_records

        # compute single-feature rewards and anomalous value ranges
        feature_rewards, anomalous_ranges = np.zeros(n_features), []
        for ft in range(n_features):
            # extract univariate time series for the feature
            normal_values, anomalous_values = normal_records[:, ft], anomalous_records[:, ft]

            # extract normal, anomalous and mixed value "segments" (i.e., ranges) for the feature
            normal_segs, anomalous_segs, mixed_segs = get_feature_segments(normal_values, anomalous_values)

            # compute single-feature reward based on the extracted segments
            feature_rewards[ft] = get_single_feature_reward(
                normal_values, anomalous_values, normal_segs, anomalous_segs, mixed_segs,
                n_records, n_normal_records, n_anomalous_records
            )

            # set the anomalous ranges for the feature as its multivalued mixed and anomalous segments
            anomalous_ranges.append([(s, e) for s, e, _ in anomalous_segs + mixed_segs if s != e])

        # set the rewards of "false positive" features to zero
        fp_features = get_fp_features(normal_records, self.scaled_std_threshold)
        if len(fp_features) > 0:
            feature_rewards[fp_features] = 0

        # set the rewards of "low-reward" features (i.e., ranking below the sharpest reward leap) to zero
        lr_features = get_low_reward_features(feature_rewards)
        if len(lr_features) > 0:
            feature_rewards[lr_features] = 0

        # only keep the feature of largest reward from each correlation cluster
        clustered_features = get_clustered_features(np.concatenate([normal_records, anomalous_records]))
        for cluster in set(clustered_features):
            cluster_features = np.argwhere(clustered_features == cluster)[:, 0]
            # only keep feature with largest reward to represent the cluster
            cluster_representative = cluster_features[feature_rewards[cluster_features].argmax()]
            feature_rewards[[f for f in cluster_features if f != cluster_representative]] = 0

        # important features are features of non-zero rewards
        important_fts = list(np.argwhere(feature_rewards != 0)[:, 0])

        # return important features and corresponding anomalous ranges
        if len(important_fts) == 0:
            warnings.warn('No explanation found for the sample.')
        return {
            'important_fts': important_fts,
            'anomalous_ranges': [anomalous_ranges[i] for i in important_fts]
        }, time.time() - start_t

    def classify_record(self, explanation, record):
        """EXstream implementation.

        A record is predicted anomalous if all its important features are within one of
        their possible anomalous ranges (included start, included end).
        """
        features, features_ranges = explanation['important_fts'], explanation['anomalous_ranges']
        for ft, ranges in zip(features, features_ranges):
            ft_pred = 0
            for range_ in ranges:
                if range_[0] <= record[ft] <= range_[1]:
                    ft_pred = 1
                    break
            if ft_pred == 0:
                # feature is not within any of its anomalous ranges
                return 0
        # all features are within one of their anomalous ranges
        return 1
