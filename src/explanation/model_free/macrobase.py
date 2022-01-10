"""MacroBase explanation discovery module.
"""
import os
import time
import warnings

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from explanation.model_free.model_free_explainers import ModelFreeExplainer


class MacroBase(ModelFreeExplainer):
    """MacroBase explanation discovery class.

    See https://cs.stanford.edu/~deepakn/assets/papers/macrobase-sigmod17.pdf for
    more details.
    """
    def __init__(self, args, output_path):
        super().__init__(args, output_path)
        # number of bins to use for histogram-based discretization
        self.n_bins = args.macrobase_n_bins
        # outlier support and relative risk ratio thresholds
        self.min_support = args.macrobase_min_support
        self.min_risk_ratio = args.macrobase_min_risk_ratio

    def explain_split_sample(self, normal_records, anomalous_records):
        """MacroBase implementation.

        An "explanation" is defined as a set of relevant features and corresponding anomalous
        value ranges (included start, excluded end). A single anomalous range is provided for
        each feature (returned as a list containing a single range at key "anomalous_ranges").

        Internally, these correspond to the most relevant itemset (i.e., set of items),
        each subset of which has sufficient outlier support and relative risk ratio.

        Features of the provided records are assumed continuous, and therefore require an
        additional "discretization" step (here performed using histogram-based binning on the
        normal records). In this context, what defines an item is the association of a feature
        and a value range.
        """
        start_t = time.time()
        # get boolean transactions with items of sufficient outlier support and relative risk ratio
        normal_transactions, anomalous_transactions, features_bins = self.get_boolean_transactions(
            normal_records, anomalous_records
        )
        # get itemsets with sufficient outlier support using FP-Growth
        itemsets_df = fpgrowth(anomalous_transactions, min_support=self.min_support, use_colnames=True)
        if itemsets_df.empty:
            warnings.warn('No explanation found for the sample.')
            return {'important_fts': [], 'anomalous_ranges': []}, time.time() - start_t

        # add relative risk ratio and cardinality information to the itemsets
        risk_ratios, n_items = [], []
        for itemset in itemsets_df['itemsets']:
            risk_ratios.append(
                self.get_itemset_risk_ratio(itemset, normal_transactions, anomalous_transactions)
            )
            n_items.append(len(itemset))
        itemsets_df = itemsets_df.assign(risk_ratio=risk_ratios)
        itemsets_df = itemsets_df.assign(cardinality=n_items)

        # sort by descending relative risk ratio, outlier support and cardinality
        itemsets_df = itemsets_df.sort_values(
            by=['risk_ratio', 'support', 'cardinality'], ascending=[False, False, False]
        )

        # important features are set to those in the itemset with best risk ratio, then support, then cardinality
        important_itemset = itemsets_df.iloc[0]['itemsets']
        important_fts, anomalous_ranges = [], []
        for item in important_itemset:
            ft, bin_idx = [int(v) for v in item.split('_')]
            important_fts.append(ft)
            anomalous_ranges.append([features_bins[ft][bin_idx], features_bins[ft][bin_idx+1]])
        # return the important features and corresponding anomalous ranges
        return {'important_fts': important_fts, 'anomalous_ranges': anomalous_ranges}, time.time() - start_t

    def get_boolean_transactions(self, normal_records, anomalous_records):
        """Returns the normal and anomalous boolean transactions corresponding to the provided records.

        Normal and anomalous transactions are returned as boolean DataFrames, where a value
        corresponds to the presence/absence of the corresponding item in the transaction.

        When building the boolean transactions, discrete items are:
        - created from an histogram-based binning of normal record values.
        - filtered so as to only keep those of sufficient support and relative risk ratio.

        Note: In this single-item case, we set the minimum number of occurrences of an itemset
        to the average bin count (i.e., average height) of the histogram of anomalous values.
        => the minimum support attribute is therefore not used here, but only in the multi-item
        itemset filtering performed later.

        Args:
            normal_records (ndarray): "normal" records of shape `(n_normal_records, n_features)`.
            anomalous_records: "anomalous records" of shape `(n_anomalous_records, n_features)`.

        Returns:
            pd.DataFrame, pd.DataFrame, list: normal and anomalous transaction DataFrames, along
                with the `bins` for each feature, where `bin_idx` corresponds to
                `(bins[bin_idx], bins[bin_idx+1])`, with included start and excluded end.
        """
        transactions, features_bins = {'normal': pd.DataFrame(), 'anomalous':  pd.DataFrame()}, []
        n_normal_records, n_anomalous_records = len(normal_records), len(anomalous_records)
        # for initial items, we set the minimum count as the average height of the anomalous histogram
        min_ft_count = max(n_anomalous_records // self.n_bins, 1)
        for ft in range(normal_records.shape[1]):
            # extract univariate time series for the feature
            normal_values, anomalous_values = normal_records[:, ft], anomalous_records[:, ft]

            # perform histogram-based binning and counting of the continuous feature
            normal_counts, normal_bins = np.histogram(normal_values, bins=self.n_bins)
            normal_bins[0], normal_bins[-1] = -np.inf, np.inf
            features_bins.append(normal_bins)
            anomalous_counts, _ = np.histogram(anomalous_values, bins=normal_bins)

            # compute relative risk ratio for each bin of sufficient support
            bin_ids = np.where(anomalous_counts >= min_ft_count)[0]
            # number of occurrences of the range within outliers and inliers
            a_o, a_i = anomalous_counts[bin_ids], normal_counts[bin_ids]
            # number of non-occurrences of the range within outliers and inliers
            b_o, b_i = n_anomalous_records - a_o, n_normal_records - a_i
            # total number of occurrences and non-occurrences of the range
            a = a_o + a_i
            b = b_o + b_i
            a[a == 0] = 1
            b[b == 0] = 1
            b_o[b_o == 0] = 1
            risk_ratios = np.nan_to_num((a_o / a) / (b_o / b))

            # only keep bins of sufficient relative risk
            for i, bin_id in zip(range(len(risk_ratios)), bin_ids):
                if risk_ratios[i] >= self.min_risk_ratio:
                    for k, v in zip(['normal', 'anomalous'], [normal_values, anomalous_values]):
                        transactions[k] = transactions[k].assign(
                            **{f'{ft}_{bin_id}': (v >= normal_bins[bin_id]) & (v < normal_bins[bin_id+1])}
                        )
        return transactions['normal'], transactions['anomalous'], features_bins

    @staticmethod
    def get_itemset_risk_ratio(itemset, normal_transactions, anomalous_transactions):
        """Returns the relative risk ratio of the provided `itemset` based on the transactions.

        Args:
            itemset (frozenset): itemset for which to compute the relative risk ratio.
            normal_transactions (pd.DataFrame): boolean DataFrame of normal transactions.
            anomalous_transactions (pd.DataFrame): boolean DataFrame of anomalous transactions.

        Returns:
            float: relative risk ratio of the itemset.
        """
        n_normal_records, n_anomalous_records = len(normal_transactions), len(anomalous_transactions)
        normal_occurrences = normal_transactions[list(itemset)].all(axis=1)
        anomalous_occurrences = anomalous_transactions[list(itemset)].all(axis=1)
        # number of occurrences of the itemset within outliers and inliers
        a_o, a_i = anomalous_occurrences.sum(), normal_occurrences.sum()
        # number of non-occurrences of the itemset within outliers and inliers
        b_o, b_i = n_anomalous_records - a_o, n_normal_records - a_i
        # total number of occurrences and non-occurrences of the itemset
        a = a_o + a_i
        b = b_o + b_i
        if a == 0:
            a = 1
        if b == 0:
            b = 1
        if b_o == 0:
            b_o = 1
        # relative risk ratio
        return np.nan_to_num((a_o / a) / (b_o / b))

    def classify_record(self, explanation, record):
        """MacroBase implementation.

        A record is predicted anomalous if all its important features are within their
        anomalous range (included start, excluded end).
        """
        features, ranges = explanation['important_fts'], explanation['anomalous_ranges']
        for ft, range_ in zip(features, ranges):
            if not (range_[0] <= record[ft] < range_[1]):
                return 0
        return 1
