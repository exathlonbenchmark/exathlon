"""Threshold selection module.

Gathers various threshold selection methods.
"""
import os
import pickle
from abc import abstractmethod

import numpy as np


class ThresholdSelector:
    """Base ThresholdSelector class.

    Args:
        args (argparse.Namespace): parsed command-line arguments.
        output_path (str): path to save the computed threshold and information to.
    """
    def __init__(self, args, output_path):
        self.output_path = output_path
        self.threshold = None

    def fit(self, scores):
        """Selects and saves the threshold based on the distribution of `scores`.

        For now, the provided outlier scores are the ones of unlabeled "samples",
        i.e. `(X, y)` pairs or windows, assumed (mostly) normal.

        Args:
            scores (ndarray): 1d-array of outlier scores.
        """
        self.select_threshold(scores)
        # save the computed threshold to the output path
        print(f'saving selected threshold under {self.output_path}...', end=' ', flush=True)
        os.makedirs(self.output_path, exist_ok=True)
        with open(os.path.join(self.output_path, 'threshold.pkl'), 'wb') as pickle_file:
            pickle.dump(self.threshold, pickle_file)
        print('done.')

    @abstractmethod
    def select_threshold(self, scores):
        """Performs the actual outlier score threshold selection.

        Args:
            scores (ndarray): 1d-array of outlier scores.
        """


class TwoStatSelector(ThresholdSelector):
    """Base class for thresholding methods based on 2 simple statistics.

    These statistical methods are presented in https://dl.acm.org/doi/pdf/10.1145/3371425.3371427.
    They all set the threshold value based on 2 statistics computed on the input scores:

    `threshold := stat_1 + thresholding_factor * stat_2`.

    To ignore the most obvious outliers in the statistical estimates, all these simple
    thresholding algorithms can be applied multiple times. For each iteration i:
    - A `threshold_i` value is computed based on the distribution of the available scores.
    - The scores above `removal_factor * threshold_i` are removed for the next iteration.
    The last computed threshold is the one retained and saved.

    Args:
        args (argparse.Namespace): parsed command-line arguments.
        output_path (str): path to save the computed threshold and information to.
    """
    def __init__(self, args, output_path):
        super().__init__(args, output_path)
        # factor of `stat_2` above `stat_1` defining the threshold
        self.thresholding_factor = args.thresholding_factor
        # number of iterations of the algorithm
        self.n_iterations = args.n_iterations
        # factor of the threshold at iteration i above which scores are removed for i+1
        self.removal_factor = args.removal_factor

    def select_threshold(self, scores):
        """Selects and overwrites the previous threshold at each iteration."""
        sub_scores = scores
        for i in range(self.n_iterations):
            self.select_iteration_threshold(sub_scores)
            # only keep scores below `removal_factor * threshold` for the next iteration
            sub_scores = scores[scores <= self.removal_factor * self.threshold]

    @abstractmethod
    def select_iteration_threshold(self, sub_scores):
        """Selects a threshold on `sub_scores` for a given algorithm iteration."""


class STDSelector(TwoStatSelector):
    """Standard Deviation (STD) method. `stat_1 == mean` and `stat_2 == std`.
    """
    def __init__(self, args, output_path):
        super().__init__(args, output_path)

    def select_iteration_threshold(self, sub_scores):
        mean, std = np.mean(sub_scores), np.std(sub_scores)
        self.threshold = mean + self.thresholding_factor * std


class MADSelector(TwoStatSelector):
    """Median Absolute Deviation (MAD) method. `stat_1 == median` and `stat_2 == MAD`.

    Assuming a normal distribution of the outlier scores, we defined `MAD` as:
    `MAD = 1.4826 * median(|X - median(X)|)`, with `X` the outlier scores.

    See https://en.wikipedia.org/wiki/Median_absolute_deviation for more details.
    """
    def __init__(self, args, output_path):
        super().__init__(args, output_path)

    def select_iteration_threshold(self, sub_scores):
        median = np.median(sub_scores)
        mad = 1.4826 * np.median(np.abs(sub_scores - median))
        self.threshold = median + self.thresholding_factor * mad


class IQRSelector(TwoStatSelector):
    """Inter-Quartile Range (IQR) method. `stat_1 == Q3` and `stat_2 == Q3 - Q1 == IQR`.

    See https://en.wikipedia.org/wiki/Interquartile_range for more details.
    """
    def __init__(self, args, output_path):
        super().__init__(args, output_path)

    def select_iteration_threshold(self, sub_scores):
        q1, q3 = np.percentile(sub_scores, 25), np.percentile(sub_scores, 75)
        iqr = q3 - q1
        self.threshold = q3 + self.thresholding_factor * iqr


# dictionary gathering references to the defined threshold selection methods
selector_classes = {
    'std': STDSelector,
    'mad': MADSelector,
    'iqr': IQRSelector
}
