"""Forecasting-based outlier score assignment classes.
"""
import os
from abc import abstractmethod

import numpy as np

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from modeling.forecasting.forecasters import Forecaster
from modeling.forecasting.helpers import get_period_sequence_target_pairs
from scoring.scorers import Scorer


class ForecastingScorer(Scorer):
    """Forecasting-based outlier score assignment base class.

    Derives record-wise outlier scores from the predictions of a trained Forecaster object.

    Args:
        args (argparse.Namespace): parsed command-line arguments.
        normality_model (Forecaster): trained Forecaster object used to derive outlier scores.
        output_path (str): path to save the scoring model and information to.
        scorer (Scorer|None): if not None, the scorer will be initialized using the provided Scorer.
    """
    def __init__(self, args, normality_model, output_path, scorer=None):
        a_t = 'ForecastingScorer objects should rely on predictions made by Forecaster objects'
        assert isinstance(normality_model, Forecaster), a_t
        super().__init__(args, normality_model, output_path, scorer)

    @abstractmethod
    def score_pairs(self, X, y):
        """Returns the outlier scores for the provided `(X, y)` (sequence, target) pairs.

        Args:
            X (ndarray): shape `(n_samples, n_back, n_features)`.
            y (ndarray): shape `(n_samples, n_features)` or `(n_samples, n_forward, n_features)`.

        Returns:
            ndarray: outlier scores for each pair, of shape `(n_pairs,)`.
        """

    def score_windows(self, X):
        """Forecasting-based implementation.

        The outlier score of a window is for now defined as the outlier score of its last
        record (assumed to be the Forecaster's target).
        => For this to work, the window size must therefore be the Forecaster's `n_back` parameter +1.
        """
        sequences, targets = [], []
        for x in X:
            sequences.append(x[:-1, :])
            targets.append(x[-1, :])
        return self.score_pairs(np.array(sequences), np.array(targets))

    def score_period(self, period):
        """Forecasting-based implementation.
        """
        return self.score_pairs(
            *get_period_sequence_target_pairs(
                period, self.normality_model.n_back, self.normality_model.n_forward
            )
        )

    def score(self, periods):
        """Forecasting-based implementation.

        To improve efficiency, we first concatenate the periods' (sequence, target) pairs,
        score them, and then separate back the periods scores.
        """
        sequences, targets = [], []
        n_pairs = []
        print('creating and concatenating (sequence, target) pairs of the periods...', end=' ', flush=True)
        for period in periods:
            p_s, p_t = get_period_sequence_target_pairs(
                period, self.normality_model.n_back, self.normality_model.n_forward
            )
            sequences.append(p_s)
            targets.append(p_t)
            n_pairs.append(p_s.shape[0])
        sequences = np.concatenate(sequences, axis=0).astype(np.float64)
        targets = np.concatenate(targets, axis=0).astype(np.float64)
        print('done.')
        print('computing outlier scores for the pairs...', end=' ', flush=True)
        scores = self.score_pairs(sequences, targets)
        print('done.')
        periods_scores, cursor = [], 0
        print('grouping back scores by period...', end=' ', flush=True)
        for np_ in n_pairs:
            periods_scores.append(scores[cursor:cursor+np_])
            cursor += np_
        print('done.')
        return np.array(periods_scores, dtype=object)


class RelativeErrorScorer(ForecastingScorer):
    """Relative forecasting error-based method (non-parametric).

    The outlier score of a record is set to its relative forecasting error by the model.
    """
    def __init__(self, args, forecaster, output_path, model=None):
        super().__init__(args, forecaster, output_path, model)

    def score_pairs(self, X, y):
        """The record scores are the relative forecasting errors.
        """
        y_pred = self.normality_model.predict(X)
        return np.linalg.norm(y - y_pred, axis=1) / np.linalg.norm(y)


# dictionary gathering references to the defined forecasting-based scoring methods
scoring_classes = {'re': RelativeErrorScorer}
