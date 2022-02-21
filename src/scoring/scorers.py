"""Outlier score assignment classes.
"""
import os
import pickle
from abc import abstractmethod


class Scorer:
    """Outlier score assignment base class.

    Derives record-wise outlier scores from the predictions of a trained "normality model".

    Args:
        args (argparse.Namespace): parsed command-line arguments.
        normality_model (Forecaster|Reconstructor): "normality model" whose predictions will be
            used to derive outlier scores (for now either a `Forecaster` or `Reconstructor` object.
        output_path (str): path to save the scoring model and information to.
        scorer (Scorer|None): if not None, the scorer will be initialized using the provided Scorer.
    """
    def __init__(self, args, normality_model, output_path, scorer=None):
        self.normality_model = normality_model
        self.output_path = output_path
        self.scorer = scorer

    @classmethod
    def from_file(cls, args, normality_model, scorer_root_path):
        """Returns a Scorer object with its parameters initialized from a pickled model.

        Args:
            args (argparse.Namespace): parsed command-line arguments.
            normality_model (Forecaster|Reconstructor): "normality model" whose predictions will be
                used to derive outlier scores (for now either a `Forecaster` or `Reconstructor` object.
            scorer_root_path (str): root path to the pickle scorer file (assumed named "scorer.pkl").

        Returns:
            Scorer: pre-initialized Scorer object.
        """
        full_scorer_path = os.path.join(scorer_root_path, 'scorer.pkl')
        print(f'loading scoring model file {full_scorer_path}...', end=' ', flush=True)
        scorer = pickle.load(open(full_scorer_path, 'rb'))
        print('done.')
        return cls(args, normality_model, '', scorer)

    @abstractmethod
    def score_windows(self, X):
        """Returns the outlier scores for the provided `X` windows.

        Args:
            X (ndarray): windows to score of shape `(n_windows, window_size, n_features)`.

        Returns:
            ndarray: outlier scores for each window, of shape `(n_windows,)`.
        """

    @abstractmethod
    def score_period(self, period):
        """Returns the record-wise outlier scores for the provided period.

        The number of returned outlier scores `n_scores` depends on the type of normality
        model used by the Scorer.

        - For Reconstructor normality models: `n_scores := period_length`.
        - For Forecaster normality models: `n_scores := period_length - n_back`, where `n_back` is
            the number of records used to forecast.

        Args:
            period (ndarray): period whose records to score of shape `(period_length, n_features)`.

        Returns:
            ndarray: period outlier scores of shape `(n_scores,)`.
        """

    @abstractmethod
    def score(self, periods):
        """Returns the record-wise outlier scores for the provided periods.

        See `score_period` for a description of `n_scores`.

        Args:
            periods (ndarray): periods whose records to score of shape `(n_periods, period_length, n_features)`;
                where `period_length` depends on the period.

        Returns:
            ndarray: shape `(n_periods, n_scores,)`.
        """
