import os
import pickle

import numpy as np

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from scoring.forecasting.forecasting_scorers import ForecastingScorer
from scoring.reconstruction.reconstruction_scorers import ReconstructionScorer
from detection.helpers import threshold_scores
from detection.threshold_selectors import selector_classes


class Detector:
    """Final anomaly detection class.

    Assigns binary anomaly predictions to the records of its input periods based on
    its scorer and threshold.

    Args:
        args (argparse.Namespace): parsed command-line arguments.
        scorer (Scorer): object assigning record-wise outlier scores to input periods.
        output_path (str): path to save the threshold and information to.
        threshold (float): if not None, the detector will be initialized using the provided
            threshold value.
    """
    def __init__(self, args, scorer, output_path, threshold=None):
        # object performing outlier score assignment
        self.scorer = scorer
        # object performing threshold selection
        self.threshold_selector = selector_classes[args.thresholding_method](args, output_path)
        if threshold is not None:
            self.threshold_selector.threshold = threshold

    @classmethod
    def from_file(cls, args, scorer, threshold_root_path):
        """Returns a Detector object with its threshold initialized from an existing file.

        Args:
            args (argparse.Namespace): parsed command-line arguments.
            scorer (Scorer): object assigning record-wise outlier scores to input periods.
            threshold_root_path (str): root path to the threshold pickle file
                (assumed named "threshold.pkl").

        Returns:
            Detector: pre-initialized Detector object.
        """
        full_threshold_path = os.path.join(threshold_root_path, 'threshold.pkl')
        print(f'loading threshold file {full_threshold_path}...', end=' ', flush=True)
        threshold = pickle.load(open(full_threshold_path, 'rb'))
        print('done.')
        return cls(args, scorer, '', threshold)

    def fit(self, *, X, y=None):
        """Fits the detector's threshold using the `(X[, y])` samples distribution.

        Args:
            X (ndarray): shape `(n_pairs, n_back, n_features)` for forecasting-based scorers;
                shape `(n_windows, window_size, n_features)` for reconstruction-based scorers.
            y (ndarray): targets of shape `(n_pairs, n_features)` or
                `(n_pairs, n_forward, n_features)`. Only relevant for forecasting models.
        """
        a_t = 'only forecasting-based and reconstruction-based scorers are supported'
        assert isinstance(self.scorer, ForecastingScorer) or \
               isinstance(self.scorer, ReconstructionScorer), a_t
        if isinstance(self.scorer, ForecastingScorer):
            a_t = '(sequence, target) pairs have to be provided for forecasting-based scorers'
            assert not (X is None or y is None), a_t
            scores = self.scorer.score_pairs(X, y)
        else:
            a_t = 'only window samples have to be provided for reconstruction-based scorers'
            assert X is not None and y is None, a_t
            scores = self.scorer.score_windows(X)
        self.threshold_selector.fit(scores)

    def predict_period(self, period):
        """Returns record-wise binary predictions for the provided period.

        Args:
            period (ndarray): shape `(period_length, n_features)`.

        Returns:
            ndarray: predictions for the period, whose shape depends on the scorer type.
                - Forecasting-based scorers: shape `(period_length - n_back,)`.
                    Where `n_back` is the number of records used to forecast.
                - Reconstruction-based scorers: shape `(period_length,)`.
        """
        return threshold_scores(
            np.array([self.scorer.score_period(period)]), self.threshold_selector.threshold
        )[0]

    def predict(self, periods):
        """Returns record-wise binary predictions for the provided periods.

        Args:
            periods (ndarray): shape `(n_periods, period_length, n_features)`.
                Where `period_length` depends on the period.

        Returns:
            ndarray: predictions for the periods, whose shape depends on the scorer type.
                - Forecasting-based scorers: shape `(n_periods, period_length - n_back)`.
                    Where `n_back` is the number of records used to forecast.
                - Reconstruction-based scorers: shape `(n_periods, period_length)`.
        """
        return threshold_scores(self.scorer.score(periods), self.threshold_selector.threshold)
