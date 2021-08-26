import os
import pickle
from abc import abstractmethod

import numpy as np

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from modeling.forecasting.helpers import get_period_sequence_target_pairs


class ForecastingScorer:
    """Forecasting-based outlier score assignment base class.

    Derives record-wise outlier scores from the predictions of a trained Forecaster object.

    Args:
        args (argparse.Namespace): parsed command-line arguments.
        forecaster (Forecaster): trained Forecaster object whose predictions are used to derive outlier scores.
        output_path (str): path to save the scoring model and information to.
        model (misc|None): if not None, the scorer will be initialized using the provided model.
    """
    def __init__(self, args, forecaster, output_path, model=None):
        self.forecaster = forecaster
        self.output_path = output_path
        self.model = model

    @classmethod
    def from_file(cls, args, forecaster, model_root_path):
        """Returns a ForecastingScorer object with its parameters initialized from a pickled model.

        Args:
            args (argparse.Namespace): parsed command-line arguments.
            forecaster (Forecaster): trained Forecaster object whose predictions are used to derive outlier scores.
            model_root_path (str): root path to the pickle model file (assumed named "model.pkl").

        Returns:
            ForecastingScorer: pre-initialized ForecastingScorer object.
        """
        full_model_path = os.path.join(model_root_path, 'model.pkl')
        print(f'loading scoring model file {full_model_path}...', end=' ', flush=True)
        model = pickle.load(open(full_model_path, 'rb'))
        print('done.')
        return cls(args, forecaster, '', model)

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
        """Returns the outlier scores for the provided `X` window samples.

        The outlier score of a window is for now defined as the outlier score of its
        last record (assumed to be the Forecaster's target).
        => For this to work, the window size must therefore be the Forecaster's `n_back`
         parameter +1.

        Args:
            X (ndarray): shape `(n_samples, window_size, n_features)`.

        Returns:
            ndarray: outlier scores for each sample, of shape `(n_samples,)`.
        """
        sequences, targets = [], []
        for x in X:
            sequences.append(x[:-1, :])
            targets.append(x[-1, :])
        return self.score_pairs(np.array(sequences), np.array(targets))

    def score_period(self, period):
        """Returns the record-wise outlier scores for the provided period.

        Args:
            period (ndarray): shape `(period_length, n_features)`.

        Returns:
            ndarray: shape `(period_length - n_back,)`. Where `n_back` is the number of records used to forecast.
        """
        return self.score_pairs(
            *get_period_sequence_target_pairs(period, self.forecaster.n_back, self.forecaster.n_forward)
        )

    def score(self, periods):
        """Returns the record-wise outlier scores for the provided periods.

        To improve efficiency, we first concatenate the periods' (sequence, target) pairs, score them,
        and then separate back the periods scores.

        Args:
            periods (ndarray): shape `(n_periods, period_length, n_features)`;
                `period_length` depending on the period.

        Returns:
            ndarray: shape `(n_periods, period_length - n_back,)`.
        """
        sequences, targets = [], []
        n_pairs = []
        print('creating and concatenating (sequence, target) pairs of the periods...', end=' ', flush=True)
        for period in periods:
            p_s, p_t = get_period_sequence_target_pairs(period, self.forecaster.n_back, self.forecaster.n_forward)
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
        y_pred = self.forecaster.predict(X)
        return np.linalg.norm(y - y_pred, axis=1) / np.linalg.norm(y)


# dictionary gathering references to the defined scoring methods
scoring_classes = {'re': RelativeErrorScorer}
