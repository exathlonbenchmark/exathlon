"""Reconstruction-based outlier score assignment classes.
"""
import os
from abc import abstractmethod

import numpy as np

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from data.helpers import get_sliding_windows
from modeling.reconstruction.reconstructors import Reconstructor
from modeling.reconstruction.evaluation import get_mean_squared_error, get_feature_loss, get_discriminator_loss
from scoring.scorers import Scorer


class ReconstructionScorer(Scorer):
    """Reconstruction-based outlier score assignment base class.

    Derives record-wise outlier scores from the predictions of a trained Reconstructor object.

    Args:
        args (argparse.Namespace): parsed command-line arguments.
        normality_model (Reconstructor): trained Reconstructor object used to derive outlier scores.
        output_path (str): path to save the scoring model and information to.
        scorer (Scorer|None): if not None, the scorer will be initialized using the provided Scorer.
    """
    def __init__(self, args, normality_model, output_path, scorer=None):
        a_t = 'ReconstructionScorer objects should rely on predictions made by Reconstructor objects'
        assert isinstance(normality_model, Reconstructor), a_t
        super().__init__(args, normality_model, output_path, scorer)

    @abstractmethod
    def score_windows(self, X):
        """Reconstruction-based implementation.
        """

    def score_period(self, period):
        """Reconstruction-based implementation.
        """
        return self.score_period_from_w_scores(
            self.score_windows(get_sliding_windows(period, self.normality_model.window_size, 1))
        )

    def score_period_from_w_scores(self, window_scores):
        """Returns record-wise outlier scores from the provided `window_scores`.

        For now, we define the outlier score of a record as the average of the scores of
        the windows it belongs to (i.e. its "enclosing windows").

        The average score of a group of consecutive `window_size` windows gives
        the outlier score of the last record of the first window in the group.
        => Applying a rolling average of `window_size` elements on the window scores
        will hence give all records scores.
        => The `window_size-1` first and last records are handled by appending
        empty window scores before and after the window scores array.

        Args:
            window_scores (ndarray): window scores of shape `(n_windows,)`.

        Returns:
            ndarray: period record-wise scores of shape `(n_records,)`.
        """
        w = self.normality_model.window_size
        # perform a rolling average on the scores with empty overlaps being filled with 0 values
        rolling_scores = np.convolve(window_scores, np.ones(w) / w, mode='full')
        # remove 0 values from consideration when computing edge averages
        increasing_lengths = 1 + np.array(list(range(w - 1)))
        # divide the first/last `window_size-1` elements by the number of non-zero
        # elements instead of the window size (which accounted for 0 values)
        rolling_scores[:(w-1)] = w * rolling_scores[:(w-1)] / increasing_lengths
        rolling_scores[-(w-1):] = w * rolling_scores[-(w-1):] / np.flip(w)
        # the resulting scores are the ones of the period records
        return rolling_scores

    def score(self, periods):
        """Reconstruction-based implementation.

        To improve efficiency, we first compute the scores of all periods windows as
        one batch, then group the window scores back by period to compute the period-wise
        record scores.
        """
        concat_windows = []
        periods_n_windows = []
        print('extracting and concatenating periods windows...', end=' ', flush=True)
        for period in periods:
            period_windows = get_sliding_windows(period, self.normality_model.window_size, 1)
            concat_windows.append(period_windows)
            periods_n_windows.append(period_windows.shape[0])
        concat_windows = np.concatenate(concat_windows, axis=0).astype(np.float64)
        print('done.')
        print('computing windows outlier scores...', end=' ', flush=True)
        concat_w_scores = self.score_windows(concat_windows)
        print('done.')
        periods_scores, cursor = [], 0
        print('grouping back windows scores by period...', end=' ', flush=True)
        for n_windows in periods_n_windows:
            periods_scores.append(
                self.score_period_from_w_scores(concat_w_scores[cursor:(cursor + n_windows)])
            )
            cursor += n_windows
        print('done.')
        return np.array(periods_scores, dtype=object)


class MSEScorer(ReconstructionScorer):
    """Mean squared error-based scoring method (non-parametric).

    The outlier score of a window is set to its mean squared reconstruction error.
    """
    def __init__(self, args, reconstructor, output_path, model=None):
        super().__init__(args, reconstructor, output_path, model)

    def score_windows(self, X):
        """The window scores are the mean squared reconstruction errors.
        """
        return get_mean_squared_error(X, self.normality_model.reconstruct(X), sample_wise=True)


class MSEDiscriminatorScorer(ReconstructionScorer):
    """MSE + Discriminator loss scoring method (non-parametric).

    The outlier score of a window is set to the convex combination of the mean
    squared error and discriminator loss of its reconstruction by the model.
    """
    def __init__(self, args, reconstructor, output_path, model=None):
        super().__init__(args, reconstructor, output_path, model)
        # weight of the MSE in the convex combination
        self.alpha = args.mse_weight
        # encoder and discriminator networks to compute the discriminator loss
        self.encoder = reconstructor.encoder
        self.discriminator = reconstructor.discriminator

    def score_windows(self, X):
        """The window scores are the convex combination of the MSE and discriminator loss.
        """
        reconstructed = self.normality_model.reconstruct(X)
        # sample-wise MSE of the batch
        mse = get_mean_squared_error(X, reconstructed, sample_wise=True)
        # sample-wise discriminator loss of the batch
        dis_loss = get_discriminator_loss(
            reconstructed, self.encoder.predict(reconstructed), self.discriminator,
            sample_wise=True
        )
        return self.alpha * np.array(mse) + (1 - self.alpha) * np.array(dis_loss)


class MSEFeatureLossScorer(ReconstructionScorer):
    """MSE + Feature loss scoring method (non-parametric).

    The outlier score of a window is set to the convex combination of the mean
    squared error and feature loss of its reconstruction by the model.
    """
    def __init__(self, args, reconstructor, output_path, model=None):
        super().__init__(args, reconstructor, output_path, model)
        # weight of the MSE in the convex combination
        self.alpha = args.mse_weight
        # encoder and discriminator networks to compute the feature loss
        self.encoder = reconstructor.encoder
        self.discriminator = reconstructor.discriminator

    def score_windows(self, X):
        """The window scores are the convex combination of the MSE and feature loss.
        """
        reconstructed = self.normality_model.reconstruct(X)
        # sample-wise MSE of the batch
        mse = get_mean_squared_error(X, reconstructed, sample_wise=True)
        # latent representation of original and reconstructed samples
        z = dict()
        for windows, k in zip([X, reconstructed], ['X', 'reconstructed']):
            z[k] = self.encoder.predict(windows)
        # sample-wise feature loss of the batch
        ft_loss = get_feature_loss(
            X, z['X'], reconstructed, z['reconstructed'], self.discriminator,
            sample_wise=True
        )
        return self.alpha * mse + (1 - self.alpha) * ft_loss


# dictionary gathering references to the defined reconstruction-based scoring methods
scoring_classes = {
    'mse': MSEScorer,
    'mse.dis': MSEDiscriminatorScorer,
    'mse.ft': MSEFeatureLossScorer
}
