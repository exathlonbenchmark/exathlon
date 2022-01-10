import os
import pickle
from abc import abstractmethod

import numpy as np

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from data.helpers import get_sliding_windows
from modeling.reconstruction.evaluation import get_mean_squared_error, get_feature_loss, get_discriminator_loss


class ReconstructionScorer:
    """Reconstruction-based outlier score assignment base class.

    Derives record-wise outlier scores from the predictions of a trained Reconstructor object.

    Args:
        args (argparse.Namespace): parsed command-line arguments.
        reconstructor (Reconstructor): trained Reconstructor object used to derive outlier scores.
        output_path (str): path to save the scoring model and information to.
        model (misc|None): if not None, the scorer will be initialized using the provided model.
    """
    def __init__(self, args, reconstructor, output_path, model=None):
        self.reconstructor = reconstructor
        self.output_path = output_path
        self.model = model

    @classmethod
    def from_file(cls, args, reconstructor, model_root_path):
        """Returns a ReconstructorScorer object with its parameters initialized from a pickled model.

        Args:
            args (argparse.Namespace): parsed command-line arguments.
            reconstructor (Reconstructor): trained Reconstructor object used to derive outlier scores.
            model_root_path (str): root path to the pickle model file (assumed named "model.pkl").

        Returns:
            ReconstructionScorer: pre-initialized ReconstructionScorer object.
        """
        full_model_path = os.path.join(model_root_path, 'model.pkl')
        print(f'loading scoring model file {full_model_path}...', end=' ', flush=True)
        model = pickle.load(open(full_model_path, 'rb'))
        print('done.')
        return cls(args, reconstructor, '', model)

    @abstractmethod
    def score_windows(self, X):
        """Returns the outlier scores for the provided `X` window samples.

        Args:
            X (ndarray): shape `(n_samples, window_size, n_features)`.

        Returns:
            ndarray: outlier scores for each sample, of shape `(n_samples,)`.
        """

    def score_period(self, period):
        """Returns the record-wise outlier scores for the provided period.

        For now, we define the outlier score of a record as the average of the scores of
        the windows it belongs to (i.e. its "enclosing windows").

        The average score of a group of consecutive `window_size` windows gives
        the outlier score of the last record of the first window in the group.
        => Applying a rolling average of `window_size` elements on the window scores
        will hence give all records scores.
        => The `window_size-1` first and last records are handled by appending
        empty window scores before and after the window scores array.

        Args:
            period (ndarray): shape `(period_length, n_features)`.

        Returns:
            ndarray: scores of shape `(period_length,)`.
        """
        # get scores for all consecutive windows of the period
        w = self.reconstructor.window_size
        window_scores = self.score_windows(get_sliding_windows(period, w, 1))
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

    def score_period_from_w_scores(self, window_scores):
        """Returns the record-wise outlier scores from the provided `window_scores`.

        See `score_period` method for details on the methodology.

        Args:
            window_scores (ndarray): window scores of shape `(n_windows,)`.

        Returns:
            ndarray: period record-wise scores of shape `(n_records,)`.
        """
        w = self.reconstructor.window_size
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
        """Returns the record-wise outlier scores for the provided periods.

        To improve efficiency, we first compute the scores of all periods windows as
        one batch, then group the window scores back by period to compute the period-wise
        record scores.

        Args:
            periods (ndarray): shape `(n_periods, period_length, n_features)`.
                Where `period_length` depends on the period.

        Returns:
            ndarray: shape `(n_periods, period_length,)`.
        """
        concat_windows = []
        periods_n_windows = []
        print('extracting and concatenating periods windows...', end=' ', flush=True)
        for period in periods:
            period_windows = get_sliding_windows(period, self.reconstructor.window_size, 1)
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
        return get_mean_squared_error(X, self.reconstructor.reconstruct(X), sample_wise=True)


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
        reconstructed = self.reconstructor.reconstruct(X)
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
        reconstructed = self.reconstructor.reconstruct(X)
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


# dictionary gathering references to the defined scoring methods
scoring_classes = {
    'mse': MSEScorer,
    'mse.dis': MSEDiscriminatorScorer,
    'mse.ft': MSEFeatureLossScorer
}
