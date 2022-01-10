"""Features transformation module, gathering all Transformer classes.
"""
import os
import functools
from abc import abstractmethod
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA, KernelPCA, FactorAnalysis

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from utils.common import PIPELINE_TRAIN_NAME
from data.helpers import save_files


def get_scaler(scaling_method, minmax_range=None):
    """Returns the Scaler object corresponding to the provided scaling method.

    Args:
        scaling_method (str): scaling method (must be either `std`, `minmax` or `robust`)
        minmax_range (list|None): optional output range if minmax scaling (default `[0, 1]`).

    Returns:
        StandardScaler|MinMaxScaler|RobustScaler: Scaler object for the method.
    """
    a_t = 'the provided scaling method must be either `std`, `minmax` or `robust`'
    assert scaling_method in ['std', 'minmax', 'robust'], a_t
    if scaling_method == 'std':
        return StandardScaler()
    elif scaling_method == 'minmax':
        return MinMaxScaler(feature_range=[0, 1] if minmax_range is None else minmax_range)
    return RobustScaler()


class Transformer:
    """Base Transformer class.

    Provides methods for fitting and transforming period arrays.

    Techniques directly extending this class are assumed to fit their model on undisturbed traces
    and apply it to all. The `model_training` argument then specifies whether the model is fit to
    *all* undisturbed traces or only the largest one.

    Args:
        args (argparse.Namespace): parsed command-line arguments.
        output_path (str): path to save the model(s) and fitting information to.
        model_name (str): name/prefix of the trained model file(s) to save.
        model_training (str|None): if specified, must be either "all.training" or "largest.training".
    """
    def __init__(self, args, output_path, model_name='transformer', model_training=None):
        self.output_path = output_path
        self.model = None
        self.model_name = model_name
        self.model_training = model_training

    def fit_transform_datasets(self, datasets, datasets_info):
        """Returns the provided datasets transformed by (a) transformation model(s).

        By default, a single model is fit to training periods and applied to all datasets.

        Args:
            datasets (dict): of the form {`set_name`: `periods`};
                with `periods` a `(n_periods, period_length, n_features)` array, `period_length`
                depending on the period.
            datasets_info (dict): of the form {`set_name`: `periods_info`};
                with `periods_info` a list of the form `[file_name, trace_type, period_rank]`
                for each period of the set.

        Returns:
            dict: the transformed datasets, in the same format.
        """
        transformed = dict()
        transformed[PIPELINE_TRAIN_NAME] = self.fit_transform(datasets[PIPELINE_TRAIN_NAME], self.model_name)
        for ds_name in [n for n in list(datasets.keys()) if n != PIPELINE_TRAIN_NAME]:
            transformed[ds_name] = self.transform(datasets[ds_name])
        return transformed

    def fit(self, periods, model_file_name):
        """Fits the transformer's model to the provided `periods` array and saves it.

        Args:
            periods (ndarray): shape `(n_periods, period_length, n_features)`.
                `period_length` depending on the period.
            model_file_name (str): name of the model file to save.
        """
        # fit and save the transformation model on concatenated periods records
        if self.model_training is None or self.model_training == 'all.training':
            self.model.fit(np.concatenate(periods, axis=0))
        else:
            a_t = 'Transformer models can only be fit to all training traces or the largest only'
            assert self.model_training == 'largest.training', a_t
            self.model.fit(periods[np.argmax([len(p) for p in periods])])
        # save the transformation model as a pickle file (if can be directly pickled, else save within the class)
        try:
            save_files(self.output_path, {model_file_name: self.model}, 'pickle')
        except TypeError:
            print('Warning: the model could not be saved through the Transformer (ignore if saved elsewhere)')
            os.remove(os.path.join(self.output_path, f'{model_file_name}.pkl'))

    def transform(self, periods):
        """Returns the provided periods transformed by the transformer's model.

        Args:
            periods (ndarray): shape `(n_periods, period_length, n_features)`.
                `period_length` depending on the period.

        Returns:
            ndarray: the periods transformed by the model, in the same format.
        """
        # get transformed concatenated periods data
        transformed = self.model.transform(np.concatenate(periods, axis=0))
        # return the transformed periods back as a 3d-array
        return unravel_periods(transformed, [period.shape[0] for period in periods])

    def fit_transform(self, periods, model_file_name):
        """Fits the transformer's model to the provided periods and returns the transformed periods.

        Args:
            periods (ndarray): shape `(n_periods, period_length, n_features)`.
                `period_length` depending on the period.
            model_file_name (str): name of the model file to save.

        Returns:
            ndarray: the periods transformed by the model, in the same format.
        """
        self.fit(periods, model_file_name)
        return self.transform(periods)


class RegularKernelPCA(Transformer):
    """RegularKernelPCA class.

    A single kernel PCA model is fit to training periods and applied to all.
    """
    def __init__(self, args, output_path):
        super().__init__(args, output_path, model_name='pca', model_training=args.pca_training)
        # model to use along with any model-specific arguments
        self.n_components = args.pca_n_components
        self.kernel = args.pca_kernel
        # we use the standard PCA class for a linear kernel to more easily plot the evolution of explained variance
        if self.kernel == 'linear':
            self.model = PCA(n_components=self.n_components, svd_solver='full')
        else:
            self.model = KernelPCA(n_components=self.n_components, kernel=self.kernel)

    def fit(self, periods, model_file_name):
        """Overrides Transformer's method to save an explained variance evolution figure.

        Note: this figure is only saved if not using an explicit number of output components.
        """
        super().fit(periods, model_file_name)
        if not isinstance(self.n_components, int) and isinstance(self.model, PCA):
            plot_explained_variance(self.model, self.n_components, self.output_path)


class RegularFactorAnalysis(Transformer):
    """RegularFactorAnalysis class.

    A single Factor Analysis (FA) model is fit to training periods and applied to all.
    """
    def __init__(self, args, output_path):
        super().__init__(args, output_path, model_name='fa', model_training=args.fa_training)
        # model to use along with any model-specific arguments
        self.model = FactorAnalysis(n_components=args.fa_n_components)


class RegularScaler(Transformer):
    """RegularScaler class.

    A single scaler model is fit to training periods and applied to all.
    """
    def __init__(self, args, output_path):
        super().__init__(args, output_path, model_name='scaler', model_training=args.reg_scaler_training)
        # model to use along with any model-specific arguments
        self.model = get_scaler(args.scaling_method, args.minmax_range)


class TraceTransformer(Transformer):
    """Base TraceTransformer class.

    Instead of fitting and saving a single model to all training periods, a model
    is fit and saved per trace file.

    Periods from all datasets are grouped by file name and chronologically sorted within
    each group.

    Args:
        args (argparse.Namespace): parsed command-line arguments.
        output_path (str): path to save the models and fitting information to.
        model_name (str): name of the trained model file that should be saved to the output path.
    """
    def __init__(self, args, output_path, model_name='transformer'):
        super().__init__(args, output_path, model_name)

    def fit_transform_datasets(self, datasets, datasets_info):
        """Overrides Transformer's method to fit a model per trace file.
        """
        # group periods by file name according to their chronological rank
        periods_by_file = dict()
        for set_name in datasets:
            for period, (file_name, _, p_rank) in zip(datasets[set_name], datasets_info[set_name]):
                # the chronological rank of a period corresponds to its index in the group
                if file_name in periods_by_file:
                    # the group already contains at least one period
                    cur_group_length = len(periods_by_file[file_name])
                    if p_rank == cur_group_length:
                        periods_by_file[file_name].append(period)
                    else:
                        if p_rank > cur_group_length:
                            n_added = (p_rank - cur_group_length) + 1
                            periods_by_file[file_name] += [None for _ in range(n_added)]
                        periods_by_file[file_name][p_rank] = period
                else:
                    # the group is currently empty
                    periods_by_file[file_name] = [None for _ in range(p_rank + 1)]
                    periods_by_file[file_name][p_rank] = period
        # convert each group to an ndarray
        for file_name in periods_by_file:
            periods_by_file[file_name] = np.array(periods_by_file[file_name], dtype=object)

        # transform periods by file name
        transformed_by_file = dict()
        for file_name in periods_by_file:
            transformed_by_file[file_name] = self.fit_transform_trace(
                periods_by_file[file_name], file_name
            )

        # return transformed periods back in the original `datasets` format
        transformed = {k: [] for k in datasets}
        for set_name in datasets:
            # each dataset period can be recovered using its file name and chronological rank
            for file_name, _, p_rank in datasets_info[set_name]:
                transformed[set_name].append(transformed_by_file[file_name][p_rank].astype(float))
            transformed[set_name] = np.array(transformed[set_name], dtype=object)
        return transformed

    @abstractmethod
    def fit_transform_trace(self, trace_periods, file_name):
        """Fits a model and transforms the provided trace periods.

        Args:
            trace_periods (ndarray): chronologically sorted trace periods to transform.
            file_name (str): trace file name, used when saving the trained model.

        Returns:
            ndarray: the transformed trace periods in the same format.
        """


class FullTraceTransformer(TraceTransformer):
    """FullTraceTransformer class. All trace records are used to fit each file's model.

    /!\\ With this method, we assume all data of a given file to be available upfront.
    Applying it could therefore be unrealistic in an online setting.

    Another drawback of this method is that *all test data will be used for
    training the transformation model*, which may lead to overestimating the
    subsequent model's performance on the test set, unless we assume that traces
    encountered in production will be in the continuity or statistically similar
    to the ones used in this pipeline.

    Args:
        args (argparse.Namespace): parsed command-line arguments.
        output_path (str): path to save the models and fitting information to.
        model_name (str): name of the trained model file that should be saved to the output path.
    """
    def __init__(self, args, output_path, model_name='transformer'):
        super().__init__(args, output_path, model_name)

    def fit_transform_trace(self, trace_periods, file_name):
        """The trace periods are fit-transformed in batch mode.
        """
        return self.fit_transform(trace_periods, f'{self.model_name}_{file_name}')


class FullTraceScaler(FullTraceTransformer):
    """FullTraceScaler class.

    Traces are rescaled separately using a model trained on all their records.
    """
    def __init__(self, args, output_path):
        super().__init__(args, output_path, model_name='scaler')
        # model to use along with any model-specific arguments
        self.model = get_scaler(args.scaling_method, args.minmax_range)


class HeadTransformer(TraceTransformer):
    """HeadTransformer class. Only head records are used to fit each trace file's model.

    With this method, we only assume `head_size` records to be available upfront for each
    trace to fit their transformation model.

    We also add the option of fit-transforming all training traces, and use the resulting
    model as a weighted pre-trained model for fit-transforming test traces.

    Args:
        args (argparse.Namespace): parsed command-line arguments.
        output_path (str): path to save the models and fitting information to.
        model_name (str): name of the trained model file that should be saved to the output path.
    """
    def __init__(self, args, output_path, model_name='transformer'):
        super().__init__(args, output_path, model_name)
        # number of records used at the beginning of each trace to fit their transformer
        self.head_size = args.head_size
        # regular transformation model and weight (if relevant)
        self.regular_model = None
        self.regular_pretraining_weight = args.regular_pretraining_weight

    def fit_transform_datasets(self, datasets, datasets_info):
        """Overrides TraceTransformer's method to fit a model per trace file's "head".

        We also add the option of fit-transforming all training traces, and use the resulting
        model as a weighted pre-trained model for fit-transforming test traces.
        """
        # using a regular transformation model pretraining
        if self.regular_pretraining_weight != -1:
            transformed = dict()
            # train a model on the training periods and transform them in batch mode
            transformed[PIPELINE_TRAIN_NAME] = self.fit_transform(datasets[PIPELINE_TRAIN_NAME], self.model_name)
            self.regular_model = deepcopy(self.model)
            # transform test periods fitting their heads on top of this pre-trained model
            disturbed = dict()
            for key, item in zip(['data', 'info'], [datasets, datasets_info]):
                disturbed[key] = {n: v for n, v in item.items() if n != PIPELINE_TRAIN_NAME}
            # get transformed periods using `TraceTransformer.fit_transform_datasets`
            disturbed['transformed'] = super().fit_transform_datasets(disturbed['data'], disturbed['info'])
            for n in disturbed['transformed']:
                transformed[n] = disturbed['transformed'][n]
        else:
            # get all transformed periods using `TraceTransformer.fit_transform_datasets`
            transformed = super().fit_transform_datasets(datasets, datasets_info)
        return transformed

    def fit_transform_trace(self, trace_periods, file_name):
        """Transforms trace periods using a model fit to its first `head_size` records.
        """
        self.fit_trace_head(trace_periods, file_name)
        return self.transform(trace_periods)

    def fit_trace_head(self, trace_periods, file_name):
        """Fits a transformer model to the `head_size` first records of the trace periods.

        Trace periods will be concatenated, as if they were a single trace. An array of periods
        has to be provided in case the first trace period is shorter than `head_size`.

        If using a "regular model pretraining", the model fit to the trace's head
        will be combined with a model priorly trained on the `train` periods.

        Args:
            trace_periods (ndarray): the trace periods to fit the model to.
            file_name (str): the name of the trace file, used when saving the model.
        """
        trace_head = np.concatenate(trace_periods, axis=0)[:self.head_size, :]
        # consider the trace head as a single period to fit the model to
        self.fit(np.array([trace_head]), f'{self.model_name}_{file_name}')
        # combine the fit model with the one fit on `train` if using regular pretraining
        if self.regular_pretraining_weight != -1:
            self.combine_regular_head()

    @abstractmethod
    def combine_regular_head(self):
        """Combines the models fit on training periods and fit on a period's head.

        The model fit on train periods is available as `self.regular_model`.
        The model fit on the head of the current period is available as `self.model`.
        The combined model should be `self.model` modified inplace.
        """


class HeadScaler(HeadTransformer):
    """HeadScaler class.

    Implements `HeadTransformer` with features rescaling as the transformation.
    """
    def __init__(self, args, output_path):
        super().__init__(args, output_path, model_name='scaler')
        # model to use along with any model-specific arguments
        self.model = get_scaler(args.scaling_method, args.minmax_range)
        # model pretrained on `train` periods if relevant
        if self.regular_pretraining_weight != -1:
            self.regular_model = get_scaler(args.scaling_method, args.minmax_range)

    def combine_regular_head(self):
        """Combines the models fit on training periods and fit on a period's head."""
        # combine standard scalers
        if isinstance(self.model, StandardScaler):
            # the mean and std are combined using convex combinations
            regular_mean, regular_std = self.regular_model.mean_, self.regular_model.scale_
            head_mean, head_std = self.model.mean_, self.model.scale_
            combined_mean = self.regular_pretraining_weight * regular_mean + \
                (1 - self.regular_pretraining_weight) * head_mean
            combined_std = self.regular_pretraining_weight * regular_std + \
                (1 - self.regular_pretraining_weight) * head_std
            combined_var = combined_std ** 2
            self.model.mean_ = combined_mean
            self.model.scale_ = combined_std
            self.model.var_ = combined_var
        # combine minmax scalers
        else:
            # the min and max are combined keeping the min and max values only
            regular_min, regular_max = self.regular_model.data_min_, self.regular_model.data_max_
            head_min, head_max = self.model.data_min_, self.model.data_max_
            combined_min = np.array(
                [min(regular_min[i], head_min[i]) for i in range(regular_min.shape[0])]
            )
            combined_max = np.array(
                [max(regular_max[i], head_max[i]) for i in range(regular_max.shape[0])]
            )
            combined_range = combined_max - combined_min
            self.model.data_min_ = combined_min
            self.model.data_max_ = combined_max
            self.model.data_range_ = combined_range


class HeadOnlineTransformer(HeadTransformer):
    """HeadOnlineTransformer class. The traces are also transformed online in addition to the head model.

    Each trace's transformer is fit to its first `head_size` records in batch mode, then
    incrementally updated using an expanding or rolling window to transform newly arriving records.

    Using this method, we enable the transformer models to adapt to changing statistics
    within traces.

    Note: if a rolling window is used, its size will be fixed to `head_size`.

    Note: like for simple head transformation, we add the option of fit-transforming
    all training traces, and use the resulting model as a weighted pre-trained model
    for fit-transforming test traces.

    Args:
        args (argparse.Namespace): parsed command-line arguments.
        output_path (str): path to save the models and fitting information to.
        model_name (str): name of the trained model file that should be saved to the output path.
    """
    def __init__(self, args, output_path, model_name='transformer'):
        super().__init__(args, output_path, model_name)
        # use either expanding or rolling window estimates
        self.online_window_type = args.online_window_type

    def fit_transform_trace(self, trace_periods, file_name):
        """Fit-transforms the trace in batch mode for its first `head_size` records, online for others.
        """
        # fit a model to the head of the trace (possibly combined with a prior regular model)
        self.fit_trace_head(trace_periods, file_name)
        # transform trace periods online
        transformed_online = self.transform_trace_online(trace_periods)
        # replace the first `head_size` transformations with the output of the trained model
        flattened_transformed = np.concatenate(transformed_online, axis=0)
        flattened_transformed[:self.head_size, :] = self.transform(
            np.array([np.concatenate(trace_periods, axis=0)[:self.head_size, :]])
        )[0]
        return unravel_periods(flattened_transformed, [period.shape[0] for period in trace_periods])

    @abstractmethod
    def transform_trace_online(self, trace_periods):
        """Fits and transforms trace periods using expanding/rolling window estimates.

        /!\\ The transformed periods will be returned with the same number of records,
        with features that could not be transformed being replaced with `None` values.

        Args:
            trace_periods (ndarray): chronologically sorted trace periods to transform online.

        Returns:
             ndarray: the transformed trace periods in the same format.
        """


class HeadOnlineScaler(HeadOnlineTransformer):
    """HeadOnlineScaler class.

    Implements `HeadOnlineTransformer` with features rescaling as the transformation.
    """
    def __init__(self, args, output_path):
        super().__init__(args, output_path, model_name='scaler')
        # model to use along with any model-specific arguments
        self.model = get_scaler(args.scaling_method, args.minmax_range)
        # model pretrained on `train` periods if relevant
        if self.regular_pretraining_weight != -1:
            self.regular_model = get_scaler(args.scaling_method, args.minmax_range)

    def transform_trace_online(self, trace_periods):
        """Transforms trace periods using expanding/rolling window estimates for the offset and scale.

        We only consider Standard and MinMax scalers here. If a value of 0 is encountered
        for the scaling factor, it will be replaced by 1.
        """
        a_1 = 'supported online scalers are currently limited to `std` and `minmax`'
        assert isinstance(self.model, MinMaxScaler) or isinstance(self.model, StandardScaler), a_1
        # derive a whole trace DataFrame from its sorted period arrays
        trace_df = pd.DataFrame(np.concatenate(trace_periods, axis=0))
        # define the windowing method based on the online window type
        a_2 = 'supported online window types only include expanding and rolling'
        assert self.online_window_type in ['expanding', 'rolling'], a_2
        if self.online_window_type == 'expanding':
            trace_windowing = functools.partial(trace_df.expanding)
        else:
            trace_windowing = functools.partial(trace_df.rolling, self.head_size)
        if isinstance(self.model, MinMaxScaler):
            min_df, max_df = trace_windowing().min().shift(1), trace_windowing().max().shift(1)
            offset_df, scale_df = min_df, (max_df - min_df)
        else:
            offset_df, scale_df = trace_windowing().mean().shift(1), trace_windowing().std().shift(1)
        transformed_df = (trace_df - offset_df) / scale_df.replace(0, 1)
        return unravel_periods(transformed_df.values, [period.shape[0] for period in trace_periods])

    def combine_regular_head(self):
        """Combines the models fit on training periods and fit on a period's head."""
        # combine standard scalers
        if isinstance(self.model, StandardScaler):
            # the mean and std are combined using convex combinations
            regular_mean, regular_std = self.regular_model.mean_, self.regular_model.scale_
            head_mean, head_std = self.model.mean_, self.model.scale_
            combined_mean = self.regular_pretraining_weight * regular_mean + \
                (1 - self.regular_pretraining_weight) * head_mean
            combined_std = self.regular_pretraining_weight * regular_std + \
                (1 - self.regular_pretraining_weight) * head_std
            combined_var = combined_std ** 2
            self.model.mean_ = combined_mean
            self.model.scale_ = combined_std
            self.model.var_ = combined_var
        # combine minmax scalers
        else:
            # the min and max are combined keeping the min and max values only
            regular_min, regular_max = self.regular_model.data_min_, self.regular_model.data_max_
            head_min, head_max = self.model.data_min_, self.model.data_max_
            combined_min = np.array(
                [min(regular_min[i], head_min[i]) for i in range(regular_min.shape[0])]
            )
            combined_max = np.array(
                [max(regular_max[i], head_max[i]) for i in range(regular_max.shape[0])]
            )
            combined_range = combined_max - combined_min
            self.model.data_min_ = combined_min
            self.model.data_max_ = combined_max
            self.model.data_range_ = combined_range


def unravel_periods(raveled, period_lengths):
    """Returns the unraveled equivalent of `raveled`, separating back contiguous periods.

    Args:
        raveled (ndarray): raveled array of shape `(n_records, n_features)`.
        period_lengths (list): lengths of the contiguous periods to extract from the raveled array.

    Returns:
        ndarray: unraveled array of shape `(n_periods, period_size, n_features)` where `period_size`
            depends on the period.
    """
    start_idx = 0
    unraveled = []
    for period_length in period_lengths:
        unraveled.append(raveled[start_idx:start_idx+period_length])
        start_idx += period_length
    return np.array(unraveled, dtype=object)


def plot_explained_variance(pca_model, explained_variance, output_path):
    """Plots the data variance explained by the model with respect to the number of output components.

    We also show the actual number of components that has been kept by the performed transformation.

    Args:
        pca_model (sklearn.decomposition.PCA): the PCA object to describe, already fit to the training data.
        explained_variance (float): the actual amount of explained variance that was specified.
        output_path (str): the path to save the figure to.
    """
    dim = pca_model.n_components_
    plt.plot(np.cumsum(pca_model.explained_variance_ratio_))
    plt.plot(
        [dim], [explained_variance],
        marker='o', markersize=6, color='red', label=f'Reduced to {dim} dimensions'
    )
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)')
    plt.title('Explained Variance vs. Dimensionality')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_path, 'variance_vs_dimension.png'))
    plt.close()


# dictionary gathering references to the defined transformation classes
transformation_classes = {
    'regular_scaling': RegularScaler,
    'trace_scaling': FullTraceScaler,
    'head_scaling': HeadScaler,
    'head_online_scaling': HeadOnlineScaler,
    'pca': RegularKernelPCA,
    'fa': RegularFactorAnalysis
}
