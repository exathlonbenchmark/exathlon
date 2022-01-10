"""Modeling split module. Constituting the train/val/test sets for normality modeling.
"""
import os
import random
from abc import abstractmethod

import numpy as np

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from utils.common import MODELING_SET_NAMES, forecasting_choices, reconstruction_choices
from data.helpers import get_aligned_shuffle, get_sliding_windows
from modeling.forecasting.helpers import get_period_sequence_target_pairs


def get_sampling_f_and_targets_presence(model_type):
    """Returns the sampling function and presence of targets based on the model type.

    Args:
        model_type (str): type of model used for normality modeling.

    Returns:
        func, bool: sampling function and presence of sample targets for the provided type of model.
    """
    a_t = 'model type must be a supported forecasting or reconstruction-based method'
    assert model_type in forecasting_choices + reconstruction_choices, a_t
    print(f'setting samples creation function for the {model_type} model...', end=' ', flush=True)
    if model_type in forecasting_choices:
        sampling_f, are_targets = get_period_sequence_target_pairs, True
    else:
        sampling_f, are_targets = get_sliding_windows, False
    print('done.')
    return sampling_f, are_targets


class DataSplitter:
    """Data splitting class for the modeling task.

    Args:
        args (argparse.Namespace): parsed command-line arguments.
        output_path (str|None): optional output path to save data splitting information to.
    """
    def __init__(self, args, output_path=None):
        self.model_type = args.model_type
        self.output_path = output_path
        # proportion of samples taken as the validation and test sets
        self.val_prop = args.modeling_val_prop
        self.test_prop = args.modeling_test_prop
        self.random_seed = args.modeling_split_seed

    def get_modeling_split(self, periods, periods_info, **sampling_args):
        """Returns the final shuffled train/val/test samples for the modeling task.

        Depending on the model type, the samples will be returned along with their corresponding targets.

        Args:
            periods (ndarray): `(n_periods, period_length, n_features)`; `period_length` depends on period.
            periods_info (list): period information lists (one per period).
            **sampling_args: arguments to be passed to the samples extraction function, returning
                a set of samples(, targets) for a given ndarray period.

        Returns:
            dict: datasets as `{(X|y)_(train|val|test): value}`, with `value` ndarray of samples/targets.
        """
        # fix random seeds for reproducibility across calls
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        # set the sampling function and the presence of targets according to the model type
        sampling_f, are_targets = get_sampling_f_and_targets_presence(self.model_type)

        # return the constituted datasets
        return self.custom_modeling_split(periods, periods_info, are_targets, sampling_f, **sampling_args)

    @abstractmethod
    def custom_modeling_split(self, periods, periods_info, are_targets, sampling_f, **sampling_args):
        """Returns the final shuffled `train/val/test` samples for the modeling task.

        Args:
            periods (ndarray): `(n_periods, period_length, n_features)`; `period_length` depends on period.
            periods_info (list): period information lists (one per period).
            are_targets (bool): whether or not the sampling function will return targets along with samples.
            sampling_f (func): samples extraction function, returns a set of samples(, targets) for a period.
            **sampling_args: arguments to be passed to the samples extraction function.
        """


class RandomSplitter(DataSplitter):
    """Random splitting class.

    The train/val/test datasets are randomly constituted from all the available periods.

    E.g. if we train on 80% of data, we randomly pick 80% of samples from all periods and send them
    to the training set.
    """
    def __init__(self, args, output_path=None):
        super().__init__(args, output_path=output_path)

    def custom_modeling_split(self, periods, periods_info, are_targets, sampling_f, **sampling_args):
        t = ' and targets' if are_targets else ''
        print(f'extracting samples{t} from all available periods...', end=' ', flush=True)
        samples, targets = np.array([]), (np.array([]) if are_targets else None)
        for period in periods:
            p_items = sampling_f(period, **sampling_args)
            p_samples = p_items[0] if are_targets else p_items
            samples = np.concatenate([samples, p_samples]) if samples.size != 0 else p_samples
            if are_targets:
                targets = np.concatenate([targets, p_items[1]]) if targets.size != 0 else p_items[1]
        print('done.')
        print(f'shuffling samples{t}...', end=' ', flush=True)
        # shuffle samples (and possibly targets) and send them to train/val/test
        datasets, set_names = dict(), MODELING_SET_NAMES
        shuffled = get_aligned_shuffle(samples, targets if are_targets else None)
        samples = shuffled[0] if are_targets else shuffled
        val_idx = int((1 - self.test_prop - self.val_prop) * samples.shape[0])
        test_idx = int((1 - self.test_prop) * samples.shape[0])
        print(f'done. (n_samples={samples.shape[0]}, val_idx={val_idx}, test_idx={test_idx}')
        for n, slice_ in zip(set_names, [slice(val_idx), slice(val_idx, test_idx), slice(test_idx, None)]):
            print(f'constituting {n} samples{t}...', end=' ', flush=True)
            datasets[f'X_{n}'] = samples[slice_, ...]
            if are_targets:
                datasets[f'y_{n}'] = shuffled[1][slice_, ...]
            print('done.')
        return datasets


class StratifiedSplitter(DataSplitter):
    """Stratified splitting class.

    The train/val/test datasets are constituted by randomly sampling within fixed-sized bins in the periods.

    E.g. if we train on 80% of data and use 3 period strata, we divide every period into 3 equal parts
    (or "bins"), randomly pick 80% of samples within each bin and send them to the training set.

    This way, we can be sure that every period and bin will be represented in all datasets (in particular in
    the test data).
    """
    def __init__(self, args, output_path=None):
        super().__init__(args, output_path=output_path)
        # number of bins per period, representing the period strata in the stratified sampling
        self.n_period_strata = args.n_period_strata

    def custom_modeling_split(self, periods, periods_info, are_targets, sampling_f, **sampling_args):
        # samples (and possibly targets) to be returned
        set_names = MODELING_SET_NAMES
        keys_list = [f'{p}_{k}' for p in (['X', 'y'] if are_targets else ['X']) for k in set_names]
        datasets = {k: np.array([]) for k in keys_list}

        print(f'dividing every period into {self.n_period_strata} equal bins...', end=' ', flush=True)
        period_bins = []
        for period in periods:
            # add list of ndarray bins
            period_bins += np.array_split(period, self.n_period_strata)
        period_bins = np.array(period_bins, dtype=object if len(period_bins) > 1 else np.float64)
        print('done.')

        print('sampling, shuffling and adding each bin to the datasets...', end=' ', flush=True)
        for bin_ in period_bins:
            items = sampling_f(bin_, **sampling_args)
            samples, targets = items if are_targets else (items, None)
            # shuffle samples (and possibly targets) and add them to `train/val/test`
            shuffled = get_aligned_shuffle(samples, targets if are_targets else None)
            samples = shuffled[0] if are_targets else shuffled
            val_idx = int((1 - self.test_prop - self.val_prop) * samples.shape[0])
            test_idx = int((1 - self.test_prop) * samples.shape[0])
            for n, slice_ in zip(set_names, [slice(val_idx), slice(val_idx, test_idx), slice(test_idx, None)]):
                if datasets[f'X_{n}'].size != 0:
                    datasets[f'X_{n}'] = np.concatenate([datasets[f'X_{n}'], samples[slice_, ...]])
                else:
                    datasets[f'X_{n}'] = samples[slice_, ...]
                if are_targets:
                    if datasets[f'y_{n}'].size != 0:
                        datasets[f'y_{n}'] = np.concatenate([datasets[f'y_{n}'], shuffled[1][slice_, ...]])
                    else:
                        datasets[f'y_{n}'] = shuffled[1][slice_, ...]
        print('done.')
        return datasets


# use a getter function to access references to data splitter classes to solve cross-import issues
def get_splitter_classes():
    """Returns a dictionary gathering references to the defined data splitter classes.

    A getter function is used to solve potential cross-import issues if using
    data-specific splitters.
    """
    return {
        'random.split': RandomSplitter,
        'stratified.split': StratifiedSplitter
    }