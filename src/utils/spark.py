"""Spark-specific utility module.
"""
import os
import copy
import argparse
from itertools import permutations

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir))
sys.path.append(src_path)
from utils.common import DATA_ROOT

# spark streaming application ids
APP_IDS = list(range(1, 11))
# spark streaming trace types
TRACE_TYPES = [
    'undisturbed', 'bursty_input', 'bursty_input_crash',
    'stalled_input', 'cpu_contention', 'process_failure'
]
# anomaly types (the integer label for a type is its corresponding index in the list +1)
ANOMALY_TYPES = [
    'bursty_input', 'bursty_input_crash', 'stalled_input',
    'cpu_contention', 'driver_failure', 'executor_failure', 'unknown'
]
# path dictionary interface containing the relevant root paths to load data from
DATA_PATHS_DICT = dict(
    {'labels': DATA_ROOT},
    **{f'app{i}': os.path.join(DATA_ROOT, f'app{i}') for i in APP_IDS}
)
# getter function returning the application id for a given trace file name
def get_app_from_name(file_name): return int(file_name.split('_')[0])


# original sampling period of records
SAMPLING_PERIOD = '1s'

# spark-specific default values for the common command-line arguments
DEFAULT_ARGS = {
    # data partitioning arguments
    'n_starting_removed': 0,
    'n_ending_removed': 0,
    'pre_sampling_period': '15s',

    # data transformation arguments
    'alter_bundles': 'spark_bundles',
    'alter_bundle_idx': 0,
    'sampling_period': '15s',
    'downsampling_position': 'first',
    'transform_chain': 'trace_scaling',
    # if a transformation step is repeated, the same arguments are used for all its instances
    'head_size': 240,
    'online_window_type': 'expanding',
    # if not -1, weight of a regular pretraining of the scaler in
    # the convex combination with its head/head-online training
    'regular_pretraining_weight': -1,
    'scaling_method': 'std',
    # only relevant for "regular" scaling
    'reg_scaler_training': 'all.training',
    'minmax_range': [0, 1],
    'pca_n_components': 13,
    'pca_kernel': 'linear',
    'pca_training': 'all.training',
    'fa_n_components': 13,
    'fa_training': 'all.training',

    # normality modeling arguments
    'modeling_split': 'stratified.split',
    'modeling_split_seed': 21,
    'n_period_strata': 3,
    'modeling_val_prop': 0.15,
    'modeling_test_prop': 0.15,
    'model_type': 'ae',
    # FORECASTING MODELS #
    'n_back': 40,
    'n_forward': 1,
    # RNN
    'rnn_unit_type': 'lstm',
    'rnn_n_hidden_neurons': [144, 40],
    'rnn_dropout': 0.0,
    'rnn_rec_dropout': 0.0,
    'rnn_optimizer': 'adam',
    'rnn_learning_rate': 7.869 * (10 ** -4),
    'rnn_n_epochs': 200,
    'rnn_batch_size': 32,
    # RECONSTRUCTION MODELS #
    'window_size': 40,
    'window_step': 1,
    # autoencoder
    'ae_latent_dim': 32,
    'ae_type': 'dense',
    'ae_enc_n_hidden_neurons': [200],
    'ae_dec_last_activation': 'linear',
    'ae_dropout': 0.0,
    'ae_dense_layers_activation': 'relu',
    'ae_rec_unit_type': 'lstm',
    'ae_rec_dropout': 0.0,
    'ae_loss': 'mse',
    'ae_optimizer': 'adam',
    'ae_learning_rate': 3.602 * (10 ** -4),
    'ae_n_epochs': 200,
    'ae_batch_size': 32,
    # BiGAN
    'bigan_latent_dim': 32,
    'bigan_enc_type': 'rec',
    'bigan_enc_arch_idx': -1,
    'bigan_enc_rec_n_hidden_neurons': [100],
    'bigan_enc_rec_unit_type': 'lstm',
    'bigan_enc_conv_n_filters': 32,
    'bigan_enc_dropout': 0.0,
    'bigan_enc_rec_dropout': 0.0,
    'bigan_gen_type': 'rec',
    'bigan_gen_last_activation': 'linear',
    'bigan_gen_arch_idx': -1,
    'bigan_gen_rec_n_hidden_neurons': [100],
    'bigan_gen_rec_unit_type': 'lstm',
    'bigan_gen_conv_n_filters': 64,
    'bigan_gen_dropout': 0.0,
    'bigan_gen_rec_dropout': 0.0,
    'bigan_dis_type': 'conv',
    'bigan_dis_arch_idx': 0,
    'bigan_dis_x_rec_n_hidden_neurons': [30, 10],
    'bigan_dis_x_rec_unit_type': 'lstm',
    'bigan_dis_x_conv_n_filters': 32,
    'bigan_dis_x_dropout': 0.0,
    'bigan_dis_x_rec_dropout': 0.0,
    'bigan_dis_z_n_hidden_neurons': [32, 10],
    'bigan_dis_z_dropout': 0.0,
    'bigan_dis_threshold': 0.0,
    'bigan_dis_optimizer': 'adam',
    'bigan_enc_gen_optimizer': 'adam',
    'bigan_dis_learning_rate': 0.0004,
    'bigan_enc_gen_learning_rate': 0.0001,
    'bigan_n_epochs': 200,
    'bigan_batch_size': 32,

    # outlier score derivation arguments
    'scoring_method': 'mse',
    'mse_weight': 0.5,

    # anomaly detection evaluation arguments
    'evaluation_type': 'ad2',
    'recall_alpha': 0.0,
    'recall_omega': 'default',
    'recall_delta': 'flat',
    'recall_gamma': 'dup',
    'precision_omega': 'default',
    'precision_delta': 'flat',
    'precision_gamma': 'dup',
    'f_score_beta': 1.0,

    # threshold selection arguments
    'thresholding_method': ['std', 'mad', 'iqr'],
    'thresholding_factor': [1.5, 2.0, 2.5, 3.0],
    'n_iterations': [1, 2],
    'removal_factor': [1.0]
}


def is_type_combination(x):
    """Argparse parsing function: returns `x` if it is either `.` or any combination of trace types.

    Note: we use this instead of the `choices` parameter because the amount of choices otherwise
    floods the argument's `help`.

    Args:
        x (str): the command-line argument to be checked.

    Returns:
        str: `x` if it is valid. Raises `ArgumentTypeError` otherwise.
    """
    if x == '.':
        return x
    type_choices = ['.'.join(t) for r in range(1, len(TRACE_TYPES) + 1) for t in permutations(TRACE_TYPES, r)]
    if x in type_choices:
        return x
    raise argparse.ArgumentTypeError('Argument has to be either `.` for all trace types, '
                                     f'or any dot-separated combination of {TRACE_TYPES}')


def add_specific_args(parsers, pipeline_step, pipeline_steps):
    """Adds Spark-specific command-line arguments to the input parsers.

    The arguments added to the provided step are propagated to the next ones in the pipeline.

    Args:
        parsers (dict): the existing parsers dictionary.
        pipeline_step (str): the pipeline step to add arguments to (as the name of its main script).
        pipeline_steps (list): the complete list of step names in the pipeline.

    Returns:
        dict: the new parsers, extended with Spark-specific arguments for the step.
    """
    step_index = pipeline_steps.index(pipeline_step)
    new_parsers = copy.deepcopy(parsers)
    arg_names, arg_params = [], []
    if pipeline_step == 'make_datasets':
        # considered traces might be restricted to a given application id (0 for no restriction)
        arg_names.append('--app-id')
        arg_params.append({
            'default': 1,
            'type': int,
            'choices': [0] + list(set(APP_IDS) - {7, 8}),
            'help': 'application id (0 for all, we exclude 7 and 8 since they '
                    'respectively have no disturbed/undisturbed traces)'
        })
        # considered traces might be restricted to a set of trace types (`.` for no restriction)
        arg_names.append('--trace-types')
        arg_params.append({
            'default': '.',
            'type': is_type_combination,
            'help': 'restricted trace types to consider (dot-separated, `.` for no restriction)'
        })
        # we might want to ignore anomalies occurring on nodes with no recorded spark application component
        ignored_anomalies_choices = ['none', 'os.only']
        arg_names.append('--ignored-anomalies')
        arg_params.append({
            'default': 'none',
            'choices': ignored_anomalies_choices,
            'help': 'ignored anomalies (either none or anomalies that had no effect on Spark metrics)'
        })
    for i, arg_name in enumerate(arg_names):
        for step in pipeline_steps[step_index:]:
            new_parsers[step].add_argument(arg_name, **arg_params[i])
    return new_parsers
