"""General utility module. Mainly handling paths and command-line arguments.
"""
import os
import argparse
import warnings
import importlib

import pandas as pd
from dotenv import load_dotenv, find_dotenv

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir))
sys.path.append(src_path)

# main pipeline step names
PIPELINE_STEPS = [
    'make_datasets',
    'build_features',
    'train_model',
    'train_scorer',
    'train_detector'
]
# explanation discovery step name (not included in `PIPELINE_STEPS` as its index may vary)
EXPLANATION_STEP = 'train_explainer'
# reporting step name
REPORTING_STEP = 'report_results'
# callable file names relating to the explainable anomaly detection pipeline
PIPELINE_CALLABLES = PIPELINE_STEPS + [EXPLANATION_STEP, 'run_pipeline']
# main pipeline and normality modeling dataset names
PIPELINE_TRAIN_NAME, PIPELINE_TEST_NAME = 'train', 'test'
MODELING_TRAIN_NAME, MODELING_VAL_NAME, MODELING_TEST_NAME = 'train', 'val', 'test'
PIPELINE_SET_NAMES = [PIPELINE_TRAIN_NAME, PIPELINE_TEST_NAME]
MODELING_SET_NAMES = [MODELING_TRAIN_NAME, MODELING_VAL_NAME, MODELING_TEST_NAME]

# output paths
load_dotenv(find_dotenv())
DATA_ROOT = os.getenv('DATA_ROOT')
OUTPUTS_ROOT = os.getenv('OUTPUTS_ROOT')
INTERIM_ROOT = os.path.join(OUTPUTS_ROOT, 'data', 'interim')
PROCESSED_ROOT = os.path.join(OUTPUTS_ROOT, 'data', 'processed')
MODELS_ROOT = os.path.join(OUTPUTS_ROOT, 'models')
REPORTS_ROOT = os.path.join(OUTPUTS_ROOT, 'reports')

# a lot of default command-line arguments depend on the data we use
USED_DATA = os.getenv('USED_DATA').lower()
DEFAULTS = importlib.import_module(f'utils.{USED_DATA}').PIPELINE_DEFAULT_ARGS
REPORTING_DEFAULTS = importlib.import_module(f'utils.{USED_DATA}').REPORTING_DEFAULT_ARGS

# anomaly types specified for the considered data
try:
    ANOMALY_TYPES = importlib.import_module(f'utils.{USED_DATA}').ANOMALY_TYPES
except ImportError:
    warnings.warn('No anomaly types were defined for the considered data.')
    ANOMALY_TYPES = []


def hyper_to_path(*parameters):
    """Returns the path extension corresponding to the provided parameter list.
    """
    # empty string parameters are ignored (and `[a, b]` lists are turned to `[a-b]`)
    parameters = [str(p).replace(', ', '-') for p in parameters if p != '']
    return '_'.join(map(str, parameters))


def get_output_path(args, execution_step, output_details=None):
    """Returns the output path for the specified execution according to the command-line arguments.

    Args:
        args (argparse.Namespace): parsed command-line arguments.
        execution_step (str): execution step, as the name of its main python file (without extension).
        output_details (str|None): additional details in case there are several possible output paths for the step.

    Returns:
        str: the full output path.
    """
    if execution_step == REPORTING_STEP:
        return os.path.join(REPORTS_ROOT, get_args_string(args, 'reporting_methods'))
    path_extensions = dict()
    pipeline_steps = PIPELINE_STEPS.copy()
    if execution_step == EXPLANATION_STEP:
        # the index of the explanation step depends on the type of method and explained elements
        pipeline_steps.insert(get_explanation_step_index(args), 'train_explainer')
    step_index = pipeline_steps.index(execution_step)
    for i in range(step_index + 1):
        path_extensions[pipeline_steps[i]] = os.path.join(
            *[get_args_string(args, c) for c in CONFIG_KEYS[pipeline_steps[i]]]
        )
    if execution_step == 'make_datasets':
        return os.path.join(
            INTERIM_ROOT,
            path_extensions['make_datasets']
        )
    if execution_step == 'build_features':
        # we consider we want the output data path by default
        return os.path.join(
            PROCESSED_ROOT if output_details is None or output_details == 'data' else MODELS_ROOT,
            path_extensions['make_datasets'],
            path_extensions['build_features']
        )
    extensions_chain = [path_extensions[pipeline_steps[i]] for i in range(step_index + 1)]
    # return either the current step model's path (default) or the step comparison path
    comparison_path = os.path.join(
        MODELS_ROOT,
        *extensions_chain[:-1]
    )
    if output_details is None or output_details == 'model':
        return os.path.join(comparison_path, extensions_chain[-1])
    assert output_details == 'comparison', f'specify `comparison` for the comparison path of `{execution_step}`'
    return comparison_path


def get_explanation_step_index(args):
    """Returns the index of the explanation step in the ordered list of main pipeline steps.

    The explanation step can be run after:
    - feature extraction, for model-free methods explaining ground-truth labels.
    - outlier score derivation, for model-dependent methods explaining outlier scores around ground-truth labels.
    - final detection, for methods explaining the binary predictions of an AD model.

    Args:
        args (argparse.Namespace): parsed command-line arguments.

    Returns:
        int: the explanation step index.
    """
    # remove irrelevant arguments for explanation discovery explaining ground-truth labels
    mf_choices, md_choices = CHOICES['train_explainer']['model_free_explanation'], \
        CHOICES['train_explainer']['model_dependent_explanation']
    assert args.explanation_method in mf_choices + md_choices
    if args.explained_predictions == 'ground.truth':
        last_relevant_script = 'build_features' if args.explanation_method in mf_choices else 'train_scorer'
    else:
        last_relevant_script = 'train_detector'
    return PIPELINE_STEPS.index(last_relevant_script) + 1


def get_data_args(args):
    """Returns the relevant and ordered list of data argument values from `args`.

    Args:
        args (argparse.Namespace): parsed command-line arguments.

    Returns:
        list: relevant list of data argument values corresponding to `args`.
    """
    data_args = [args.data]
    if args.data == 'spark':
        data_args.append(args.app_id)
        if args.trace_types != '.':
            data_args.append(args.trace_types.replace('_contention', '').replace('_input', ''))
        if args.ignored_anomalies != 'none':
            data_args.append(f'{args.ignored_anomalies.replace(".only", "")}.ignored')
    for arg, arg_text in zip([args.n_starting_removed, args.n_ending_removed], ['first', 'last']):
        if arg != 0:
            data_args.append(f'{arg_text}.{arg}.removed')
    return [*data_args, args.pre_sampling_period]


def get_features_args(args):
    """Returns the relevant and ordered list of features building argument values from `args`.

    Args:
        args (argparse.Namespace): parsed command-line arguments.

    Returns:
        list: relevant list of features building argument values corresponding to `args`.
    """
    downsampling_args = []
    # only include downsampling arguments if downsampling is applied
    if args.data_sampling_period != args.pre_sampling_period:
        downsampling_args += [args.data_sampling_period, args.data_downsampling_position]
    if args.labels_sampling_period != args.pre_sampling_period:
        downsampling_args += [args.labels_sampling_period]
    alter_args = []
    if args.alter_bundles != '.':
        alter_args += [args.alter_bundles.replace('_bundles', ''), args.alter_bundle_idx]
    transform_args = [
        args.transform_chain
            .replace('scaling', 'scl')
            .replace('head', 'hd')
            .replace('online', 'oln')
            .replace('regular', 'reg')
            .replace('_', '-')
    ] if args.transform_chain != '.' else ['']
    # if the same transform step is repeated, the same arguments are used for all instances
    if 'head' in args.transform_chain:
        head_args = [args.head_size]
        if args.regular_pretraining_weight != -1:
            head_args += [args.regular_pretraining_weight]
        if 'online' in args.transform_chain:
            head_args = [
                args.online_window_type.replace('expanding', 'exp').replace('rolling', 'rol')
            ] + head_args
        transform_args += head_args
    if 'scaling' in args.transform_chain:
        transform_args.append(args.scaling_method)
        if args.scaling_method == 'minmax':
            transform_args.append(args.minmax_range)
        if 'regular_scaling' in args.transform_chain and args.reg_scaler_training != 'all.training':
            transform_args.append(args.reg_scaler_training)
    if 'pca' in args.transform_chain:
        transform_args += [args.pca_n_components, args.pca_kernel]
        if args.pca_training != 'all.training':
            transform_args.append(args.pca_training)
    if 'fa' in args.transform_chain:
        transform_args += [args.fa_n_components]
        if args.fa_training != 'all.training':
            transform_args.append(args.fa_training)
    return [*downsampling_args, *alter_args, *transform_args]


def get_modeling_task_args(args):
    """Returns the relevant and ordered list of modeling task argument values from `args`.

    Args:
        args (argparse.Namespace): parsed command-line arguments.

    Returns:
        list: relevant list of modeling task argument values corresponding to `args`.
    """
    modeling_data_args = [a for a, v in zip([args.modeling_n_periods, args.modeling_data_prop], [-1, 1]) if a != v]
    # if taking all data from a single period, then it is the largest one and therefore not random
    if len(modeling_data_args) > 0 and not (args.modeling_n_periods == args.modeling_data_prop == 1):
        modeling_data_args += [args.modeling_data_seed]
    splitting_args = [args.modeling_split.replace('.split', ''), args.modeling_split_seed]
    if args.modeling_split == 'stratified.split':
        splitting_args += [args.n_period_strata]
    splitting_args += [args.modeling_val_prop, args.modeling_test_prop]
    task_name, task_args = '', []
    if args.model_type in CHOICES['train_model']['forecasting']:
        task_name = 'fore'
        task_args += [args.n_back, args.n_forward]
    elif args.model_type in CHOICES['train_model']['reconstruction']:
        task_name = 'reco'
        task_args += [args.window_size, args.window_step]
    return modeling_data_args + splitting_args + [task_name] + task_args


def get_model_args(args):
    """Returns the relevant and ordered list of model argument values from `args`.

    Args:
        args (argparse.Namespace): parsed command-line arguments.

    Returns:
        list: relevant list of model argument values corresponding to `args`.
    """
    model_args = [args.model_type]
    if args.model_type == 'rnn':
        model_args += [
            args.rnn_unit_type, args.rnn_n_hidden_neurons,
            f'{args.rnn_dropout:.2f}', f'{args.rnn_rec_dropout:.2f}',
            args.rnn_optimizer, f'{args.rnn_learning_rate:.7f}',
            args.rnn_n_epochs, args.rnn_batch_size
        ]
    if args.model_type == 'ae':
        rec_unit_type, rec_dropout = [], []
        if args.ae_type == 'rec':
            rec_unit_type, rec_dropout = [args.ae_rec_unit_type], [args.ae_rec_dropout]
        model_args += [
            args.ae_type, *rec_unit_type,
            args.ae_enc_n_hidden_neurons, args.ae_dec_last_activation,
            args.ae_latent_dim, args.ae_dropout, *rec_dropout,
            args.ae_loss, args.ae_optimizer, f'{args.ae_learning_rate:.7f}',
            args.ae_n_epochs, args.ae_batch_size
        ]
    if args.model_type == 'bigan':
        enc_args, gen_args = [args.bigan_enc_type], [args.bigan_gen_type, args.bigan_gen_last_activation]
        dis_args = [args.bigan_dis_type]
        # encoder network arguments
        if args.bigan_enc_arch_idx != -1:
            # hardcoded architecture
            enc_args.append(args.bigan_enc_arch_idx)
        else:
            enc_dropout_args = [args.bigan_enc_dropout]
            if args.bigan_enc_type == 'rec':
                enc_args += [
                    args.bigan_enc_rec_n_hidden_neurons,
                    args.bigan_enc_rec_unit_type,
                ]
                enc_dropout_args.append(args.bigan_enc_rec_dropout)
            elif args.bigan_enc_type == 'conv':
                enc_args += [
                    args.bigan_enc_conv_n_filters
                ]
            enc_args += enc_dropout_args
        # generator network arguments
        if args.bigan_gen_arch_idx != -1:
            # hardcoded architecture
            gen_args.append(args.bigan_gen_arch_idx)
        else:
            gen_dropout_args = [args.bigan_gen_dropout]
            if args.bigan_gen_type == 'rec':
                gen_args += [
                    args.bigan_gen_rec_n_hidden_neurons,
                    args.bigan_gen_rec_unit_type,
                ]
                gen_dropout_args.append(args.bigan_gen_rec_dropout)
            elif args.bigan_gen_type == 'conv':
                gen_args += [
                    args.bigan_gen_conv_n_filters
                ]
            gen_args += gen_dropout_args
        # discriminator network arguments
        if args.bigan_dis_arch_idx != -1:
            # hardcoded architecture
            dis_args.append(args.bigan_dis_arch_idx)
        else:
            # x path
            dis_dropout_args = [args.bigan_dis_x_dropout]
            if args.bigan_dis_type == 'rec':
                dis_args += [
                    args.bigan_dis_x_rec_n_hidden_neurons,
                    args.bigan_dis_x_rec_unit_type,
                ]
                dis_dropout_args.append(args.bigan_dis_x_rec_dropout)
            elif args.bigan_dis_type == 'conv':
                dis_args += [
                    args.bigan_dis_x_conv_n_filters
                ]
            dis_args += dis_dropout_args
            # z path
            dis_args += [args.bigan_dis_z_n_hidden_neurons, args.bigan_dis_z_dropout]
        model_args += [
            args.bigan_latent_dim,
            *enc_args, *gen_args, *dis_args,
            f'{args.bigan_dis_threshold:.2f}',
            args.bigan_dis_optimizer, f'{args.bigan_dis_learning_rate:.7f}',
            args.bigan_enc_gen_optimizer, f'{args.bigan_enc_gen_learning_rate:.7f}',
            args.bigan_n_epochs, args.bigan_batch_size
        ]
    return model_args


def get_scoring_args(args):
    """Returns the relevant and ordered list of scoring argument values from `args`.

    Args:
        args (argparse.Namespace): parsed command-line arguments.

    Returns:
        list: relevant list of scoring argument values corresponding to `args`.
    """
    scoring_args = [args.scoring_method]
    if args.scoring_method in ['mse.dis', 'mse.ft']:
        scoring_args.append(args.mse_weight)
    return scoring_args


def get_ad_evaluation_args(args):
    """Returns the relevant and ordered list of AD evaluation argument values from `args`.

    Args:
        args (argparse.Namespace): parsed command-line arguments.

    Returns:
        list: relevant list of AD evaluation argument values corresponding to `args`.
    """
    ad_evaluation_args = [args.evaluation_type]
    if args.evaluation_type == 'range':
        ad_evaluation_args += [
            args.recall_alpha, args.recall_omega, args.recall_delta, args.recall_gamma,
            args.precision_omega, args.precision_delta, args.precision_gamma
        ]
    return ad_evaluation_args + [args.f_score_beta]


def get_ed_evaluation_args(args):
    """Returns the relevant and ordered list of ED evaluation argument values from `args`.

    Args:
        args (argparse.Namespace): parsed command-line arguments.

    Returns:
        list: relevant list of ED evaluation argument values corresponding to `args`.
    """
    ed_evaluation_args = [args.ed_eval_min_anomaly_length, args.ed1_consistency_n_disturbances]
    # min normal length, subsampling and accuracy-related arguments are only relevant for model-free methods
    if args.explanation_method in CHOICES['train_explainer']['model_free_explanation']:
        ed_evaluation_args += [
            args.mf_eval_min_normal_length,
            args.mf_ed1_consistency_sampled_prop,
            args.mf_ed1_accuracy_n_splits, args.mf_ed1_accuracy_test_prop
        ]
    # anomaly coverage is only relevant for model-dependent methods
    elif args.explanation_method in CHOICES['train_explainer']['model_dependent_explanation']:
        ed_evaluation_args += [
            args.md_eval_small_anomalies_expansion, args.md_eval_large_anomalies_coverage
        ]
    return ed_evaluation_args


def get_thresholding_args(args):
    """Returns the relevant and ordered list of thresholding argument values from `args`.

    Args:
        args (argparse.Namespace): parsed command-line arguments.

    Returns:
        list: relevant list of thresholding argument values corresponding to `args`.
    """
    thresholding_args = [args.thresholding_method]
    if args.thresholding_method in CHOICES['train_detector']['two_stat_ts_sel']:
        thresholding_args += [args.thresholding_factor, args.n_iterations, args.removal_factor]
    return thresholding_args


def get_best_thresholding_args(args):
    """Returns the provided `args` with the "best" thresholding parameters in terms
        of global F1-score on the test dataset.

    Args:
        args (argparse.Namespace): parsed command-line arguments.

    Returns:
        argparse.Namespace: updated args with best thresholding parameters.
    """
    thresholding_comparison_path = get_output_path(args, 'train_detector', 'comparison')
    full_thresholding_comparison_path = os.path.join(
        thresholding_comparison_path,
        f'{get_args_string(args, "ad_evaluation")}_detection_comparison.csv'
    )
    thresholding_comparison_df = pd.read_csv(full_thresholding_comparison_path, index_col=[0, 1]).astype(float)
    # for spark data, the "best" threshold is defined as maximizing the average F1-score across applications
    aggregation_str = 'app_avg' if args.data == 'spark' else 'global'
    best_thresholding_row = thresholding_comparison_df.loc[
        thresholding_comparison_df.index.get_level_values('granularity') == aggregation_str
    ].sort_values('TEST_GLOBAL_F1.0_SCORE', ascending=False).iloc[0]
    return get_new_thresholding_args_from_str(args, best_thresholding_row.name[0])


def get_new_thresholding_args_from_str(args, thresholding_str):
    """Returns the provided `args` with updated thresholding parameters from `thresholding_str`.

    `thresholding_str` must be consistent with the output of `get_args_string(args, "thresholding")`
    for a single combination of thresholding arguments.

    Args:
        args (argparse.Namespace): parsed command-line arguments.
        thresholding_str (str): thresholding parameters string of the form
            `{thresholding_method}_{thresholding_factor}_{n_iterations}_{removal_factor}`.

    Returns:
        argparse.Namespace: updated args with thresholding parameters from `thresholding_str`.
    """
    thresholding_args = thresholding_str.split('_')
    args_dict = vars(args)
    args_dict['thresholding_method'] = thresholding_args[0]
    args_dict['thresholding_factor'] = float(thresholding_args[1])
    args_dict['n_iterations'] = int(thresholding_args[2])
    args_dict['removal_factor'] = float(thresholding_args[3])
    return argparse.Namespace(**args_dict)


def get_explanation_args(args):
    """Returns the relevant and ordered list of explanation discovery argument values from `args`.

    Args:
        args (argparse.Namespace): parsed command-line arguments.

    Returns:
        list: relevant list of explanation discovery argument values corresponding to `args`.
    """
    explanation_args = [args.explanation_method]
    if args.explanation_method in CHOICES['train_explainer']['model_free_explanation']:
        if args.explanation_method == 'exstream':
            explanation_args += [args.exstream_fp_scaled_std_threshold]
        if args.explanation_method == 'macrobase':
            explanation_args += [
                args.macrobase_n_bins, args.macrobase_min_support, args.macrobase_min_risk_ratio
            ]
    if args.explanation_method == 'lime':
        explanation_args += [args.lime_n_features]
    return explanation_args


def get_reporting_methods_args(reporting_args):
    """Returns the relevant and ordered list of argument values for the methods compared
        in the reporting step from `reporting_args`.

    Args:
        reporting_args (argparse.Namespace): parsed reporting command-line arguments.

    Returns:
        list: relevant list of reporting methods argument values corresponding to `reporting_args`.
    """
    return [reporting_args.evaluation_step, reporting_args.compared_methods_id]


def get_reporting_performance_args(reporting_args, include_agg_method=True):
    """Returns the relevant and ordered list of argument values for the performance considered
        in the reporting step from `reporting_args`.

    Note: including the "aggregation method" to the returned arguments should only be specified
    if some results were actually aggregated, that is, if multiple argument values were provided
    for at least some of the compared methods for the evaluation step.

    Args:
        reporting_args (argparse.Namespace): parsed reporting command-line arguments.
        include_agg_method (bool): whether to include the aggregation method to the returned
            argument values.

    Returns:
        list: relevant list of reporting performance argument values corresponding to `reporting_args`.
    """
    reporting_args_dict, reporting_performance_args = vars(reporting_args), []
    if include_agg_method and reporting_args.report_type == 'table':
        reporting_performance_args += [reporting_args.aggregation_method]
    reporting_performance_args += [
        reporting_args_dict[f'{reporting_args.evaluation_step}_set_name'],
        reporting_args_dict[f'{reporting_args.evaluation_step}_metrics']
    ]
    if reporting_args.evaluation_step != 'modeling':
        reporting_performance_args += [
            reporting_args_dict[f'{reporting_args.evaluation_step}_granularity'],
            reporting_args_dict[f'{reporting_args.evaluation_step}_anomaly_types']
        ]
    return reporting_performance_args + [reporting_args.report_type]


# argument getter functions dictionary
ARGS_GETTER_DICT = {
    'data': get_data_args,
    'features': get_features_args,
    'modeling_task': get_modeling_task_args,
    'model': get_model_args,
    'scoring': get_scoring_args,
    'ad_evaluation': get_ad_evaluation_args,
    'ed_evaluation': get_ed_evaluation_args,
    'thresholding': get_thresholding_args,
    'explanation': get_explanation_args,
    'reporting_methods': get_reporting_methods_args,
    'reporting_performance': get_reporting_performance_args
}

# configuration keys corresponding to the main pipeline steps
CONFIG_KEYS = {
    'make_datasets': ['data'],
    'build_features': ['features'],
    # separate arguments of the modeling task from the ones of the model performing that task
    'train_model': ['modeling_task', 'model'],
    'train_scorer': ['scoring'],
    'train_detector': ['thresholding'],
    'train_explainer': ['explanation']
}


def get_args_string(args, args_getter_key, **kwargs):
    """Returns the arguments string for the provided command-line arguments and getter function.

    This string can be used as a prefix to make sure considered methods are comparable.

    Args:
        args (argparse.Namespace): parsed command-line arguments.
        args_getter_key (str): argument values list getter function.
            Must be a key of `ARGS_GETTER_DICT`.
        **kwargs: optional keyword arguments to pass to the arguments getter function.

    Returns:
        str: the arguments string, in the regular path-like format.
    """
    a_t = f'the provided getter function key must be in {list(ARGS_GETTER_DICT.keys())}'
    assert args_getter_key in ARGS_GETTER_DICT.keys(), a_t
    return hyper_to_path(*ARGS_GETTER_DICT[args_getter_key](args, **kwargs))


def get_modeling_task_and_classes(args):
    """Returns the type of modeling task and available model classes for the task,
        based on the command-line arguments.

    Elements are imported inside the function to solve cross-import issues.

    Args:
        args (argparse.Namespace): parsed command-line arguments.

    Returns:
        str, dict: the task type and corresponding model classes dictionary.
    """
    from modeling.forecasting.forecasters import forecasting_classes
    from modeling.reconstruction.reconstructors import reconstruction_classes
    if args.model_type in CHOICES['train_model']['forecasting']:
        return 'forecasting', forecasting_classes
    return 'reconstruction', reconstruction_classes


def is_percentage(x):
    """Argparse parsing function: returns `x` as a float if it is between 0 and 1.

    Args:
        x (str): the command-line argument to be checked.

    Returns:
        float: `x` if it is valid. Raises `ArgumentTypeError` otherwise.
    """
    x = float(x)
    if not (0 <= x <= 1):
        raise argparse.ArgumentTypeError('Argument has to be between 0 and 1')
    return x


def is_number(x):
    """Argparse parsing function: returns `x` as an int or float if it is a number.

    Args:
        x (str): the command-line argument to be checked.

    Returns:
        int|float: `x` if it is valid. Raises `ArgumentTypeError` otherwise.
    """
    x = float(x)
    if x.is_integer():
        return int(x)
    return x


def is_percentage_or_minus_1(x):
    """Argparse parsing function: returns `x` as an int if it is -1 or float if between 0 and 1.

    Args:
        x (str): the command-line argument to be checked.

    Returns:
        int|float: `x` if it is valid. Raises `ArgumentTypeError` otherwise.
    """
    x = float(x)
    if x == -1:
        return int(x)
    if 0 <= x <= 1:
        return x
    raise argparse.ArgumentTypeError('Argument has to be either -1 or between 0 and 1')


def is_percentage_or_int(x):
    """Argparse parsing function: returns `x` as an int if strictly positive int or float if between 0 and 1.

    Args:
        x (str): the command-line argument to be checked.

    Returns:
        int|float: `x` if it is valid. Raises `ArgumentTypeError` otherwise.
    """
    x = float(x)
    if x.is_integer() and x > 0:
        return int(x)
    if 0 <= x <= 1:
        return x
    raise argparse.ArgumentTypeError('Argument has to be either a strictly positive int or between 0 and 1')


# parsers for each callable script
parsers = dict()

# possible choices for categorical command-line arguments (common to all data)
CHOICES = {
    'make_datasets': {
        'data': ['spark']
    },
    'build_features': {
        'data_downsampling_position': ['first', 'middle', 'last'],
        'alter_bundles': ['.', 'spark_bundles'],
        'transform_chain': [
            f'{s}{d}' for s in [
                'regular_scaling', 'trace_scaling', 'head_scaling', 'head_online_scaling'
            ] for d in ['', '.pca', '.fa']
        ],
        'scaling_method': ['minmax', 'std', 'robust'],
        'kernel': ['linear', 'rbf'],
        'reg_transformer_training': ['all.training', 'largest.training'],
        # when using `head_online_scaling`, the size of the rolling window is fixed to `head_size`
        'online_window_type': ['expanding', 'rolling']
    },
    'train_model': {
        'forecasting': ['naive.forecasting', 'rnn'],
        'reconstruction': ['ae', 'bigan'],
        'modeling_split': ['random.split', 'stratified.split'],
        # model hyperparameters choices
        'activation': ['relu', 'selu', 'elu', 'sigmoid', 'linear'],
        'unit': ['rnn', 'lstm', 'gru'],
        'optimizer': ['sgd', 'adam', 'nadam', 'rmsprop', 'adadelta'],
        'ae_type': ['dense', 'rec'],
        'ae_dec_last_activation': ['linear', 'sigmoid', 'elu'],
        'ae_loss': ['mse', 'bce'],
        'enc_type': ['rec', 'conv'],
        'gen_type': ['rec', 'conv'],
        'dis_type': ['rec', 'conv'],
        'bigan_gen_last_activation': ['linear', 'sigmoid', 'tanh']
    },
    'train_scorer': {
        'forecasting_scoring': ['re', 'nll'],
        'reconstruction_scoring': ['mse', 'mse.dis', 'mse.ft'],
        # use point-based, custom range-based or default requirements evaluation
        'evaluation_type': ['point', 'range', *[f'ad{i}' for i in range(1, 5)]],
        # function choices for custom range-based evaluation
        # either default or normalized so that the size reward cannot exceed its equivalent under a flat bias
        'omega': ['default', 'flat.normalized'],
        'delta': ['flat', 'front', 'back'],
        # fully allow duplicates, fully penalize them or penalize them using an inverse polynomial penalty
        'gamma': ['dup', 'no.dup', 'inv.poly']
    },
    'train_detector': {
        # methods for selecting the outlier score threshold
        'two_stat_ts_sel': ['std', 'mad', 'iqr']
    },
    'train_explainer': {
        # explanation discovery methods (relying on a model or not)
        'model_free_explanation': ['exstream', 'macrobase'],
        'model_dependent_explanation': ['lime'],
        # whether to "explain" ground-truth labels or predictions from an AD method
        'explained_predictions': ['ground.truth', 'model'],
        # model-dependent evaluation (expansion and coverage policies of small and large anomalies, respectively)
        'md_eval_small_anomalies_expansion': ['none', 'before', 'after', 'both'],
        'md_eval_large_anomalies_coverage': ['all', 'center', 'end']
    },
    'run_pipeline': {
        'pipeline_type': ['ad', 'ed', 'ad.ed']
    },
    'report_results': {
        'evaluation_step': ['modeling', 'scoring', 'detection', 'explanation'],
        'report_type': ['table'],
        'aggregation_method': ['median'],
        'modeling_set_name': ['train', 'val', 'test'],
        'modeling_metrics': [
            'n_epochs', 'avg_epoch_time', 'tot_train_time', 'mse', 'mae', 'dis_loss', 'ft_loss'
        ],
        **{f'{step}_set_name': ['test'] for step in ['scoring', 'detection', 'explanation']},
        **{f'{step}_granularity': ['global'] for step in ['scoring', 'detection', 'explanation']},
        **{
            f'{step}_anomaly_types': ['global', 'all', 'avg', *[type_ for type_ in ANOMALY_TYPES]]
            for step in ['scoring', 'detection', 'explanation']
        },
        **{
            f'{step}_anomaly_avg_type': ['all', 'reported']
            for step in ['scoring', 'detection', 'explanation']
        },
        'scoring_metrics': ['auprc'],
        'detection_metrics': ['f_score', 'precision', 'recall'],
        'explanation_metrics': [
            'prop_covered', 'prop_explained', 'time',
            *[
                f'ed{i}_{m}' for i in [1, 2]
                for m in ['conciseness', 'consistency', 'precision', 'recall', 'f1_score']
            ]
        ]
    }
}
# add combined, higher-level, choices
CHOICES['train_model']['model'] = \
    CHOICES['train_model']['forecasting'] + \
    CHOICES['train_model']['reconstruction']
CHOICES['train_scorer']['scoring_method'] = \
    CHOICES['train_scorer']['forecasting_scoring'] + \
    CHOICES['train_scorer']['reconstruction_scoring']
CHOICES['train_explainer']['explanation_method'] = \
    CHOICES['train_explainer']['model_free_explanation'] + \
    CHOICES['train_explainer']['model_dependent_explanation']
# add data-specific possible choices
CHOICES = importlib.import_module(f'utils.{USED_DATA}').add_specific_choices(CHOICES)

# arguments for `make_datasets.py`
parsers['make_datasets'] = argparse.ArgumentParser(
    description='Make train and test sets from the input data', add_help=False
)
parsers['make_datasets'].add_argument(
    '--data', default=USED_DATA, choices=CHOICES['make_datasets']['data'],
    help='data to use as input to the pipeline'
)
for name in ['starting', 'ending']:
    parsers['make_datasets'].add_argument(
        f'--n-{name}-removed', default=DEFAULTS[f'n_{name}_removed'], type=int,
        help=f'number of {name} records to remove from the all the considered trace files'
    )
parsers['make_datasets'].add_argument(
    '--pre-sampling-period', default=DEFAULTS['pre_sampling_period'],
    help='optional pre-redefinition of the sampling period for more efficient storage'
)

# additional arguments for `build_features.py`
parsers['build_features'] = argparse.ArgumentParser(
    parents=[parsers['make_datasets']], description='Build features for the model inputs', add_help=False
)
parsers['build_features'].add_argument(
    '--data-sampling-period', default=DEFAULTS['data_sampling_period'],
    help='the data records will be downsampled to the provided period'
)
parsers['build_features'].add_argument(
    '--data-downsampling-position', default=DEFAULTS['data_downsampling_position'],
    choices=CHOICES['build_features']['data_downsampling_position'],
    help='whether to downsample records first, last or between alteration and transformation'
)
parsers['build_features'].add_argument(
    '--labels-sampling-period', default=DEFAULTS['labels_sampling_period'],
    help='the labels will be downsampled to the provided period'
)
parsers['build_features'].add_argument(
    '--alter-bundles', default=DEFAULTS['alter_bundles'],
    choices=CHOICES['build_features']['alter_bundles'],
    help='list of features alteration bundles we want to choose from (`.` for none)'
)
parsers['build_features'].add_argument(
    '--alter-bundle-idx', default=DEFAULTS['alter_bundle_idx'], type=int,
    help='alteration bundle index in the bundles list we chose'
)
parsers['build_features'].add_argument(
    '--transform-chain', default=DEFAULTS['transform_chain'],
    choices=CHOICES['build_features']['transform_chain'],
    help='features transformation chain, dot-separated (`.` for no transformation)'
)
parsers['build_features'].add_argument(
    '--head-size', default=DEFAULTS['head_size'], type=int,
    help='if relevant, number of records used at the beginning of each trace to train a transformer'
)
parsers['build_features'].add_argument(
    '--online-window-type', default=DEFAULTS['online_window_type'],
    choices=CHOICES['build_features']['online_window_type'],
    help='whether to use an expanding or rolling window when using head-online scaling'
)
parsers['build_features'].add_argument(
    '--regular-pretraining-weight', default=DEFAULTS['regular_pretraining_weight'],
    type=is_percentage_or_minus_1,
    help='if not -1, use regular pretraining for head/head-online transformers with this weight'
)
parsers['build_features'].add_argument(
    '--scaling-method', default=DEFAULTS['scaling_method'],
    choices=CHOICES['build_features']['scaling_method'],
    help='feature (re)scaling method'
)
parsers['build_features'].add_argument(
    '--minmax-range', default=DEFAULTS['minmax_range'], nargs='+', type=int,
    help='range of output features if using minmax scaling'
)
parsers['build_features'].add_argument(
    '--reg-scaler-training', default=DEFAULTS['reg_scaler_training'],
    choices=CHOICES['build_features']['reg_transformer_training'],
    help='whether to train the Regular Scaling model on all training traces or only the largest one'
)
parsers['build_features'].add_argument(
    '--pca-n-components', default=DEFAULTS['pca_n_components'], type=is_number,
    help='number of components or percentage of explained variance for PCA'
)
parsers['build_features'].add_argument(
    '--pca-kernel', default=DEFAULTS['pca_kernel'],
    choices=CHOICES['build_features']['kernel'],
    help='kernel for PCA'
)
parsers['build_features'].add_argument(
    '--pca-training', default=DEFAULTS['pca_training'],
    choices=CHOICES['build_features']['reg_transformer_training'],
    help='whether to train the PCA model on all training traces or only the largest one'
)
parsers['build_features'].add_argument(
    '--fa-n-components', default=DEFAULTS['fa_n_components'], type=int,
    help='number of components to keep after Factor Analysis (FA)'
)
parsers['build_features'].add_argument(
    '--fa-training', default=DEFAULTS['fa_training'],
    choices=CHOICES['build_features']['reg_transformer_training'],
    help='whether to train the Factor Analysis model on all training traces or only the largest one'
)

# additional arguments for `train_model.py`
parsers['train_model'] = argparse.ArgumentParser(
    parents=[parsers['build_features']], description='Train a model to perform a downstream task', add_help=False
)
# train/val/test datasets constitution for the modeling task
parsers['train_model'].add_argument(
    '--modeling-n-periods', default=DEFAULTS['modeling_n_periods'], type=int,
    help='number of periods to set as input normal data (first largest, then selected at random)'
)
parsers['train_model'].add_argument(
    '--modeling-data-prop', default=DEFAULTS['modeling_data_prop'], type=is_percentage,
    help='proportion of input normal data to consider when constituting the modeling datasets'
)
parsers['train_model'].add_argument(
    '--modeling-data-seed', default=DEFAULTS['modeling_data_seed'], type=int,
    help='random seed to use when selecting the subset of normal data used for modeling'
)
parsers['train_model'].add_argument(
    '--modeling-split', default=DEFAULTS['modeling_split'],
    choices=CHOICES['train_model']['modeling_split'],
    help='splitting strategy for constituting the modeling `train/val/test` sets'
)
parsers['train_model'].add_argument(
    '--modeling-split-seed', default=DEFAULTS['modeling_split_seed'], type=int,
    help='random seed to use when constituting the modeling `train/val/test` sets'
)
parsers['train_model'].add_argument(
    '--modeling-val-prop', default=DEFAULTS['modeling_val_prop'], type=is_percentage,
    help='proportion of `train` going to the modeling validation set'
)
parsers['train_model'].add_argument(
    '--modeling-test-prop', default=DEFAULTS['modeling_test_prop'], type=is_percentage,
    help='proportion of `train` going to the modeling test set'
)
parsers['train_model'].add_argument(
    '--n-period-strata', default=DEFAULTS['n_period_strata'], type=int,
    help='number of bins per period if using stratified modeling split'
)
parsers['train_model'].add_argument(
    '--model-type', default=DEFAULTS['model_type'],
    choices=CHOICES['train_model']['model'],
    help='type of model used to perform the downstream task'
)
# forecasting-specific arguments
parsers['train_model'].add_argument(
    '--n-back', default=DEFAULTS['n_back'], type=int,
    help='number of records to look back to perform forecasts'
)
parsers['train_model'].add_argument(
    '--n-forward', default=DEFAULTS['n_forward'], type=int, choices=[1],
    help='number of records to forecast forward (fixed to 1 for now)'
)
# reconstruction-specific arguments
parsers['train_model'].add_argument(
    '--window-size', default=DEFAULTS['window_size'], type=int,
    help='size of the windows to reconstruct in number of records'
)
parsers['train_model'].add_argument(
    '--window-step', default=DEFAULTS['window_step'], type=int,
    help='number of records between each extracted training window'
)
# whether to tune hyperparameters using keras tuner
parsers['train_model'].add_argument(
    '--keras-tuning', action='store_true', help='if provided, tune hyperparameters using keras tuner'
)
# FORECASTING MODELS #
# RNN hyperparameters
parsers['train_model'].add_argument(
    '--rnn-n-hidden-neurons', default=DEFAULTS['rnn_n_hidden_neurons'], nargs='+', type=int,
    help='number of neurons for each hidden layer of the RNN (before regression)'
)
parsers['train_model'].add_argument(
    '--rnn-unit-type', default=DEFAULTS['rnn_unit_type'],
    choices=CHOICES['train_model']['unit'],
    help='type of recurrent units used by the network'
)
parsers['train_model'].add_argument(
    '--rnn-dropout', default=DEFAULTS['rnn_dropout'], type=is_percentage,
    help='dropout rate for the feed-forward layers of the RNN'
)
parsers['train_model'].add_argument(
    '--rnn-rec-dropout', default=DEFAULTS['rnn_rec_dropout'], type=is_percentage,
    help='dropout rate for the recurrent layers of the RNN'
)
# RECONSTRUCTION MODELS #
# Autoencoder hyperparameters
parsers['train_model'].add_argument(
    '--ae-latent-dim', default=DEFAULTS['ae_latent_dim'], type=int,
    help='latent dimension for the autoencoder'
)
parsers['train_model'].add_argument(
    '--ae-type', default=DEFAULTS['ae_type'], choices=CHOICES['train_model']['ae_type'],
    help='type of autoencoder network'
)
parsers['train_model'].add_argument(
    '--ae-enc-n-hidden-neurons', default=DEFAULTS['ae_enc_n_hidden_neurons'], nargs='+', type=int,
    help='number of neurons for each hidden layer of the encoder (before the coding)'
)
parsers['train_model'].add_argument(
    '--ae-dec-last-activation', default=DEFAULTS['ae_dec_last_activation'],
    choices=CHOICES['train_model']['ae_dec_last_activation'],
    help='activation function of the last decoder layer'
)
parsers['train_model'].add_argument(
    '--ae-dropout', default=DEFAULTS['ae_dropout'], type=is_percentage,
    help='dropout rate for the feed-forward layers of the autoencoder'
)
parsers['train_model'].add_argument(
    '--ae-dense-layers-activation', default=DEFAULTS['ae_dense_layers_activation'],
    choices=CHOICES['train_model']['activation'],
    help='activation function for the intermediate layers of the autoencoder (if dense)'
)
parsers['train_model'].add_argument(
    '--ae-rec-unit-type', default=DEFAULTS['ae_rec_unit_type'],
    choices=CHOICES['train_model']['unit'],
    help='type of recurrent units used by the autoencoder (if recurrent)'
)
parsers['train_model'].add_argument(
    '--ae-rec-dropout', default=DEFAULTS['ae_rec_dropout'], type=is_percentage,
    help='dropout rate for the recurrent layers of the autoencoder'
)
parsers['train_model'].add_argument(
    '--ae-loss', default=DEFAULTS['ae_loss'],
    choices=CHOICES['train_model']['ae_loss'],
    help='loss function of the autoencoder'
)
# BiGAN hyperparameters
parsers['train_model'].add_argument(
    '--bigan-latent-dim', default=DEFAULTS['bigan_latent_dim'], type=int,
    help='latent dimension for the BiGAN network'
)
parsers['train_model'].add_argument(
    '--bigan-dis-threshold', default=DEFAULTS['bigan_dis_threshold'], type=float,
    help='only update D if its loss was above this value on the previous batch'
)
for name, ch in zip(
        ['encoder', 'generator', 'discriminator'],
        [
            CHOICES['train_model']['enc_type'],
            CHOICES['train_model']['gen_type'],
            CHOICES['train_model']['dis_type']
        ]
):
    abv = name[:3]
    parsers['train_model'].add_argument(
        f'--bigan-{abv}-type', default=DEFAULTS[f'bigan_{abv}_type'], choices=ch,
        help=f'type of {name} network'
    )
    parsers['train_model'].add_argument(
        f'--bigan-{abv}-arch-idx', default=DEFAULTS[f'bigan_{abv}_arch_idx'], type=int,
        help=f'if not -1, index of the architecture to use for this {name} type'
    )
    a_abv, d_abv = abv, abv
    if abv == 'dis':
        a_abv, d_abv, name = 'dis-x', 'dis_x', 'x path of D'
    parsers['train_model'].add_argument(
        f'--bigan-{a_abv}-rec-n-hidden-neurons',
        default=DEFAULTS[f'bigan_{d_abv}_rec_n_hidden_neurons'], nargs='+', type=int,
        help=f'number of neurons for each recurrent layer of {name}'
    )
    parsers['train_model'].add_argument(
        f'--bigan-{a_abv}-rec-unit-type',
        default=DEFAULTS[f'bigan_{d_abv}_rec_unit_type'], choices=CHOICES['train_model']['unit'],
        help=f'type of recurrent units used by the {name} (if recurrent)'
    )
    parsers['train_model'].add_argument(
        f'--bigan-{a_abv}-conv-n-filters',
        default=DEFAULTS[f'bigan_{d_abv}_conv_n_filters'], type=int,
        help=f'initial number of filters used by the {name} (if convolutional)'
    )
    parsers['train_model'].add_argument(
        f'--bigan-{a_abv}-dropout', default=DEFAULTS[f'bigan_{d_abv}_dropout'], type=is_percentage,
        help=f'dropout rate for the feed-forward layers of the {name}'
    )
    parsers['train_model'].add_argument(
        f'--bigan-{a_abv}-rec-dropout',
        default=DEFAULTS[f'bigan_{d_abv}_rec_dropout'], type=is_percentage,
        help=f'dropout rate for the recurrent layers of the {name}'
    )
parsers['train_model'].add_argument(
    '--bigan-gen-last-activation', default=DEFAULTS['bigan_gen_last_activation'],
    choices=CHOICES['train_model']['bigan_gen_last_activation'],
    help='activation function of the last generator layer'
)
# z path of the discriminator
parsers['train_model'].add_argument(
    '--bigan-dis-z-n-hidden-neurons',
    default=DEFAULTS['bigan_dis_z_n_hidden_neurons'], nargs='+', type=int,
    help='number of neurons for each layer of the z path of D'
)
parsers['train_model'].add_argument(
    '--bigan-dis-z-dropout', default=DEFAULTS['bigan_dis_z_dropout'], type=is_percentage,
    help='dropout rate for the feed-forward layers of the z path of D'
)

# SHARED WITH DIFFERENT PREFIXES #
for mt, mt_name in zip(['rnn', 'ae', 'bigan'], ['RNN', 'autoencoder', 'BiGAN']):
    if mt == 'bigan':
        for n, text in zip(['dis', 'enc_gen'], ['discriminator', 'encoder and generator']):
            dashed_n = n.replace('_', '-')
            parsers['train_model'].add_argument(
                f'--{mt}-{dashed_n}-optimizer', default=DEFAULTS[f'{mt}_{n}_optimizer'],
                choices=CHOICES['train_model']['optimizer'],
                help=f'optimization algorithm used for training the {text}'
            )
            parsers['train_model'].add_argument(
                f'--{mt}-{dashed_n}-learning-rate', default=DEFAULTS[f'{mt}_{n}_learning_rate'], type=float,
                help=f'learning rate used by the {text}'
            )
    else:
        parsers['train_model'].add_argument(
            f'--{mt}-optimizer', default=DEFAULTS[f'{mt}_optimizer'],
            choices=CHOICES['train_model']['optimizer'],
            help=f'optimization algorithm used for training the {mt_name} network'
        )
        parsers['train_model'].add_argument(
            f'--{mt}-learning-rate', default=DEFAULTS[f'{mt}_learning_rate'], type=float,
            help=f'learning rate used by the {mt_name} network'
        )
    parsers['train_model'].add_argument(
        f'--{mt}-n-epochs', default=DEFAULTS[f'{mt}_n_epochs'], type=int,
        help=f'maximum number of epochs to train the {mt_name} network for'
    )
    parsers['train_model'].add_argument(
        f'--{mt}-batch-size', default=DEFAULTS[f'{mt}_batch_size'], type=int,
        help=f'batch size used for training the {mt_name} network'
    )

# additional arguments for `train_scorer.py`
parsers['train_scorer'] = argparse.ArgumentParser(
    parents=[parsers['train_model']], description='Derive outlier scores from a trained model', add_help=False
)
parsers['train_scorer'].add_argument(
    '--scoring-method', default=DEFAULTS['scoring_method'],
    choices=CHOICES['train_scorer']['scoring_method'],
    help='outlier score derivation method'
)
parsers['train_scorer'].add_argument(
    '--mse-weight', default=DEFAULTS['mse_weight'], type=float,
    help='MSE weight if the outlier score is a convex combination of the MSE and another loss'
)
# parameters defining the Precision, Recall and F-score
parsers['train_scorer'].add_argument(
    '--evaluation-type', default=DEFAULTS['evaluation_type'],
    choices=CHOICES['train_scorer']['evaluation_type'],
    help='type of anomaly detection evaluation'
)
parsers['train_scorer'].add_argument(
    '--recall-alpha', default=DEFAULTS['recall_alpha'], type=is_percentage,
    help=f'existence reward factor for range-based Recall'
)
for metric in ['recall', 'precision']:
    metric_text = metric.capitalize()
    f_choices = [
        CHOICES['train_scorer']['omega'],
        CHOICES['train_scorer']['delta'],
        CHOICES['train_scorer']['gamma']
    ]
    for f_name, f_desc, choices in zip(['omega', 'delta', 'gamma'], ['size', 'bias', 'cardinality'], f_choices):
        parsers['train_scorer'].add_argument(
            f'--{metric}-{f_name}', default=DEFAULTS[f'{metric}_{f_name}'], choices=choices,
            help=f'{f_desc} function for range-based {metric_text}'
        )
parsers['train_scorer'].add_argument(
    '--f-score-beta', default=DEFAULTS['f_score_beta'], type=float,
    help='beta parameter for the F-Score (relative importance of Recall over Precision)'
)

# additional arguments for `train_detector.py`
parsers['train_detector'] = argparse.ArgumentParser(
    parents=[parsers['train_scorer']],
    description='Select the outlier score threshold and derive binary predictions', add_help=False
)
# threshold selection parameters
parsers['train_detector'].add_argument(
    '--thresholding-method', default=DEFAULTS['thresholding_method'], nargs='+',
    help='list of outlier score threshold selection methods to try'
)
parsers['train_detector'].add_argument(
    '--thresholding-factor', default=DEFAULTS['thresholding_factor'], nargs='+', type=float,
    help='`ts = stat_1 + thresholding_factor * stat_2` for simple statistical methods (list to try)'
)
parsers['train_detector'].add_argument(
    '--n-iterations', default=DEFAULTS['n_iterations'], nargs='+', type=int,
    help='number of thresholding iterations, each time removing the most obvious outliers (list to try)'
)
parsers['train_detector'].add_argument(
    '--removal-factor', default=DEFAULTS['removal_factor'], nargs='+', type=float,
    help='scores above `removal_factor * ts@{iteration_i}` will be removed for iteration i+1 (list to try)'
)

# additional arguments for `train_explainer.py`
parsers['train_explainer'] = argparse.ArgumentParser(
    parents=[parsers['train_detector']],
    description='Train an explanation discovery model to explain anomalies', add_help=False
)
parsers['train_explainer'].add_argument(
    '--explanation-method', default=DEFAULTS['explanation_method'],
    choices=CHOICES['train_explainer']['explanation_method'],
    help='explanation discovery method'
)
parsers['train_explainer'].add_argument(
    '--explained-predictions', default=DEFAULTS['explained_predictions'],
    choices=CHOICES['train_explainer']['explained_predictions'],
    help='positive predictions to explain (either ground-truth or outputs of an AD model)'
)
# common evaluation parameters
parsers['train_explainer'].add_argument(
    '--ed-eval-min-anomaly-length', default=DEFAULTS['ed_eval_min_anomaly_length'], type=int,
    help='minimum anomaly length for an instance to be considered in ED evaluation'
)
parsers['train_explainer'].add_argument(
    '--ed1-consistency-n-disturbances', default=DEFAULTS['ed1_consistency_n_disturbances'], type=int,
    help='number of disturbances performed when computing ED1 consistency'
)
# model-free evaluation parameters
parsers['train_explainer'].add_argument(
    '--mf-eval-min-normal-length', default=DEFAULTS['mf_eval_min_normal_length'], type=int,
    help='minimum normal data length for an instance to be considered in model-free evaluation'
)
parsers['train_explainer'].add_argument(
    '--mf-ed1-consistency-sampled-prop', default=DEFAULTS['mf_ed1_consistency_sampled_prop'], type=is_percentage,
    help='proportion of records sampled when computing ED1 consistency for model-free methods'
)
parsers['train_explainer'].add_argument(
    '--mf-ed1-accuracy-n-splits', default=DEFAULTS['mf_ed1_accuracy_n_splits'], type=int,
    help='number of random splits performed when computing ED1 accuracy for model-free methods'
)
parsers['train_explainer'].add_argument(
    '--mf-ed1-accuracy-test-prop', default=DEFAULTS['mf_ed1_accuracy_test_prop'], type=is_percentage,
    help='proportion of records used as test data when computing ED1 accuracy for model-free methods'
)
# model-dependent evaluation parameters
parsers['train_explainer'].add_argument(
    '--md-eval-small-anomalies-expansion', default=DEFAULTS['md_eval_small_anomalies_expansion'],
    choices=CHOICES['train_explainer']['md_eval_small_anomalies_expansion'],
    help='expansion policy of small anomalies when evaluating model-dependent ED methods'
)
parsers['train_explainer'].add_argument(
    '--md-eval-large-anomalies-coverage', default=DEFAULTS['md_eval_large_anomalies_coverage'],
    choices=CHOICES['train_explainer']['md_eval_large_anomalies_coverage'],
    help='coverage policy of large anomalies when evaluating model-dependent ED methods'
)
# EXstream hyperparameters
parsers['train_explainer'].add_argument(
    '--exstream-fp-scaled-std-threshold', default=DEFAULTS['exstream_fp_scaled_std_threshold'], type=float,
    help='scaled std threshold above which EXstream should define a feature as "false positive"'
)
# MacroBase hyperparameters
parsers['train_explainer'].add_argument(
    '--macrobase-n-bins', default=DEFAULTS['macrobase_n_bins'], type=int,
    help='number of bins to use for MacroBase\'s histogram-based discretization'
)
parsers['train_explainer'].add_argument(
    '--macrobase-min-support', default=DEFAULTS['macrobase_min_support'], type=is_percentage,
    help='outlier support threshold to use for MacroBase\'s multi-item itemset filtering'
)
parsers['train_explainer'].add_argument(
    '--macrobase-min-risk-ratio', default=DEFAULTS['macrobase_min_risk_ratio'], type=float,
    help='relative risk ratio threshold to use for MacroBase\'s itemset filtering'
)
# LIME hyperparameters
parsers['train_explainer'].add_argument(
    '--lime-n-features', default=DEFAULTS['lime_n_features'], type=int,
    help='number of features to report in the explanations derived by LIME'
)

# additional arguments for `run_pipeline.py`
parsers['run_pipeline'] = argparse.ArgumentParser(
    parents=[parsers['train_explainer']], description='Run a complete pipeline', add_help=False
)
parsers['run_pipeline'].add_argument(
    '--pipeline-type', default=DEFAULTS['pipeline_type'],
    choices=CHOICES['run_pipeline']['pipeline_type'],
    help='type of pipeline to run (AD only, ED only or AD + ED)'
)

# add data-specific arguments and `help` arguments back to parsers
add_specific_args = importlib.import_module(f'utils.{USED_DATA}').add_specific_args
for k in PIPELINE_CALLABLES:
    parsers = add_specific_args(parsers, k, PIPELINE_CALLABLES)
    parsers[k].add_argument(
        '-h', '--help', action='help',
        default=argparse.SUPPRESS, help='show this help message and exit'
    )

# additional arguments for `report_results.py`
parsers['report_results'] = argparse.ArgumentParser(description='Report performance of the specified method(s)')
# reporting methods arguments
parsers['report_results'].add_argument(
    '--evaluation-step', default=REPORTING_DEFAULTS['evaluation_step'],
    choices=CHOICES['report_results']['evaluation_step'],
    help='evaluation step to report performance for (modeling, scoring, detection or explanation)'
)
parsers['report_results'].add_argument(
    '--compared-methods-id', default=REPORTING_DEFAULTS['compared_methods_id'], type=int,
    help='unique identifier to manually update when running a new methods comparison to save'
)
# general reporting performance arguments
parsers['report_results'].add_argument(
    '--report-type', default=REPORTING_DEFAULTS['report_type'],
    choices=CHOICES['report_results']['report_type'], help='type of report to produce'
)
parsers['report_results'].add_argument(
    '--aggregation-method', default=REPORTING_DEFAULTS['aggregation_method'],
    choices=CHOICES['report_results']['aggregation_method'],
    help='method used to aggregate argument combinations in table reporting'
)
# step-specific reporting performance arguments
for step in CHOICES['report_results']['evaluation_step']:
    arg_names, nargs_list = ['set_name', 'metrics'], [dict(), {'nargs': '+'}]
    if step != 'modeling':
        arg_names += ['granularity', 'anomaly_types', 'anomaly_avg_type']
        nargs_list += [dict(), {'nargs': '+'}, dict()]
    for arg_name, nargs in zip(arg_names, nargs_list):
        parsers['report_results'].add_argument(
            f'--{step}-{arg_name.replace("_", "-")}', default=REPORTING_DEFAULTS[f'{step}_{arg_name}'],
            **nargs, choices=CHOICES['report_results'][f'{step}_{arg_name}'],
            help=f'{arg_name.replace("_", " ").capitalize()} to use in the {step} performance reporting'
        )


def get_command_line_string(args_namespace, script_name):
    """Returns the subset of `args_namespace` defined for `script_name` as a full command-line string."""
    return get_str_from_formatted_args_dict(get_script_args_as_formatted_dict(args_namespace, script_name))


def get_script_args_as_formatted_dict(args_ns, script_name):
    """Returns the subset of `args_namespace` defined for `script_name` as a formatted dictionary.

    Args:
        args_ns (argparse.Namespace): the larger set of arguments, as outputted by `parse_args`.
        script_name (str): we want to restrict the arguments to the ones defined for this script name.

    Returns:
        dict: the restricted arguments, as a formatted dictionary.
            `key`: argument name in command-line format. Example: --my-arg-name.
            `value`: argument value.
            Note: the arguments with action='store_true' are handled in the same way as if they were
            entered by a user: `True` = present, `False` = absent.
    """
    # only keep arguments defined for the script name
    args_dict = get_script_args_dict(vars(args_ns), script_name)
    # remove `False` arguments and empty out values of `True` arguments
    args_dict = get_handled_store_true(args_dict)
    # turn the dictionary keys into command-line argument names
    return {('--' + key_).replace('_', '-'): v for key_, v in args_dict.items()}


def get_script_args_dict(args_dict, script_name, remove_irrelevant=False):
    """Returns the subset of `args_dict` defined for `script_name`.

    Args:
        args_dict (dict): the larger set of parsed command-line arguments, as a dictionary.
        script_name (str): we want to restrict the arguments to the ones defined for this script name.
        remove_irrelevant (bool): whether to remove irrelevant arguments depending on some argument values.
            For now, only relevant for an explanation step explaining ground-truth labels.

    Returns:
        dict: the restricted arguments.
    """
    script_dict = {
        key_: v for key_, v in args_dict.items()
        if key_ in [e.dest for e in parsers[script_name]._actions if e.dest != 'help']
    }
    if not remove_irrelevant or script_name != 'train_explainer' or args_dict['explained_predictions'] == 'model':
        # return all argument values for the provided step
        return script_dict
    last_relevant_script = PIPELINE_STEPS[get_explanation_step_index(argparse.Namespace(**script_dict))-1]
    return get_dicts_difference(
        script_dict,
        get_dicts_difference(
            get_script_args_dict(script_dict, 'train_detector'),
            get_script_args_dict(script_dict, last_relevant_script)
        )
    )


def get_dicts_difference(dict_1, dict_2):
    """Returns `dict_1` without elements whose keys are `dict_2`"""
    return {key: dict_1[key] for key in set(dict_1) - set(dict_2)}


def get_handled_store_true(args_dict):
    """Returns the input arguments dictionary with handled arguments of type action='store_true'.

    Such arguments are either `True` or `False`, meaning they are either specified or not in the command-line call.
    We want to reproduce this behavior for the arguments dictionary.

    Args:
        args_dict (dict): original arguments dictionary.

    Returns:
        dict: the same dictionary with removed `False` arguments and emptied out `True` arguments.
    """
    # remove arguments whose values are `False`, since that means they were not specified
    args_dict = {key_: v for key_, v in args_dict.items() if not (type(v) == bool and not v)}
    # empty values of arguments whose values are `True`, since they should simply be mentioned without a value
    for key_, v in args_dict.items():
        if type(v) == bool and v:
            args_dict[key_] = ''
    return args_dict


def get_str_from_formatted_args_dict(args_dict):
    """Turns a formatted arguments dictionary into a command-line string, as if the arguments were entered by a user.

    Args:
        args_dict (dict): the arguments as a formatted dictionary.

    Returns:
        str: the corresponding command-line string.
    """
    args_str = str(args_dict)
    for c in '{}\',:[]':
        args_str = args_str.replace(c, '')
    return args_str
