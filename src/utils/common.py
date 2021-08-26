"""General utility module. Mainly handling paths and command-line arguments.
"""
import os
import argparse
import importlib

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

# a lot of default command-line arguments depend on the data we use
USED_DATA = os.getenv('USED_DATA').lower()
DEFAULTS = importlib.import_module(f'utils.{USED_DATA}').DEFAULT_ARGS


def hyper_to_path(*parameters):
    """Returns the path extension corresponding to the provided parameter list.
    """
    # empty string parameters are ignored (and `[a, b]` lists are turned to `[a-b]`)
    parameters = [str(p).replace(', ', '-') for p in parameters if p != '']
    return '_'.join(map(str, parameters))


def get_output_path(args, pipeline_step, output_details=None):
    """Returns the output path for the specified step of the AD pipeline according to the command-line arguments.

    Args:
        args (argparse.Namespace): parsed command-line arguments.
        pipeline_step (str): step of the AD pipeline, as the name of the main python file (without extension).
        output_details (str|None): additional details in case there are several possible output paths for the step.

    Returns:
        str: the full output path.
    """
    path_extensions = dict()
    step_index = PIPELINE_STEPS.index(pipeline_step)
    path_extensions['make_datasets'] = get_args_string(args, 'data')
    if step_index >= 1:
        path_extensions['build_features'] = get_args_string(args, 'features')
    if step_index >= 2:
        # separate arguments of the modeling task from the ones of the model performing that task
        path_extensions['train_model'] = os.path.join(
            get_args_string(args, 'modeling_task'), get_args_string(args, 'model')
        )
    if step_index >= 3:
        path_extensions['train_scorer'] = get_args_string(args, 'scoring')
    if step_index >= 4:
        path_extensions['train_detector'] = get_args_string(args, 'thresholding')
    if pipeline_step == 'make_datasets':
        return os.path.join(
            INTERIM_ROOT,
            path_extensions['make_datasets']
        )
    if pipeline_step == 'build_features':
        # we consider we want the output data path by default
        return os.path.join(
            PROCESSED_ROOT if output_details is None or output_details == 'data' else MODELS_ROOT,
            path_extensions['make_datasets'],
            path_extensions['build_features']
        )
    extensions_chain = [path_extensions[PIPELINE_STEPS[i]] for i in range(step_index + 1)]
    # return either the current step model's path (default) or the step comparison path
    comparison_path = os.path.join(
        MODELS_ROOT,
        *extensions_chain[:-1]
    )
    if output_details is None or output_details == 'model':
        return os.path.join(comparison_path, extensions_chain[-1])
    assert output_details == 'comparison', f'specify `comparison` for the comparison path of `{pipeline_step}`'
    return comparison_path


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
    downsampling_args = [args.sampling_period]
    # downsampling position only has an impact if downsampling is applied
    if args.sampling_period != args.pre_sampling_period:
        downsampling_args += [args.downsampling_position]
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
    splitting_args = [args.modeling_split.replace('.split', ''), args.modeling_split_seed]
    if args.modeling_split == 'stratified.split':
        splitting_args += [args.n_period_strata]
    splitting_args += [args.modeling_val_prop, args.modeling_test_prop]
    task_name, task_args = '', []
    if args.model_type in forecasting_choices:
        task_name = 'fore'
        task_args += [args.n_back, args.n_forward]
    elif args.model_type in reconstruction_choices:
        task_name = 'reco'
        task_args += [args.window_size, args.window_step]
    return splitting_args + [task_name] + task_args


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


def get_evaluation_args(args):
    """Returns the relevant and ordered list of evaluation argument values from `args`.

    Args:
        args (argparse.Namespace): parsed command-line arguments.

    Returns:
        list: relevant list of evaluation argument values corresponding to `args`.
    """
    evaluation_args = [args.evaluation_type]
    if args.evaluation_type == 'range':
        evaluation_args += [
            args.recall_alpha, args.recall_omega, args.recall_delta, args.recall_gamma,
            args.precision_omega, args.precision_delta, args.precision_gamma
        ]
    return evaluation_args + [args.f_score_beta]


def get_thresholding_args(args):
    """Returns the relevant and ordered list of thresholding argument values from `args`.

    Args:
        args (argparse.Namespace): parsed command-line arguments.

    Returns:
        list: relevant list of thresholding argument values corresponding to `args`.
    """
    thresholding_args = [args.thresholding_method]
    if args.thresholding_method in two_stat_ts_sel_choices:
        thresholding_args += [args.thresholding_factor, args.n_iterations, args.removal_factor]
    return thresholding_args


# argument getter functions dictionary
ARGS_GETTER_DICT = {
    'data': get_data_args,
    'features': get_features_args,
    'modeling_task': get_modeling_task_args,
    'model': get_model_args,
    'scoring': get_scoring_args,
    'evaluation': get_evaluation_args,
    'thresholding': get_thresholding_args
}


def get_args_string(args, args_getter_key):
    """Returns the arguments string for the provided command-line arguments and getter function.

    This string can be used as a prefix to make sure considered methods are comparable.

    Args:
        args (argparse.Namespace): parsed command-line arguments.
        args_getter_key (str): argument values list getter function.
            Must be a key of `ARGS_GETTER_DICT`.

    Returns:
        str: the arguments string, in the regular path-like format.
    """
    a_t = f'the provided getter function key must be in {list(ARGS_GETTER_DICT.keys())}'
    assert args_getter_key in ARGS_GETTER_DICT.keys(), a_t
    return hyper_to_path(*ARGS_GETTER_DICT[args_getter_key](args))


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
    if args.model_type in forecasting_choices:
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

# possible choices for categorical command-line arguments
# MAKE_DATASETS
data_choices = ['spark']

# BUILD_FEATURES
downsampling_position_choices = ['first', 'middle', 'last']
alter_bundles_choices = ['.', 'spark_bundles']
transform_choices = [
    f'{s}{d}' for s in [
        'regular_scaling', 'trace_scaling', 'head_scaling', 'head_online_scaling'
    ] for d in ['', '.pca', '.fa']
]
scaling_choices = ['minmax', 'std', 'robust']
kernel_choices = ['linear', 'rbf']
reg_transformer_training_choices = ['all.training', 'largest.training']
# when using `head_online_scaling`, the size of the rolling window is fixed to `head_size`
online_window_choices = ['expanding', 'rolling']

# TRAIN_MODEL
forecasting_choices = ['naive.forecasting', 'rnn']
reconstruction_choices = ['ae', 'bigan']
modeling_split_choices = ['random.split', 'stratified.split']
# model hyperparameters choices
model_choices = forecasting_choices + reconstruction_choices
activation_choices = ['relu', 'elu', 'sigmoid', 'linear']
unit_choices = ['rnn', 'lstm', 'gru']
opt_choices = ['sgd', 'adam', 'nadam', 'rmsprop', 'adadelta']
ae_type_choices = ['dense', 'rec']
enc_type_choices = gen_type_choices = dis_type_choices = ['rec', 'conv']
ae_dec_last_act_choices = ['linear', 'sigmoid', 'elu']
ae_loss_choices = ['mse', 'bce']
bigan_gen_last_act_choices = ['linear', 'sigmoid', 'tanh']

# TRAIN_SCORER
forecasting_scoring_choices = ['re', 'nll']
reconstruction_scoring_choices = ['mse', 'mse.dis', 'mse.ft']
scoring_choices = forecasting_scoring_choices + reconstruction_scoring_choices
# use point-based, custom range-based or default requirements evaluation
evaluation_choices = ['point', 'range', *[f'ad{i}' for i in range(1, 5)]]
# function choices for custom range-based evaluation
# either default or normalized so that the size reward cannot exceed its equivalent under a flat bias
omega_choices = ['default', 'flat.normalized']
delta_choices = ['flat', 'front', 'back']
# fully allow duplicates, fully penalize them or penalize them using an inverse polynomial penalty
gamma_choices = ['dup', 'no.dup', 'inv.poly']

# TRAIN_DETECTOR
# methods for selecting the outlier score threshold
two_stat_ts_sel_choices = ['std', 'mad', 'iqr']

# arguments for `make_datasets.py`
parsers['make_datasets'] = argparse.ArgumentParser(
    description='Make train and test sets from the input data', add_help=False
)
parsers['make_datasets'].add_argument(
    '--data', default=USED_DATA, choices=data_choices,
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
    '--sampling-period', default=DEFAULTS['sampling_period'],
    help='the records will be downsampled to the provided period'
)
parsers['build_features'].add_argument(
    '--downsampling-position', default=DEFAULTS['downsampling_position'], choices=downsampling_position_choices,
    help='whether to downsample periods first, last or between alteration and transformation'
)
parsers['build_features'].add_argument(
    '--alter-bundles', default=DEFAULTS['alter_bundles'], choices=alter_bundles_choices,
    help='list of features alteration bundles we want to choose from (`.` for none)'
)
parsers['build_features'].add_argument(
    '--alter-bundle-idx', default=DEFAULTS['alter_bundle_idx'], type=int,
    help='alteration bundle index in the bundles list we chose'
)
parsers['build_features'].add_argument(
    '--transform-chain', default=DEFAULTS['transform_chain'], choices=transform_choices,
    help='features transformation chain, dot-separated (`.` for no transformation)'
)
parsers['build_features'].add_argument(
    '--head-size', default=DEFAULTS['head_size'], type=int,
    help='if relevant, number of records used at the beginning of each trace to train a transformer'
)
parsers['build_features'].add_argument(
    '--online-window-type', default=DEFAULTS['online_window_type'], choices=online_window_choices,
    help='whether to use an expanding or rolling window when using head-online scaling'
)
parsers['build_features'].add_argument(
    '--regular-pretraining-weight', default=DEFAULTS['regular_pretraining_weight'],
    type=is_percentage_or_minus_1,
    help='if not -1, use regular pretraining for head/head-online transformers with this weight'
)
parsers['build_features'].add_argument(
    '--scaling-method', default=DEFAULTS['scaling_method'], choices=scaling_choices,
    help='feature (re)scaling method'
)
parsers['build_features'].add_argument(
    '--minmax-range', default=DEFAULTS['minmax_range'], nargs='+', type=int,
    help='range of output features if using minmax scaling'
)
parsers['build_features'].add_argument(
    '--reg-scaler-training', default=DEFAULTS['reg_scaler_training'], choices=reg_transformer_training_choices,
    help='whether to train the Regular Scaling model on all training traces or only the largest one'
)
parsers['build_features'].add_argument(
    '--pca-n-components', default=DEFAULTS['pca_n_components'], type=is_number,
    help='number of components or percentage of explained variance for PCA'
)
parsers['build_features'].add_argument(
    '--pca-kernel', default=DEFAULTS['pca_kernel'], choices=kernel_choices,
    help='kernel for PCA'
)
parsers['build_features'].add_argument(
    '--pca-training', default=DEFAULTS['pca_training'], choices=reg_transformer_training_choices,
    help='whether to train the PCA model on all training traces or only the largest one'
)
parsers['build_features'].add_argument(
    '--fa-n-components', default=DEFAULTS['fa_n_components'], type=int,
    help='number of components to keep after Factor Analysis (FA)'
)
parsers['build_features'].add_argument(
    '--fa-training', default=DEFAULTS['fa_training'], choices=reg_transformer_training_choices,
    help='whether to train the Factor Analysis model on all training traces or only the largest one'
)

# additional arguments for `train_model.py`
parsers['train_model'] = argparse.ArgumentParser(
    parents=[parsers['build_features']], description='Train a model to perform a downstream task', add_help=False
)
# train/val/test datasets constitution for the modeling task
parsers['train_model'].add_argument(
    '--modeling-split', default=DEFAULTS['modeling_split'], choices=modeling_split_choices,
    help='splitting strategy for constituting the modeling `train/val/test` sets'
)
parsers['train_model'].add_argument(
    '--modeling-split-seed', default=DEFAULTS['modeling_split_seed'], type=int,
    help='random seed used when constituting the modeling `train/val/test` sets'
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
    '--model-type', default=DEFAULTS['model_type'], choices=model_choices,
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
    '--rnn-unit-type', default=DEFAULTS['rnn_unit_type'], choices=unit_choices,
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
    '--ae-type', default=DEFAULTS['ae_type'], choices=ae_type_choices,
    help='type of autoencoder network'
)
parsers['train_model'].add_argument(
    '--ae-enc-n-hidden-neurons', default=DEFAULTS['ae_enc_n_hidden_neurons'], nargs='+', type=int,
    help='number of neurons for each hidden layer of the encoder (before the coding)'
)
parsers['train_model'].add_argument(
    '--ae-dec-last-activation', default=DEFAULTS['ae_dec_last_activation'],
    choices=ae_dec_last_act_choices, help='activation function of the last decoder layer'
)
parsers['train_model'].add_argument(
    '--ae-dropout', default=DEFAULTS['ae_dropout'], type=is_percentage,
    help='dropout rate for the feed-forward layers of the autoencoder'
)
parsers['train_model'].add_argument(
    '--ae-dense-layers-activation', default=DEFAULTS['ae_dense_layers_activation'], choices=activation_choices,
    help='activation function for the intermediate layers of the autoencoder (if dense)'
)
parsers['train_model'].add_argument(
    '--ae-rec-unit-type', default=DEFAULTS['ae_rec_unit_type'], choices=unit_choices,
    help='type of recurrent units used by the autoencoder (if recurrent)'
)
parsers['train_model'].add_argument(
    '--ae-rec-dropout', default=DEFAULTS['ae_rec_dropout'], type=is_percentage,
    help='dropout rate for the recurrent layers of the autoencoder'
)
parsers['train_model'].add_argument(
    '--ae-loss', default=DEFAULTS['ae_loss'], choices=ae_loss_choices,
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
        [enc_type_choices, gen_type_choices, dis_type_choices]
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
        default=DEFAULTS[f'bigan_{d_abv}_rec_unit_type'], choices=unit_choices,
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
    choices=bigan_gen_last_act_choices, help='activation function of the last generator layer'
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
                f'--{mt}-{dashed_n}-optimizer', default=DEFAULTS[f'{mt}_{n}_optimizer'], choices=opt_choices,
                help=f'optimization algorithm used for training the {text}'
            )
            parsers['train_model'].add_argument(
                f'--{mt}-{dashed_n}-learning-rate', default=DEFAULTS[f'{mt}_{n}_learning_rate'], type=float,
                help=f'learning rate used by the {text}'
            )
    else:
        parsers['train_model'].add_argument(
            f'--{mt}-optimizer', default=DEFAULTS[f'{mt}_optimizer'], choices=opt_choices,
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
    '--scoring-method', default=DEFAULTS['scoring_method'], choices=scoring_choices,
    help='outlier score derivation method'
)
parsers['train_scorer'].add_argument(
    '--mse-weight', default=DEFAULTS['mse_weight'], type=float,
    help='MSE weight if the outlier score is a convex combination of the MSE and another loss'
)
# parameters defining the Precision, Recall and F-score
parsers['train_scorer'].add_argument(
    '--evaluation-type', default=DEFAULTS['evaluation_type'], choices=evaluation_choices,
    help='type of anomaly detection evaluation'
)
parsers['train_scorer'].add_argument(
    '--recall-alpha', default=DEFAULTS['recall_alpha'], type=is_percentage,
    help=f'existence reward factor for range-based Recall'
)
for metric in ['recall', 'precision']:
    metric_text = metric.capitalize()
    f_choices = [omega_choices, delta_choices, gamma_choices]
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
# additional arguments for `run_pipeline.py`
parsers['run_pipeline'] = argparse.ArgumentParser(
    parents=[parsers['train_detector']], description='Run a complete anomaly detection pipeline',
    add_help=False
)

# add data-specific arguments
add_specific_args = importlib.import_module(f'utils.{os.getenv("USED_DATA").lower()}').add_specific_args
EXTENDED_PIPELINE_STEPS = PIPELINE_STEPS + ['run_pipeline']
for k in EXTENDED_PIPELINE_STEPS:
    parsers = add_specific_args(parsers, k, EXTENDED_PIPELINE_STEPS)
# add back `help` arguments to parsers
for key in EXTENDED_PIPELINE_STEPS:
    parsers[key].add_argument(
        '-h', '--help', action='help', default=argparse.SUPPRESS, help='show this help message and exit'
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
    args_dict = {key_: v for key_, v in vars(args_ns).items() if key_ in [
        e.dest for e in parsers[script_name]._actions if e.dest != 'help'
    ]}
    # remove `False` arguments and empty out values of `True` arguments
    args_dict = get_handled_store_true(args_dict)
    # turn the dictionary keys into command-line argument names
    return {('--' + key_).replace('_', '-'): v for key_, v in args_dict.items()}


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
