"""Normality modeling module.

Trains a model to perform a task capturing the behavior of the pipeline's train set.
"""
import os
import time

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from utils.common import PIPELINE_TRAIN_NAME, MODELING_TRAIN_NAME, MODELING_VAL_NAME, parsers, get_output_path
from utils.common import get_modeling_task_and_classes, get_args_string
from data.helpers import load_mixed_formats
from modeling.data_splitters import get_splitter_classes
from modeling.forecasting.evaluation import save_forecasting_evaluation
from modeling.reconstruction.evaluation import save_reconstruction_evaluation


if __name__ == '__main__':
    # parse and get command-line arguments
    args = parsers['train_model'].parse_args()

    # set input and output paths
    INFO_PATH = get_output_path(args, 'make_datasets')
    INPUT_PATH = get_output_path(args, 'build_features', 'data')
    OUTPUT_PATH = get_output_path(args, 'train_model', 'model')
    COMPARISON_PATH = get_output_path(args, 'train_model', 'comparison')

    # load training periods and information
    files = load_mixed_formats(
        [INPUT_PATH, INFO_PATH],
        [PIPELINE_TRAIN_NAME, f'{PIPELINE_TRAIN_NAME}_info'],
        ['numpy', 'pickle']
    )

    # set data splitter for constituting the train/val/test sets of the modeling task
    data_splitter = get_splitter_classes()[args.modeling_split](args, output_path=OUTPUT_PATH)

    # modeling task and available model classes to perform it
    task_type, task_classes = get_modeling_task_and_classes(args)

    # task-specific elements
    a_t = 'only forecasting and reconstruction tasks are supported'
    assert task_type in ['forecasting', 'reconstruction'], a_t
    if task_type == 'forecasting':
        model_args = {'n_back': args.n_back, 'n_forward': args.n_forward}
        data_items = ['X', 'y']
        evaluation_saving_f = save_forecasting_evaluation
    else:
        model_args = {'window_size': args.window_size, 'window_step': args.window_step}
        data_items = ['X']
        evaluation_saving_f = save_reconstruction_evaluation

    # constitute the train/val/test sets of the modeling task
    data = data_splitter.get_modeling_split(
        files[PIPELINE_TRAIN_NAME], files[f'{PIPELINE_TRAIN_NAME}_info'], **model_args
    )

    # define model depending on the model type
    model = task_classes[args.model_type](args, OUTPUT_PATH)
    # either tune hyperparameters or train using full command arguments
    if args.keras_tuning:
        model.tune_hp(data)
    else:
        # fit/validate model on modeling training and validation sets
        model.fit(
            *[data[f'{i}_{n}'] for n in [MODELING_TRAIN_NAME, MODELING_VAL_NAME] for i in data_items]
        )
        # save model's training, validation and test performance for comparison
        config_name = get_args_string(args, 'model') + f'_{time.strftime("run.%Y.%m.%d.%H.%M.%S")}'
        evaluation_saving_f(
            data, model, get_args_string(args, 'modeling_task'), config_name, COMPARISON_PATH
        )
