"""Final detection module.

Selects an outlier score threshold to derive and evaluate final anomaly detection
on the pipeline's test set.

The threshold is selected based on the outlier score distribution of the modeling's
test samples (normal/unlabeled data).

To improve efficiency, the threshold selection method and parameters have to be provided
as lists of values to try (instead of single values).
"""
import os
import argparse
import importlib
import itertools

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from utils.common import (
    PIPELINE_TRAIN_NAME, PIPELINE_TEST_NAME, MODELING_TEST_NAME, parsers,
    get_output_path, get_args_string, get_modeling_task_and_classes
)
from data.helpers import load_mixed_formats, load_datasets_data
from modeling.data_splitters import get_splitter_classes
from modeling.forecasting.helpers import get_trimmed_periods
from metrics.evaluation import save_evaluation
from metrics.ad_evaluators import evaluation_classes
from detection.detector import Detector


if __name__ == '__main__':
    # parse and get command-line arguments
    args = parsers['train_detector'].parse_args()

    # set input and output paths
    DATA_INFO_PATH = get_output_path(args, 'make_datasets')
    DATA_INPUT_PATH = get_output_path(args, 'build_features', 'data')
    MODEL_INPUT_PATH = get_output_path(args, 'train_model')
    SCORER_INPUT_PATH = get_output_path(args, 'train_scorer')
    OUTPUT_PATH = get_output_path(args, 'train_detector', 'model')
    COMPARISON_PATH = get_output_path(args, 'train_detector', 'comparison')

    # load the periods records, labels and information used to derive and evaluate anomaly predictions
    thresholding_sets = [PIPELINE_TEST_NAME]
    thresholding_data = load_datasets_data(DATA_INPUT_PATH, DATA_INFO_PATH, thresholding_sets)

    # load the model and scoring classes for the relevant type of task
    task_type, model_classes = get_modeling_task_and_classes(args)
    a_t = 'the type of task must be either `forecasting` or `reconstruction`'
    assert task_type in ['forecasting', 'reconstruction'], a_t
    scoring_classes = importlib.import_module(f'scoring.{task_type}.{task_type}_scorers').scoring_classes

    # non-parametric models are simply initialized without loading anything
    if args.model_type == 'naive.forecasting':
        model = model_classes[args.model_type](args, '')
    # others are loaded from disk
    else:
        model = model_classes[args.model_type].from_file(args, MODEL_INPUT_PATH)

    # initialize the relevant scorer based on command-line arguments
    scorer = scoring_classes[args.scoring_method](args, model, OUTPUT_PATH)

    if task_type == 'forecasting':
        # adapt labels to the task cutting out the first `n_back` records of each period
        for set_name in thresholding_sets:
            thresholding_data[f'y_{set_name}'] = get_trimmed_periods(
                thresholding_data[f'y_{set_name}'], args.n_back
            )
        # keyword arguments for the data splitting function
        kwargs = {'n_back': args.n_back, 'n_forward': args.n_forward}
    else:
        kwargs = {'window_size': args.window_size, 'window_step': args.window_step}

    # get the relevant anomaly detection evaluator
    evaluator = evaluation_classes[args.evaluation_type](args)

    # load test samples of the modeling set
    print('loading training periods and information...', end=' ', flush=True)
    modeling_files = load_mixed_formats(
        [DATA_INPUT_PATH, DATA_INFO_PATH],
        [PIPELINE_TRAIN_NAME, f'{PIPELINE_TRAIN_NAME}_info'],
        ['numpy', 'pickle']
    )
    print('done.')
    print('recovering modeling test samples...', end=' ', flush=True)
    data_splitter = get_splitter_classes()[args.modeling_split](args)
    data = data_splitter.get_modeling_split(
        modeling_files[PIPELINE_TRAIN_NAME], modeling_files[f'{PIPELINE_TRAIN_NAME}_info'], **kwargs
    )
    modeling_test_data = {
        k.replace(f'_{MODELING_TEST_NAME}', ''): v for k, v in data.items() if MODELING_TEST_NAME in k
    }
    print('done.')

    # derive and evaluate final predictions for every thresholding combination
    looped_arg_names = ['thresholding_method', 'thresholding_factor', 'n_iterations', 'removal_factor']
    args_dict = vars(args)
    values_combinations = itertools.product(*[args_dict[looped] for looped in looped_arg_names])
    # remove variations of `removal_factor` if `n_iterations` is 1, as it does not apply
    values_combinations = set([tuple([*v[:-1], 1.0]) if v[2] == 1 else tuple(v) for v in values_combinations])
    for arg_values in values_combinations:
        # update command-line arguments and paths
        for i, arg_value in enumerate(arg_values):
            args_dict[looped_arg_names[i]] = arg_value
        args = argparse.Namespace(**args_dict)
        OUTPUT_PATH = get_output_path(args, 'train_detector', 'model')
        COMPARISON_PATH = get_output_path(args, 'train_detector', 'comparison')

        # build the anomaly detector corresponding to the scorer and command-line arguments
        detector = Detector(args, scorer, OUTPUT_PATH)

        # fit the outlier score threshold to the modeling's test samples
        detector.fit(**modeling_test_data)

        # derive the detector's record-wise predictions on the pipeline's test periods
        thresholding_processed = {k: v for k, v in thresholding_data.items() if k not in thresholding_sets}
        for set_name in thresholding_sets:
            thresholding_processed[f'{set_name}_preds'] = detector.predict(thresholding_data[set_name])

        # save final anomaly detection performance
        config_name = get_args_string(args, 'thresholding')
        evaluation_string = get_args_string(args, 'ad_evaluation')
        save_evaluation(
            'detection', thresholding_processed, evaluator, evaluation_string, config_name,
            COMPARISON_PATH, used_data=args.data, method_path=OUTPUT_PATH
        )
