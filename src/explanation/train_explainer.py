"""Explanation discovery module.
"""
import os
import importlib

import numpy as np

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from utils.common import (
    PIPELINE_TRAIN_NAME, PIPELINE_TEST_NAME, MODELING_TEST_NAME, ANOMALY_TYPES, CHOICES,
    parsers, get_output_path, get_modeling_task_and_classes, get_args_string, get_best_thresholding_args
)
from data.helpers import load_datasets_data, load_mixed_formats
from modeling.data_splitters import get_splitter_classes
from modeling.forecasting.helpers import get_trimmed_periods
from detection.detector import Detector
from explanation.explainers import get_explanation_classes
from metrics.evaluation import save_evaluation
from metrics.ed_evaluators import evaluation_classes


if __name__ == '__main__':
    # parse and get command line arguments
    args = parsers['train_explainer'].parse_args()

    # set input and output paths
    DATA_INFO_PATH = get_output_path(args, 'make_datasets')
    DATA_INPUT_PATH = get_output_path(args, 'build_features', 'data')

    # load the periods records, labels and information used to train and evaluate explanation discovery
    explanation_sets = [PIPELINE_TEST_NAME]
    explanation_data = load_datasets_data(DATA_INPUT_PATH, DATA_INFO_PATH, explanation_sets)

    # remove "unknown" anomalies from consideration if explaining ground-truth spark anomalies
    if args.data == 'spark' and args.explained_predictions == 'ground.truth':
        unknown_class = ANOMALY_TYPES.index('unknown') + 1
        for set_name in explanation_sets:
            for i in range(len(explanation_data[f'y_{set_name}'])):
                unknown_mask = explanation_data[f'y_{set_name}'][i] == unknown_class
                explanation_data[f'y_{set_name}'][i][unknown_mask] = 0

    # some ED methods might require additional arguments, as well as a fitting procedure
    explainer_args, training_samples = dict(), None

    # get ED method type
    if args.explanation_method in CHOICES['train_explainer']['model_dependent_explanation']:
        method_type = 'model_dependent'
    elif args.explanation_method in CHOICES['train_explainer']['model_free_explanation']:
        method_type = 'model_free'
    else:
        raise ValueError('the provided ED method must be either "model-free" or "model-dependent"')

    # load scorer and/or detector predictions if relevant
    if method_type == 'model_dependent' or args.explained_predictions == 'model':
        # modeling and scoring output paths
        MODEL_INPUT_PATH = get_output_path(args, 'train_model')
        SCORER_INPUT_PATH = get_output_path(args, 'train_scorer')
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
        scorer = scoring_classes[args.scoring_method](args, model, '')
        if task_type == 'forecasting':
            # adapt labels to the task cutting out the first `n_back` records of each period
            for set_name in explanation_sets:
                explanation_data[f'y_{set_name}'] = get_trimmed_periods(
                    explanation_data[f'y_{set_name}'], args.n_back
                )
            # keyword arguments for the data splitting function
            kwargs = {'n_back': args.n_back, 'n_forward': args.n_forward}
        else:
            kwargs = {'window_size': args.window_size, 'window_step': args.window_step}

        # model-dependent ED methods
        if method_type == 'model_dependent':
            # add "AD model" argument to the explainer and set it to the scorer
            explainer_args['ad_model'] = scorer
            if args.explanation_method == 'lime':
                # fit the method to the test samples of the modeling set
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
                if 'y' in modeling_test_data:
                    # add a dimension if labels are single-record targets
                    if len(modeling_test_data['y'].shape) == 2:
                        n_samples, n_features = modeling_test_data['y'].shape
                        modeling_test_data['y'] = modeling_test_data['y'].reshape((n_samples, 1, n_features))
                    # samples are defined as concatenations of input and target records
                    training_samples = np.array([
                        np.concatenate([X, y]) for X, y in zip(modeling_test_data['X'], modeling_test_data['y'])
                    ])
                else:
                    training_samples = modeling_test_data['X']

        # ED methods working on model predictions need the predictions of the detector
        if args.explained_predictions == 'model':
            # if thresholding parameters provided as a list, pick the "best" ones for the method
            if type(args.thresholding_method) == list:
                args = get_best_thresholding_args(args)
            # load the final detector with relevant threshold and replace labels with model predictions
            THRESHOLD_PATH = get_output_path(args, 'train_detector')
            detector = Detector.from_file(args, scorer, THRESHOLD_PATH)
            for set_name in explanation_sets:
                explanation_data[f'y_{set_name}'] = detector.predict(explanation_data[set_name])
                # prepend negative predictions for the first records if forecasting-based detector
                if args.model_type in CHOICES['train_model']['forecasting']:
                    for i in range(len(explanation_data[f'y_{set_name}'])):
                        explanation_data[f'y_{set_name}'][i] = np.concatenate([
                            np.zeros(shape=(args.n_back,), dtype=int),
                            explanation_data[f'y_{set_name}'][i]
                        ])

    # set output and comparison paths at the end so that they use potentially updated args
    OUTPUT_PATH = get_output_path(args, 'train_explainer', 'model')
    COMPARISON_PATH = get_output_path(args, 'train_explainer', 'comparison')

    # initialize the relevant explainer based on command-line and method-specific arguments
    explainer = get_explanation_classes()[args.explanation_method](args, OUTPUT_PATH, **explainer_args)

    # fit explainer to training samples if they were set
    if training_samples is not None:
        explainer.fit(training_samples)

    # save explanation discovery performance
    config_name = get_args_string(args, 'explanation')
    evaluation_string = get_args_string(args, 'ed_evaluation')
    evaluator = evaluation_classes[method_type](args, explainer)
    save_evaluation(
        'explanation', explanation_data, evaluator, evaluation_string, config_name, COMPARISON_PATH,
        used_data=args.data, method_path=OUTPUT_PATH,
        ignore_anomaly_types=(args.explained_predictions == 'model')
    )
