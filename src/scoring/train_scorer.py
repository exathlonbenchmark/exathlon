"""Outlier score assignment module.

From the provided model, derives an outlier score assignment method and evaluates it
on the pipeline's test set.
"""
import os
import importlib

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from utils.common import (
    PIPELINE_TEST_NAME, parsers, get_output_path, get_args_string, get_modeling_task_and_classes
)
from data.helpers import load_datasets_data
from modeling.forecasting.helpers import get_trimmed_periods
from metrics.evaluation import save_evaluation
from metrics.ad_evaluators import evaluation_classes


if __name__ == '__main__':
    # parse and get command-line arguments
    args = parsers['train_scorer'].parse_args()

    # set input and output paths
    DATA_INFO_PATH = get_output_path(args, 'make_datasets')
    DATA_INPUT_PATH = get_output_path(args, 'build_features', 'data')
    MODEL_INPUT_PATH = get_output_path(args, 'train_model')
    OUTPUT_PATH = get_output_path(args, 'train_scorer', 'model')
    COMPARISON_PATH = get_output_path(args, 'train_scorer', 'comparison')

    # load the periods records, labels and information used to evaluate the outlier scores
    scoring_sets = [PIPELINE_TEST_NAME]
    scoring_data = load_datasets_data(DATA_INPUT_PATH, DATA_INFO_PATH, scoring_sets)

    # load the model and scoring classes for the relevant type of task
    task_type, model_classes = get_modeling_task_and_classes(args)
    scoring_classes = importlib.import_module(f'scoring.{task_type}.{task_type}_scorers').scoring_classes

    if task_type == 'forecasting':
        # adapt labels to the task cutting out the first `n_back` records of each period
        for set_name in scoring_sets:
            scoring_data[f'y_{set_name}'] = get_trimmed_periods(
                scoring_data[f'y_{set_name}'], args.n_back
            )

    # non-parametric models are simply initialized without loading anything
    if args.model_type == 'naive.forecasting':
        model = model_classes[args.model_type](args, '')
    # others are loaded from disk
    else:
        model = model_classes[args.model_type].from_file(args, MODEL_INPUT_PATH)

    # initialize the relevant scorer based on command-line arguments
    scorer = scoring_classes[args.scoring_method](args, model, OUTPUT_PATH)

    # scoring methods are directly used here
    for set_name in scoring_sets:
        scoring_data[f'{set_name}_scores'] = scorer.score(scoring_data[set_name])
        scoring_data.pop(set_name)

    # save outlier score derivation performance
    config_name, evaluation_string = get_args_string(args, 'scoring'), get_args_string(args, 'ad_evaluation')
    evaluator = evaluation_classes[args.evaluation_type](args)
    save_evaluation(
        'scoring', scoring_data, evaluator, evaluation_string, config_name,
        COMPARISON_PATH, used_data=args.data, method_path=OUTPUT_PATH
    )
