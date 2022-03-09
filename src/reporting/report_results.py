"""Results reporting module.

Reports and optionally saves the performance of the specified arguments combination(s), assuming
all have already been run using the pipeline.
"""
import os
import copy
import time
import argparse
import warnings

import pandas as pd

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from utils.common import parsers, ANOMALY_TYPES, CONFIG_KEYS, get_output_path, get_args_string, get_script_args_dict
from reporting.helpers import check_reporting_args, get_args_combinations, get_column_names, add_avg_columns
from reporting.methods import COMPARED_ARGS, COMPARED_METHOD_TITLES


# dictionary associating evaluation step keys to their relevant keys for retrieving method arguments
CORRESPONDENCE_DICT = {
    # modeling evaluation step
    'modeling': {
        # main script name of the corresponding pipeline step
        'pipeline_step': 'train_model',
        # key to obtain the evaluation parameters (comparison spreadsheet name)
        'evaluation_key': 'modeling_task',
        # key to obtain the method's configuration (index in the comparison spreadsheet)
        'config_key': CONFIG_KEYS['train_model'][1]
    },
    'scoring': {
        'pipeline_step': 'train_scorer', 'evaluation_key': 'ad_evaluation',
        'config_key': CONFIG_KEYS['train_scorer'][0]
    },
    'detection': {
        'pipeline_step': 'train_detector', 'evaluation_key': 'ad_evaluation',
        'config_key': CONFIG_KEYS['train_detector'][0]
    },
    'explanation': {
        'pipeline_step': 'train_explainer', 'evaluation_key': 'ed_evaluation',
        'config_key': CONFIG_KEYS['train_explainer'][0]
    }
}


def report_results(reporting_args, compared_args, method_titles, output_path=None):
    """Returns and optionally saves a report comparing the results obtained for `compared_args`.

    Args:
        reporting_args (argparse.Namespace): parsed command-line reporting arguments.
        compared_args (list): list of pipeline argument dictionaries for the methods to compare.
        method_titles (list): method titles to use for each of the compared arguments.
        output_path (str|None): optional output path to save the report to.

    Returns:
        pd.DataFrame|None: the produced report if relevant, according to the report type.
    """
    # further check the provided reporting arguments
    check_reporting_args(reporting_args, ANOMALY_TYPES)

    # get pipeline step, evaluation and configuration keys from the evaluation step
    evaluation_step = reporting_args.evaluation_step
    keys = {
        k.split('_')[0]: CORRESPONDENCE_DICT[evaluation_step][k]
        for k in ['pipeline_step', 'evaluation_key', 'config_key']
    }
    # get relevant performance specifications for the evaluation step
    reporting_args_dict, perf_spec_names = copy.deepcopy(vars(reporting_args)), ['set_name', 'metrics']
    if evaluation_step != 'modeling':
        perf_spec_names += ['granularity', 'anomaly_types', 'anomaly_avg_type']
    perf_specs = {n: reporting_args_dict[f'{evaluation_step}_{n}'] for n in perf_spec_names}

    # only consider compared arguments that are relevant to the pipeline step
    compared_step_args = [
        get_script_args_dict(args_dict, keys['pipeline'], remove_irrelevant=True)
        for args_dict in compared_args
    ]

    # if "all" was specified in the anomaly types, replace it with all explicit anomaly types
    try:
        # inclusive start and exclusive end of explicit anomaly types to add
        start_idx = perf_specs['anomaly_types'].index('all')
        end_idx = start_idx + len(ANOMALY_TYPES)
        # elements occurring after the "all" key
        after_all = perf_specs['anomaly_types'][start_idx+1:]
        # replace "all" with all the anomaly types
        perf_specs['anomaly_types'][start_idx:end_idx] = ANOMALY_TYPES
        # add back what was after the "all" key
        perf_specs['anomaly_types'] += after_all
    except ValueError:
        pass

    # patterns of argument names for which "single values" are already lists
    list_args_patterns = ['n_hidden', 'minmax_range']

    # loop through each compared arguments/method
    report, multiple_args_combinations = None, False
    for i, args_space_dict in enumerate(compared_step_args):
        # DataFrame gathering the results of all argument combinations for the method
        method_df = None

        # turn single-valued arguments to single-valued lists
        for k in args_space_dict:
            if not isinstance(args_space_dict[k], list) or \
                    (any([p in k for p in list_args_patterns]) and not isinstance(args_space_dict[k][0], list)):
                args_space_dict[k] = [args_space_dict[k]]

        # loop through each arguments combination
        args_combinations = get_args_combinations(args_space_dict)
        if len(args_combinations) > 1:
            # at least one method was provided with multiple relevant argument values
            multiple_args_combinations = True
        for args_dict in args_combinations:
            args = argparse.Namespace(**args_dict)

            # rename any f-score metric so that it includes the specified value of beta
            if 'f_score' in perf_specs['metrics']:
                perf_specs['metrics'][perf_specs['metrics'].index('f_score')] = \
                    f'f{args.f_score_beta}_score'

            # load results comparison spreadsheet for the provided argument values
            comparison_root = get_output_path(args, keys['pipeline'], 'comparison')
            evaluation_string = get_args_string(args, keys['evaluation'])
            if evaluation_step in ['scoring', 'detection', 'explanation']:
                # scoring, detection or explanation results comparison
                comparison_path = os.path.join(
                    comparison_root,
                    f'{evaluation_string}_{evaluation_step}_comparison.csv'
                )
            else:
                # modeling results comparison
                comparison_path = os.path.join(
                    comparison_root, f'{evaluation_string}_comparison.csv'
                )
            try:
                comparison_df = pd.read_csv(comparison_path)
            except FileNotFoundError:
                warnings.warn(f'Could not load file corresponding to evaluation string "{evaluation_string}".')
                continue

            # select results row for the provided argument values and granularity if relevant
            config_name = get_args_string(args, keys['config'])
            # regex=False to avoid interpreting brackets in the configuration string
            results_row = comparison_df[comparison_df['method'].str.contains(config_name, regex=False)]
            if len(results_row) == 0:
                # do not warn if the troublesome combination does not make sense to explore
                if not (args.n_iterations == 1 and args.removal_factor != 1.0):
                    warnings.warn(f'Could not find index "{config_name}" in the comparison spreadsheet.')
                continue
            if evaluation_step in ['scoring', 'detection', 'explanation']:
                results_row = results_row[results_row['granularity'] == perf_specs['granularity']]
            elif len(results_row) > 1:
                # if more than one run for the same model configuration, consider the latest one
                timestamp_index = results_row['method'].iloc[0].index('_run.') + 5
                latest_time = max(
                    [time.strptime(t, '%Y.%m.%d.%H.%M.%S') for t in results_row['method'].str[timestamp_index:]]
                )
                results_row = results_row[
                    results_row['method'].str.contains(time.strftime('%Y.%m.%d.%H.%M.%S', latest_time))
                ]
            # drop columns that served as index
            dropped_columns = ['method']
            if perf_specs['granularity'] is not None:
                dropped_columns.append('granularity')
            results_row = results_row.drop(dropped_columns, axis=1)

            # add columns for the average metrics across anomaly types if specified
            if perf_specs['anomaly_types'] is not None and 'avg' in perf_specs['anomaly_types']:
                results_row = add_avg_columns(
                    results_row, perf_specs['set_name'], perf_specs['metrics'], ANOMALY_TYPES,
                    perf_specs['anomaly_avg_type'], perf_specs['anomaly_types']
                )

            # select results columns according to the specified metrics and anomaly types if relevant
            filtered_results_row = results_row[get_column_names(
                results_row, perf_specs['set_name'], perf_specs['metrics'], perf_specs['anomaly_types']
            )]

            # add filtered results row to the method DataFrame
            method_df = filtered_results_row if method_df is None else method_df.append(filtered_results_row)

        if method_df is not None:
            # for table reporting, each method DataFrame is aggregated using the specified method
            if reporting_args.report_type == 'table':
                agg_method_series = method_df.agg(reporting_args.aggregation_method)
                agg_method_series.name = method_titles[i]
                if report is None:
                    report = pd.DataFrame(columns=agg_method_series.index)
                report = report.append(agg_method_series)

    if report is not None:
        full_output_path = None
        if output_path is not None:
            # create output path if it does not exist
            os.makedirs(output_path, exist_ok=True)

            # get reporting performance string from the reporting arguments
            reporting_performance_str = get_args_string(
                reporting_args, 'reporting_performance',
                include_agg_method=(reporting_args.report_type == 'table' and multiple_args_combinations)
            )

            # extend the provided output path with the file name (without extension)
            full_output_path = os.path.join(output_path, f'{reporting_performance_str}')

        # table reporting
        if reporting_args.report_type == 'table':
            report.index.name = 'METHOD'
            if full_output_path is not None:
                csv_output_path = f'{full_output_path}.csv'
                print(f'saving table to {csv_output_path}...', end=' ', flush=True)
                report.to_csv(csv_output_path, index=True)
                print('done.')

    # return the report to allow displaying it in some other way if relevant
    return report


if __name__ == '__main__':
    # parse and get command-line arguments
    args = parsers['report_results'].parse_args()

    # get output path
    OUTPUT_PATH = get_output_path(args, 'report_results')

    # report performance of the compared methods using the provided arguments
    report_results(args, COMPARED_ARGS, COMPARED_METHOD_TITLES, OUTPUT_PATH)
