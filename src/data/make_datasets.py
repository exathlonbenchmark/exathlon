"""Data partitioning module (train/test constitution).
"""
import os
import importlib

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from utils.common import parsers, get_output_path
from data.helpers import get_resampled
from data.data_managers import get_management_classes

if __name__ == '__main__':
    # get command-line arguments
    args = parsers['make_datasets'].parse_args()

    # set output path
    OUTPUT_PATH = get_output_path(args, 'make_datasets')

    # set data manager depending on the input data
    data_manager = get_management_classes()[args.data](args)

    # load period DataFrames, labels and information
    DATA_PATHS_DICT = importlib.import_module(f'utils.{args.data}').DATA_PATHS_DICT
    period_dfs, labels, periods_info = data_manager.load_raw_data(DATA_PATHS_DICT)

    # add an `Anomaly` column to the period DataFrames based on the labels
    period_dfs = data_manager.add_anomaly_column(period_dfs, labels, periods_info)

    # handle NaN values in the raw period DataFrames
    period_dfs, periods_info = data_manager.get_handled_nans(period_dfs, periods_info)

    # resample periods with their original resolution to avoid duplicate indices (max to remove the effect of -1s)
    sampling_period = importlib.import_module(f'utils.{args.data}').SAMPLING_PERIOD
    period_dfs = get_resampled(period_dfs, sampling_period, agg='max', anomaly_col=True)

    if args.data == 'spark':
        # handle any -1 executor and OS values that unexpectedly occurred during monitoring
        period_dfs = data_manager.get_handled_executor_features(period_dfs)
        period_dfs = data_manager.get_handled_os_features(period_dfs)

    # apply the specified period pruning if relevant
    period_dfs = data_manager.get_pruned_periods(period_dfs)

    # resample period DataFrames to the new sampling period if different from the original
    if args.pre_sampling_period != sampling_period:
        period_dfs = get_resampled(period_dfs, args.pre_sampling_period, anomaly_col=True)

    # save pipeline train/test period DataFrames and information
    data_manager.save_pipeline_datasets(period_dfs, periods_info, OUTPUT_PATH)
