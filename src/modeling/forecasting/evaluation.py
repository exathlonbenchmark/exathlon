"""Forecasting evaluation module.
"""
import os

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from utils.common import MODELING_SET_NAMES
from modeling.helpers import prepend_training_info


def save_forecasting_evaluation(data_dict, forecaster, modeling_task_string, config_name, spreadsheet_path):
    """Adds the forecasting evaluation of this configuration to a sorted comparison spreadsheet.

    Args:
        data_dict (dict): train, val and test sequences and targets, as `{X|y_(modeling_set_name): ndarray}`.
             - Shapes `(n_samples, n_back, n_features)` for sequences.
             - Shapes `(n_samples, n_features)` or `(n_samples, n_forward, n_features)` for targets.
        forecaster (Forecaster): the Forecaster object to evaluate.
        modeling_task_string (str): formatted modeling task arguments to compare models under the same task.
        config_name (str): unique configuration identifier serving as an index in the spreadsheet.
        spreadsheet_path (str): comparison spreadsheet path.

    Returns:
        pd.DataFrame: the 1-row evaluation DataFrame holding the computed metrics.
    """
    # we only handle 1 forecast-ahead for now
    assert len(data_dict['y_train'].shape) == 2, 'only 1-forecast-ahead evaluation is supported for now'

    # set the full path for the comparison spreadsheet
    full_spreadsheet_path = os.path.join(spreadsheet_path, f'{modeling_task_string}_comparison.csv')

    # compute and add forecasting metrics
    column_names, metrics = [], []
    for set_name in MODELING_SET_NAMES:
        print(f'evaluating forecasting metrics on the {set_name} set...', end=' ', flush=True)
        set_metrics = forecaster.evaluate(data_dict[f'X_{set_name}'], data_dict[f'y_{set_name}'])
        for metric_name in set_metrics:
            metrics.append(set_metrics[metric_name])
            column_names.append((set_name + '_' + metric_name).upper())
        print('done.')
    # prepend training time information
    metrics, column_names = prepend_training_info(forecaster, metrics, column_names)
    evaluation_df = pd.DataFrame(columns=column_names, data=[metrics], index=[config_name])
    evaluation_df.index.name = 'method'

    # add the new evaluation to the comparison spreadsheet, or create it if it does not exist
    try:
        comparison_df = pd.read_csv(full_spreadsheet_path, index_col=0).astype(float)
        print(f'adding evaluation of `{config_name}` to {full_spreadsheet_path}...', end=' ', flush=True)
        comparison_df.loc[evaluation_df.index[0], :] = evaluation_df.values[0]
        comparison_df.sort_values(by=list(reversed(column_names)), ascending=True, inplace=True)
        comparison_df.to_csv(full_spreadsheet_path)
        print('done.')
    except FileNotFoundError:
        print(f'creating {full_spreadsheet_path} with evaluation of `{config_name}`...', end=' ', flush=True)
        evaluation_df.to_csv(full_spreadsheet_path)
        print('done.')
    return evaluation_df


def get_mean_squared_error(y, y_pred):

    """Returns the MSE of the forecaster's predictions."""
    return mean_squared_error(y, y_pred)


def get_mean_absolute_error(y, y_pred):
    """Returns the MAE of the forecaster's predictions."""
    return mean_absolute_error(y, y_pred)
