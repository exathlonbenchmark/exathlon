"""Reporting helpers module.
"""
import warnings
import itertools


class InvalidSpecificationsError(Exception):
    """Exception raised in case invalid specifications combinations were provided.

    Args:
        message (str): description of the error.
    """
    def __init__(self, message):
        self.message = message


def check_reporting_args(reporting_args, anomaly_types):
    """Further checks the provided reporting argument values, raising an `InvalidSpecificationsError`
        if they are not valid.

    Args:
        reporting_args (argparse.Namespace): parsed reporting command-line arguments.
        anomaly_types (list): anomaly types for the considered data.
    """
    if reporting_args.evaluation_step in ['scoring', 'detection', 'explanation']:
        d, keys = vars(reporting_args), ['anomaly_types', 'anomaly_avg_type']
        reported_types, anomaly_avg_type = [d[f'{reporting_args.evaluation_step}_{k}'] for k in keys]
        if 'avg' in reported_types:
            if len(anomaly_types) == 0:
                e_m = 'Cannot average across anomaly types if types were not specified for the data.'
                raise InvalidSpecificationsError(e_m)
            if anomaly_avg_type == 'reported':
                if len(set(reported_types) - {'global', 'avg'}) == 0:
                    e_m = 'Cannot average only across reported anomaly types if none were reported.'
                    raise InvalidSpecificationsError(e_m)


def get_args_combinations(args_space_dict):
    """Returns every possible argument combinations from the space defined by `args_space_dict`.

    Args:
        args_space_dict (dict): arguments space dictionary, of the form `{arg_name: value_list}`,
            where `arg_name` is the argument name and `value_list` the list of values to try.

    Returns:
        list: the list of possible argument dictionaries from the space.
    """
    return [dict(zip(args_space_dict, c)) for c in itertools.product(*args_space_dict.values())]


def add_avg_columns(results_row, set_name, metrics, anomaly_types, avg_type='all', reported_types=None):
    """Returns `results_row` with added metrics averaged across anomaly types according to `avg_type`.

    Args:
        results_row (pd.DataFrame): one-row DataFrame coming from a results comparison spreadsheet.
        set_name (str): dataset name to report metrics on.
        metrics (list): list of metrics to average across anomaly types.
        anomaly_types (list): list of (all) anomaly types specified for the data.
        avg_type (str): anomaly types averaging strategy. Depending on the considered data, could
            be either `all`, `reported` or `all_but_unknown`.
        reported_types (list|None): anomaly types to show in the reporting, only relevant if
            `avg_type` is `reported`. Any `global` key will be ignored no matter `avg_type`.

    Returns:
        pd.DataFrame: `results_row` extended with relevant metrics averaged across anomaly types.
    """
    upper_set, upper_metrics = set_name.upper(), [m.upper() for m in metrics]
    avg_idx, ignored_patterns = 0, ['GLOBAL']
    if avg_type == 'all_but_unknown':
        ignored_patterns += ['UNKNOWN']
    elif avg_type == 'reported':
        a_t = '`reported_types` must be provided if specifying average across reported types only'
        assert reported_types is not None, a_t
        ignored_patterns += [t.upper() for t in anomaly_types if t not in reported_types]
    for i, u_m in enumerate(upper_metrics):
        metric_cols = [
            c for c in results_row.columns
            if all([p in c for p in [u_m, upper_set]])
            and not any([p in c for p in ignored_patterns])
        ]
        metric_avg = results_row[metric_cols].mean(axis=1)
        if not metric_avg.isnull().any():
            results_row.insert(avg_idx, f'{upper_set}_AVG_{u_m}', metric_avg)
            avg_idx += 1
        else:
            warnings.warn(f'Could not compute {metrics[i]} average across anomaly types.')
    return results_row


def get_column_names(results_row, set_name, metrics, reported_types=None):
    """Returns the column names to select in `results_row`, according to the specified `set_name`,
        `metrics` and `reported_types` to consider.

    Args:
        results_row (pd.DataFrame): one-row DataFrame coming from a results comparison spreadsheet,
            potentially extended with "average" columns coming from `add_avg_columns`.
        set_name (str): dataset name to report metrics on.
        metrics (list): list of metrics to report.
        reported_types (list|None): optional list of anomaly types to consider, potentially
            including `global` or `avg`. Only relevant for steps other than `modeling`.

    Returns:
        list: the column names to select in `results_row` according to the reporting specifications.
    """
    upper_set, upper_metrics = set_name.upper(), [m.upper() for m in metrics]
    if reported_types is not None:
        upper_types = [t.upper() for t in reported_types]
        common_patterns, set_patterns = [], []
        for u_t in upper_types:
            for u_m in upper_metrics:
                common_patterns.append(f'{u_t}_{u_m}')
                set_patterns.append(f'{upper_set}_{u_t}_{u_m}')
    else:
        common_patterns = upper_metrics
        set_patterns = [f'{upper_set}_{u_m}' for u_m in upper_metrics]
    common_patterns, set_patterns = set(common_patterns), set(set_patterns)
    # we do not use set operations here to preserve the original order of columns
    column_names = []
    for c in results_row.columns:
        if any([p in c for p in common_patterns]) or any([p in c for p in set_patterns]):
            column_names.append(c)
    return column_names
