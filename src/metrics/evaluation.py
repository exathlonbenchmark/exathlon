"""Anomaly detection evaluation module.

Gathers "scoring" and "detection" evaluation:
- Scoring evaluation assesses the ability of the provided outlier scores to separate
    out normal and anomalous records
- Detection evaluation assesses the ability of the provided binary predictions to
    accurately flag anomalies.
"""
import os
import importlib

import numpy as np
import pandas as pd

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from utils.spark import get_app_from_name
from metrics.evaluators import get_auc
from visualization.evaluation import plot_pr_curves
from visualization.functions import plot_curves
from visualization.periods.array import plot_scores_distributions


def get_metric_names_dict(evaluation_step, beta=None):
    """Returns a dictionary mapping the computed metrics to their name for `evaluation_step`.

    Args:
        evaluation_step (str): AD evaluation step (must be either "scoring" or "detection").
        beta (float|None): beta parameter for the F-score, must be provided and
            between 0 and 1 if `evaluation_step` is "detection".

    Returns:
        dict: the dictionary mapping the computed metrics to their name.
    """
    a_t = 'the AD evaluation step must be either `scoring` or `detection`'
    assert evaluation_step in ['scoring', 'detection'], a_t
    # scoring evaluation
    if evaluation_step == 'scoring':
        return {'auprc': 'AUPRC'}
    # detection evaluation
    assert beta is not None and 0 <= beta <= 1, 'beta must be set and between 0 and 1'
    return {
        'precision': 'PRECISION',
        'recall': 'RECALL',
        'f-score': f'F{beta}_SCORE'
    }


def get_column_names(periods_labels, anomaly_types, metric_names_dict, set_name):
    """Returns the column names to use for an evaluation DataFrame.

    Args:
        periods_labels (ndarray): periods labels of shape `(n_periods, period_length,)`.
            Where `period_length` depends on the period.
        anomaly_types (list|None): names of the anomaly types that we might encounter in the data (if relevant).
        metric_names_dict (dict): dictionary mapping metrics to compute to their name.
        set_name (str): the set name will be prepended to the column names.

    Returns:
        list: the column names to use for the evaluation.
    """
    set_prefix = f'{set_name.upper()}_'
    if anomaly_types is None:
        return [f'{set_prefix}{m}' for m in metric_names_dict.values()]
    column_names = []
    # positive classes encountered in the periods of the dataset
    pos_classes = np.delete(np.unique(np.concatenate(periods_labels, axis=0)), 0)
    # anomaly "classes" (`global` + one per type)
    class_names = ['global'] + [anomaly_types[i-1] for i in range(1, len(anomaly_types) + 1) if i in pos_classes]
    for cn in class_names:
        # precision is only considered globally
        class_metric_names = list(metric_names_dict.values()) if cn == 'global' else \
            [v for k, v in metric_names_dict.items() if k != 'precision']
        column_names += [f'{set_prefix}{cn.upper()}_{m}' for m in class_metric_names]
    return column_names


def save_ad_evaluation(evaluation_step, data_dict, evaluator, evaluation_string, config_name, spreadsheet_path,
                       *, used_data=None, method_path=None):
    """Adds the AD evaluation of the provided configuration to a sorted comparison spreadsheet.

    If the used data is specified and provides multiple anomaly types, the evaluation will be
    both performed "globally" (i.e. considering all types together) and type-wise.

    Note: the term "global" is both used when making no difference between applications or traces
    for spark data ("global granularity") and when making no difference between anomaly types ("global type").

    Args:
        evaluation_step (str): AD evaluation step (must be either "scoring" or "detection").
        data_dict (dict): datasets periods record-wise outlier scores/binary predictions, labels and information:
            - scores/predictions as `{(set_name)_scores|preds: ndarray}`.
            - labels as `{y_(set_name): ndarray}`.
            - info as `{(set_name)_info: ndarray}`. With each period info of the form `(file_name, anomaly_type, rank)`.
            For the first two array types, shape `(n_periods, period_length)`;
                `period_length` depending on the period.
        evaluator (metrics.Evaluator): object defining the anomaly detection metrics of interest.
        evaluation_string (str): formatted evaluation string to compare models under the same requirements.
        config_name (str): unique configuration identifier serving as an index in the spreadsheet.
        spreadsheet_path (str): comparison spreadsheet path.
        used_data (str|None): used data (if relevant, used to derive the types of anomaly we might encounter).
        method_path (str): scoring/detection method path, to save any extended evaluation to.
    """
    # set anomaly types and colors depending on the used data if relevant
    anomaly_types, metrics_colors = None, None
    if used_data is not None:
        try:
            anomaly_types = importlib.import_module(f'utils.{used_data}').ANOMALY_TYPES
            metrics_colors = importlib.import_module(f'visualization.helpers.{used_data}').METRICS_COLORS
        except ImportError:
            pass

    # set elements depending on the evaluation step
    a_t = 'the AD evaluation step must be either `scoring` or `detection`'
    assert evaluation_step in ['scoring', 'detection'], a_t
    metric_names_dict = get_metric_names_dict(evaluation_step, beta=evaluator.beta)
    if evaluation_step == 'scoring':
        get_metrics_row, pred_type = get_scoring_metrics_row, 'scores'
    else:
        get_metrics_row, pred_type = get_detection_metrics_row, 'preds'

    # set the full path for the comparison spreadsheet
    full_spreadsheet_path = os.path.join(
        spreadsheet_path, f'{evaluation_string}_{evaluation_step}_comparison.csv'
    )

    # evaluated dataset names
    set_names = [n.replace(f'_{pred_type}', '') for n in data_dict if f'_{pred_type}' in n]

    # evaluation DataFrame for each considered dataset
    set_evaluation_dfs = []
    for n in set_names:
        # setup column space and hierarchical index for the current dataset
        periods_labels, periods_preds = data_dict[f'y_{n}'], data_dict[f'{n}_{pred_type}']
        column_names = get_column_names(periods_labels, anomaly_types, metric_names_dict, n)
        set_evaluation_dfs.append(
            pd.DataFrame(
                columns=column_names, index=pd.MultiIndex.from_tuples([], names=['method', 'granularity'])
            )
        )
        evaluation_df = set_evaluation_dfs[-1]

        # add metrics when considering all traces and applications the same
        evaluation_df.loc[(config_name, 'global'), :] = get_metrics_row(
            periods_labels, periods_preds, evaluator, column_names, metric_names_dict,
            anomaly_types, granularity='global', method_path=method_path,
            evaluation_string=evaluation_string, metrics_colors=metrics_colors
        )

        # if using spark data, add metrics for each application and trace
        if used_data == 'spark':
            # application-wise performance
            periods_info = data_dict[f'{n}_info']
            app_ids = set([get_app_from_name(info[0]) for info in periods_info])
            for app_id in app_ids:
                app_indices = [i for i, info in enumerate(periods_info) if get_app_from_name(info[0]) == app_id]
                app_labels, app_preds, app_info = [], [], []
                for i in range(len(periods_info)):
                    if i in app_indices:
                        app_labels.append(periods_labels[i])
                        app_preds.append(periods_preds[i])
                        app_info.append(periods_info[i])
                evaluation_df.loc[(config_name, f'app{app_id}'), :] = get_metrics_row(
                    app_labels, app_preds, evaluator, column_names, metric_names_dict,
                    anomaly_types, granularity='app', metrics_colors=metrics_colors,
                    method_path=method_path, periods_key=f'app{app_id}'
                )
                # trace-wise performance
                trace_names = set([info[0] for info in app_info])
                for trace_name in trace_names:
                    trace_indices = [i for i, info in enumerate(app_info) if info[0] == trace_name]
                    trace_labels, trace_preds, trace_info = [], [], []
                    for j in range(len(app_info)):
                        if j in trace_indices:
                            trace_labels.append(app_labels[j])
                            trace_preds.append(app_preds[j])
                            trace_info.append(app_info[j])
                    evaluation_df.loc[(config_name, trace_info[0][0]), :] = get_metrics_row(
                        trace_labels, trace_preds, evaluator, column_names, metric_names_dict,
                        anomaly_types, granularity='trace', method_path=method_path, periods_key=trace_name
                    )
                # average performance across traces (in-trace separation ability)
                trace_rows = evaluation_df.loc[~(
                    (evaluation_df.index.get_level_values('granularity').str.contains('global')) |
                    (evaluation_df.index.get_level_values('granularity').str.contains('app'))
                )]
                evaluation_df.loc[(config_name, 'trace_avg'), :] = trace_rows.mean(axis=0)
            # average performance across applications (in-application separation ability)
            app_rows = evaluation_df.loc[
                evaluation_df.index.get_level_values('granularity').str.contains('app')
            ]
            evaluation_df.loc[(config_name, 'app_avg'), :] = app_rows.mean(axis=0)

    # add the new evaluation to the comparison spreadsheet, or create it if it does not exist
    evaluation_df = pd.concat(set_evaluation_dfs, axis=1)
    try:
        comparison_df = pd.read_csv(full_spreadsheet_path, index_col=[0, 1]).astype(float)
        print(f'adding evaluation of `{config_name}` to {full_spreadsheet_path}...', end=' ', flush=True)
        for index_key in evaluation_df.index:
            comparison_df.loc[index_key, :] = evaluation_df.loc[index_key, :].values
        comparison_df.to_csv(full_spreadsheet_path)
        print('done.')
    except FileNotFoundError:
        print(f'creating {full_spreadsheet_path} with evaluation of `{config_name}`...', end=' ', flush=True)
        evaluation_df.to_csv(full_spreadsheet_path)
        print('done.')


def get_scoring_metrics_row(labels, scores, evaluator, column_names, metric_names_dict,
                            anomaly_types, granularity, method_path=None, evaluation_string=None,
                            metrics_colors=None, periods_key=None):
    """Returns the metrics row to add to a scoring evaluation DataFrame.

    Note: the column names do not determine the metrics that will be computed by the function
    but only their names (only AUPRC is computed here no matter the provided name).

    Args:
        labels (ndarray): periods labels of shape `(n_periods, period_length)`.
            Where `period_length` depends on the period.
        scores (ndarray): periods outlier scores with the same shape as `labels`.
        evaluator (metrics.Evaluator): object defining the binary metrics of interest.
        column_names (list): list of column names corresponding to the metrics to compute.
        metric_names_dict (dict): dictionary mapping metrics to compute to their name.
        anomaly_types (list|None): names of the anomaly types that we might encounter in the data (if relevant).
        granularity (str): evaluation granularity.
            Must be either `global`, for overall, `app`, for app-wise or `trace`, for trace-wise.
        method_path (str|None): scoring method path, to save any extended evaluation to.
        evaluation_string (str|None): formatted evaluation string to compare models under the same requirements.
        metrics_colors (dict|str|None): color to use for the curves if single value, color to use
            for each anomaly type if dict (the keys must then belong to `anomaly_types`).
        periods_key (str): if granularity is not `global`, name to use to identify the periods.
            Has to be of the form `appX` if `app` granularity or `trace_name` if `trace` granularity.

    Returns:
        list: list of metrics to add to the evaluation DataFrame (corresponding to `column_names`).
    """
    a_t = 'the provided granularity must be either `global`, `app` or `trace`'
    assert granularity in ['global', 'app', 'trace'], a_t

    # set metrics keys to make sure the output matches to order of `column_names`
    metrics_row = pd.DataFrame(columns=column_names)
    metrics_row.append(pd.Series(), ignore_index=True)

    # recover the dataset prefix, name and title from the column names
    set_prefix = f'{column_names[0].split("_")[0]}_'
    set_name = set_prefix[:-1].lower()
    set_title = set_name.capitalize()

    # set the metric names from the metric names dictionary
    auprc_name = metric_names_dict["auprc"]

    # compute the PR curve(s) using the Precision and Recall metrics defined by the evaluator
    f, p, r, pr_ts = evaluator.precision_recall_curves(labels, scores, return_f_scores=True)
    # we do not use average metrics across types here
    for d in f, r:
        d.pop('avg')
    # case of a single anomaly type
    if anomaly_types is None:
        metrics_row.at[0, f'{set_prefix}{auprc_name}'] = get_auc(r['global'], p)
        r, f = r['global'], f['global']
    # case of multiple known anomaly types
    else:
        # `global` and interpretable anomaly types that belong to the keys of recall curves
        class_names = ['global'] + [anomaly_types[i-1] for i in range(1, len(anomaly_types) + 1) if i in r]
        # add the metric and column corresponding to each type
        for cn in class_names:
            if cn == 'global':
                metrics_row.at[0, f'{set_prefix}GLOBAL_{auprc_name}'] = get_auc(r['global'], p)
            else:
                label_key = anomaly_types.index(cn) + 1
                metrics_row.at[0, f'{set_prefix}{cn.upper()}_{auprc_name}'] = get_auc(r[label_key], p)
                # update the label keys to reflect interpretable anomaly types suited for visualizations
                r[cn], f[cn] = r.pop(label_key), f.pop(label_key)

    if granularity in ['global', 'app']:
        # save the distributions of outlier scores grouped by record type
        periods_suffix, fig_title_suffix = '', ''
        if granularity == 'global':
            periods_suffix, fig_title_suffix = '_global', ' Globally'
        elif periods_key is not None:
            periods_suffix, fig_title_suffix = f'_{periods_key}', f' for Application {periods_key[3:]}'
        full_output_path = os.path.join(method_path, f'{set_name}{periods_suffix}_scores_distributions.png')
        a_t = 'the provided colors must be either a dict or `None`'
        assert type(metrics_colors) == dict or metrics_colors is None, a_t
        plot_scores_distributions(
            scores, labels,
            fig_title=f'{set_title} Scores Distributions{fig_title_suffix}',
            type_colors=metrics_colors,
            anomaly_types=anomaly_types,
            full_output_path=full_output_path
        )

    if granularity == 'global':
        # save the full PR curve(s) under the method path
        full_output_path = os.path.join(method_path, f'{evaluation_string}_{set_name}_pr_curve.png')
        plot_pr_curves(
            r, p, pr_ts,
            fig_title=f'Precision-Recall Curve(s) on the {set_title} Set',
            colors=metrics_colors,
            full_output_path=full_output_path
        )
        # save the F-score curve(s) under the method path
        full_output_path = os.path.join(method_path, f'{evaluation_string}_{set_name}_f{evaluator.beta}_curve.png')
        plot_curves(
            pr_ts, f,
            fig_title=f'F{evaluator.beta}-Score(s) on the {set_title} Set',
            xlabel='Threshold',
            ylabel=f'F{evaluator.beta}-Score',
            colors=metrics_colors,
            show_max_values=True,
            full_output_path=full_output_path
        )
    return metrics_row.iloc[0].tolist()


def get_detection_metrics_row(labels, preds, evaluator, column_names, metric_names_dict,
                              anomaly_types, granularity, method_path=None, evaluation_string=None,
                              metrics_colors=None, periods_key=None):
    """Returns the metrics row to add to a detection evaluation DataFrame.

    Args:
        labels (ndarray): periods labels of shape `(n_periods, period_length)`.
            Where `period_length` depends on the period.
        preds (ndarray): periods binary predictions with the same shape as `labels`.
        evaluator (metrics.Evaluator): object defining the binary metrics of interest.
        column_names (list): list of column names corresponding to the metrics to compute.
        metric_names_dict (dict): dictionary mapping metrics to compute to their name.
        anomaly_types (list|None): names of the anomaly types that we might encounter in the data (if relevant).
        granularity (str): evaluation granularity.
            Must be either `global`, for overall, `app`, for app-wise or `trace`, for trace-wise.
        method_path (str|None): detection method path, to save any extended evaluation to.
        evaluation_string (str|None): formatted evaluation string to compare models under the same requirements.
        metrics_colors (dict|str|None): color to use for the curves if single value, color to use
            for each anomaly type if dict (the keys must then belong to `anomaly_types`).
        periods_key (str): if granularity is not `global`, name to use to identify the periods.
            Has to be of the form `appX` if `app` granularity or `trace_name` if `trace` granularity.

    Returns:
        list: list of metrics to add to the evaluation DataFrame (corresponding to `column_names`).
    """
    assert granularity in ['global', 'app', 'trace']

    # set metrics keys to make sure the output matches to order of `column_names`
    metrics_row = pd.DataFrame(columns=column_names)
    metrics_row.append(pd.Series(), ignore_index=True)
    metric_names = list(metric_names_dict.values())

    # recover the dataset prefix from the column names
    set_prefix = f'{column_names[0].split("_")[0]}_'

    # compute the metrics defined by the evaluator
    f, p, r = evaluator.compute_metrics(labels, preds)
    # we do not use average metrics across types here
    for d in f, r:
        d.pop('avg')
    # case of a single anomaly type
    if anomaly_types is None:
        for m_name, m_value in zip(metric_names, [p, r['global'], f['global']]):
            metrics_row.at[0, f'{set_prefix}{m_name}'] = m_value
    # case of multiple known anomaly types
    else:
        # `global` and interpretable anomaly types that belong to the keys of recall curves
        class_names = ['global'] + [anomaly_types[i-1] for i in range(1, len(anomaly_types) + 1) if i in r]
        # add the metric and column corresponding to each type
        for cn in class_names:
            if cn == 'global':
                for m_name, m_value in zip(metric_names, [p, r['global'], f['global']]):
                    metrics_row.at[0, f'{set_prefix}GLOBAL_{m_name}'] = m_value
            else:
                # precision is only considered globally
                class_metric_names = [v for k, v in metric_names_dict.items() if k != 'precision']
                label_key = anomaly_types.index(cn) + 1
                for m_name, m_value in zip(class_metric_names, [r[label_key], f[label_key]]):
                    metrics_row.at[0, f'{set_prefix}{cn.upper()}_{m_name}'] = m_value
    return metrics_row.iloc[0].tolist()
