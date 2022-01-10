"""Spark-specific data management module.
"""
import os

import numpy as np
import pandas as pd

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from utils.common import PIPELINE_SET_NAMES
from utils.spark import APP_IDS, TRACE_TYPES, ANOMALY_TYPES
from data.data_managers import DataManager
from metrics.ad_evaluators import extract_binary_ranges_ids


def load_trace(trace_path):
    """Loads a Spark trace as a pd.DataFrame from its full input path.

    Args:
        trace_path (str): full path of the trace to load (with file extension).

    Returns:
        pd.DataFrame: the trace indexed by time, with columns processed to be consistent between traces.
    """
    # load trace DataFrame with time as its converted datetime index
    trace_df = pd.read_csv(trace_path)
    trace_df.index = pd.to_datetime(trace_df['t'], unit='s')
    trace_df = trace_df.drop('t', axis=1)

    # remove the previous file prefix from streaming metrics for their name to be consistent across traces
    trace_df.columns = [
        c.replace(f'{"_".join(c.split("_")[1:10])}_', '') if 'StreamingMetrics' in c else c
        for c in trace_df.columns
    ]

    # return the DataFrame with sorted columns (they might be in a different order depending on the file)
    return trace_df.reindex(sorted(trace_df.columns), axis=1)


class SparkManager(DataManager):
    """Spark-specific data management class.
    """
    def __init__(self, args):
        super().__init__(args)
        # restrict traces to a specific application id or trace types
        self.app_id = args.app_id
        self.trace_types = TRACE_TYPES if args.trace_types == '.' else args.trace_types.split('.')
        # optionally ignore some events (set them as normal)
        self.ignored_anomalies = args.ignored_anomalies

    def load_raw_data(self, data_paths_dict):
        """Spark-specific raw data loading.

        - The loaded periods as the application(s) traces.
        - The labels as the ground-truth table gathering events information for disturbed traces.
        - The periods information as lists initialized in the form `[file_name, trace_type]`.
        """
        period_dfs, periods_info = [], []
        print('loading ground-truth table...', end=' ', flush=True)
        labels = pd.read_csv(os.path.join(data_paths_dict['labels'], 'ground_truth.csv'))

        # convert timestamps to datetime format
        for c in ['root_cause_start', 'root_cause_end', 'extended_effect_end']:
            labels[c] = pd.to_datetime(labels[c], unit='s')
        print('done.')

        # select relevant application keys (we also exclude 7 and 8 if all applications)
        app_keys = [f'app{self.app_id}'] if self.app_id != 0 else [f'app{i}' for i in set(APP_IDS) - {7, 8}]

        # load traces of the selected application(s) and type(s)
        for app_key in app_keys:
            print(f'loading traces of application {app_key.replace("app", "")}')
            app_path = data_paths_dict[app_key]
            file_names = os.listdir(app_path)
            for trace_type in self.trace_types:
                type_file_names = [
                    fn for fn in file_names if int(fn.split('_')[1]) == TRACE_TYPES.index(trace_type)
                ]
                if len(type_file_names) > 0:
                    print(f'loading {trace_type.replace("_", " ")} traces...', end=' ', flush=True)
                    for fn in type_file_names:
                        period_dfs.append(load_trace(os.path.join(app_path, fn)))
                        periods_info.append([fn[:-4], trace_type])
                    print('done.')
        assert len(period_dfs) > 0, 'no traces for the provided application(s) and type(s).'
        return period_dfs, labels, periods_info

    def add_anomaly_column(self, period_dfs, labels, periods_info):
        """Spark-specific `Anomaly` column extension.

        Note: `periods_info` is assumed to be of the form `[file_name, trace_type]`.

        `Anomaly` will be set to 0 if the record is outside any anomaly range, otherwise it will be
        set to another value depending on the range type (as defined by utils.spark.ANOMALY_TYPES).
        => The label for a given range type corresponds to its index in the ANOMALY_TYPES list +1.
        """
        print('adding an `Anomaly` column to the Spark traces...', end=' ', flush=True)
        extended_dfs = [period_df.copy() for period_df in period_dfs]
        for i, period_df in enumerate(extended_dfs):
            period_df['Anomaly'] = 0
            file_name, trace_type = periods_info[i]
            if trace_type != 'undisturbed':
                for a_t in labels[labels['trace_name'] == file_name].itertuples():
                    # ignore anomalies that had no impact on the recorded application if specified
                    if not (self.ignored_anomalies == 'os.only' and a_t.anomaly_details == 'no_application_impact'):
                        a_start = a_t.root_cause_start
                        # either set the anomaly end to the root cause or extended effect end if the latter is set
                        a_end = a_t.root_cause_end if pd.isnull(a_t.extended_effect_end) else \
                            a_t.extended_effect_end
                        # set the label of an anomaly type as its index in the types list +1
                        period_df.loc[(period_df.index >= a_start) & (period_df.index <= a_end), 'Anomaly'] = \
                            ANOMALY_TYPES.index(a_t.anomaly_type) + 1
        print('done.')
        return extended_dfs

    def get_handled_nans(self, period_dfs, periods_info):
        """Spark-specific handling of NaN values.

        The only NaN values that were recorded were for inactive executors, found equivalent
        of them being -1.
        """
        print('handling NaN values encountered in Spark traces...', end=' ', flush=True)
        handled_dfs = [period_df.copy() for period_df in period_dfs]
        for period_df in handled_dfs:
            period_df.fillna(-1, inplace=True)
        print('done.')
        return handled_dfs, periods_info

    @staticmethod
    def get_handled_executor_features(period_dfs):
        """Returns the period DataFrames with "handled" executor features.

        By looking at the features, we saw that some executor spots sometimes went to -1,
        presumably because we did not receive data from them within a given delay, without
        meaning the executors were not active anymore.

        The goal of this method is to detect such scenarios, in which case all -1 features are replaced
        with their last valid occurrence.

        Note: it was checked that if a given feature for an executor spot is -1, then all features
        from that spot are also -1, except for 1 or 2 records of some traces, which we argue is negligible.
        => To handle these very few cases, we would explicitly set all features to -1.

        Args:
            period_dfs (list): the list of input period DataFrames. Assumed without NaNs.

        Returns:
            list: the new period DataFrames, with handled executor features.
        """
        print('handling executor features in Spark traces...', end=' ', flush=True)
        # periods with handled executor features
        handled_dfs = [period_df.copy() for period_df in period_dfs]
        # copies used to extract continuous ranges of -1s
        extraction_dfs = [period_df.copy() for period_df in period_dfs]
        for handled_df, extraction_df in zip(handled_dfs, extraction_dfs):
            for executor_spot in range(1, 6):
                # take an arbitrary counter feature of the executor and extract -1 ranges for it
                counter_name = f'{executor_spot}_executor_runTime_count'
                extraction_df.loc[handled_df[counter_name] == -1, counter_name] = 1
                extraction_df.loc[handled_df[counter_name] != -1, counter_name] = 0
                for start_idx, end_idx in extract_binary_ranges_ids(extraction_df[counter_name].values):
                    # only consider filling -1s if the range is between two non-(-1) ranges (end is excluded)
                    if start_idx != 0 and end_idx != len(handled_df):
                        # if the counter was not reset, fill all executor features with their last valid value
                        preceding_counter = handled_df[counter_name].iloc[start_idx-1]
                        following_counter = handled_df[counter_name].iloc[end_idx]
                        if following_counter >= preceding_counter:
                            # end is included for the `loc` method
                            start_time, end_time = handled_df.index[start_idx], handled_df.index[end_idx-1]
                            valid_time = handled_df.index[start_idx-1]
                            for ft_name in [c for c in handled_df.columns if c[:2] == f'{executor_spot}_']:
                                handled_df.loc[start_time:end_time, ft_name] = handled_df.loc[valid_time, ft_name]
        print('done.')
        return handled_dfs

    @staticmethod
    def get_handled_os_features(period_dfs):
        """Returns the period DataFrames with "handled" OS features.

        Similarly to some executor features, some OS features might happen to be -1 for no
        other reason than their real value not being sent fast enough by the monitoring software.

        This is here true for all encountered -1 values that do not span an entire trace.
        In such scenarios, we replace -1 features with their last valid value (or their next one
        if their last is not available).

        Args:
            period_dfs (list): the list of input period DataFrames. Assumed without NaNs.

        Returns:
            list: the new period DataFrames, with handled OS features.
        """
        print('handling OS features in Spark traces...', end=' ', flush=True)
        os_ft_names = [c for c in period_dfs[0].columns if c[:4] == 'node']
        handled_dfs = [period_df.copy() for period_df in period_dfs]
        for handled_df in handled_dfs:
            handled_df[os_ft_names] = handled_df[os_ft_names].replace(-1, np.nan).ffill().bfill().fillna(-1)
        print('done.')
        return handled_dfs

    def get_pipeline_split(self, period_dfs, periods_info):
        """Spark-specific pipeline train/test splitting.

        Note: `periods_info` is assumed to be of the form `[file_name, trace_type]`.
        It will however be returned in the form `[file_name, trace_type, period_rank]`.
        Where `period_rank` is the chronological rank of the period in its file.

        Note: `period_rank` would only be relevant if we were to split traces into
        multiple periods, which we do not do here but is supported in the rest of the pipeline.

        All undisturbed traces are sent to `train`. All disturbed traces are sent to `test`.
        """
        datasets = {k: dict() for k in ['dfs', 'info']}
        for ds_name in PIPELINE_SET_NAMES:
            datasets['dfs'][ds_name], datasets['info'][ds_name] = [], []
            # all undisturbed traces are sent to `train`, all disturbed traces to `test`
            print(f'constituting {ds_name} periods...', end=' ', flush=True)
            set_ids = [
                i for i, info in enumerate(periods_info) if (ds_name == 'train') == (info[1] == 'undisturbed')
            ]
            for i, period_df in enumerate(period_dfs):
                if i in set_ids:
                    datasets['dfs'][ds_name].append(period_df)
                    # every period is alone (hence at rank 0) in its trace file
                    datasets['info'][ds_name].append(periods_info[i] + [0])
            print('done.')
        # return the period DataFrames and information per dataset
        return datasets['dfs'], datasets['info']
