"""Spark-specific features alteration functions and bundles.
"""
import numpy as np
import pandas as pd


def add_executors_avg(period_df, original_treatment):
    """Adds executor features averaged across active executors, keeping or not the original ones.

    An executor is defined as "inactive" for a given feature if the value of the feature for
    this executor is -1.

    Note: it was checked that if a given feature for an executor spot is -1, then all features
    from that spot are also -1, except for 1 or 2 records of some traces, which we argue is negligible.
    => To handle these few cases, we should have set all the features to -1 (see
    `data.spark_manager.SparkManager.get_handled_executor_features`).
    """
    assert original_treatment in ['drop', 'keep'], 'original features treatment can only be `keep` or `drop`'

    # make sure to only try to average executor features
    exec_ft_names = [c[2:] for c in period_df.columns if c[:2] == '1_']
    # features groups to average across, each group of the form [`1_ft`, `2_ft`, ..., `5_ft`]
    avg_groups = [[c for c in period_df.columns if c[2:] == efn] for efn in exec_ft_names]

    # add features groups averaged across active executors to the result DataFrame
    averaged_df = pd.DataFrame()
    for group in avg_groups:
        # create `avg_ft` from [`1_ft`, `2_ft`, ..., `5_ft`]
        averaged_df = averaged_df.assign(
            **{f'avg_{group[0][2:]}': period_df[group].replace(-1, np.nan).mean(axis=1).fillna(-1)}
        )
    # prepend original input features if we choose to keep them
    if original_treatment == 'keep':
        averaged_df = pd.concat([period_df, averaged_df], axis=1)
    return averaged_df


def add_nodes_avg(period_df, original_treatment):
    """Adds node features averaged across nodes, keeping or not the original ones.
    """
    assert original_treatment in ['drop', 'keep'], 'original features treatment can only be `keep` or `drop`'

    # make sure to only try to average node features
    node_ft_names = [c[6:] for c in period_df.columns if c[:4] == 'node']
    # features groups to average across, each group of the form [`node5_ft`, `node6_ft`, ..., `node8_ft`]
    avg_groups = [[c for c in period_df.columns if c[6:] == nfn] for nfn in node_ft_names]

    # add features groups averaged across nodes to the result DataFrame
    averaged_df = pd.DataFrame()
    for group in avg_groups:
        # create `avg_node_ft` from [`node5_ft`, `node6_ft`, ..., `node8_ft`]
        averaged_df = averaged_df.assign(
            **{f'avg_node_{group[0][6:]}': period_df[group].mean(axis=1)}
        )
    # prepend original input features if we choose to keep them
    if original_treatment == 'keep':
        averaged_df = pd.concat([period_df, averaged_df], axis=1)
    return averaged_df


# list of features alteration bundles relevant to spark data
SPARK_BUNDLES = [
    # bundle #0: set of 19 custom features
    {
        # features to add as is
        (
            'driver_StreamingMetrics_streaming_lastCompletedBatch_processingDelay_value',
            'driver_StreamingMetrics_streaming_lastCompletedBatch_schedulingDelay_value',
            'driver_StreamingMetrics_streaming_lastCompletedBatch_totalDelay_value'
        ): 'identity',
        # features to 1-difference, dropping the original ones
        (
            'driver_StreamingMetrics_streaming_totalCompletedBatches_value',
            'driver_StreamingMetrics_streaming_totalProcessedRecords_value',
            'driver_StreamingMetrics_streaming_totalReceivedRecords_value',
            'driver_StreamingMetrics_streaming_lastReceivedBatch_records_value',
            'driver_BlockManager_memory_memUsed_MB_value',
            'driver_jvm_heap_used_value',
            *[f'node{i}_CPU_ALL_Idle%' for i in range(5, 9)]
        ): 'difference_1_drop',
        # features to average across active executors and 1-difference, dropping the original inputs every time
        (
            *[f'{i}_executor_filesystem_hdfs_write_ops_value' for i in range(1, 6)],
            *[f'{i}_executor_cpuTime_count' for i in range(1, 6)],
            *[f'{i}_executor_runTime_count' for i in range(1, 6)],
            *[f'{i}_executor_shuffleRecordsRead_count' for i in range(1, 6)],
            *[f'{i}_executor_shuffleRecordsWritten_count' for i in range(1, 6)],
            *[f'{i}_jvm_heap_used_value' for i in range(1, 6)]
        ): 'execavg_drop.difference_1_drop'
    }
]
