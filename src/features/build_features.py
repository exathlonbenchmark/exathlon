"""Features building module.

Turns the raw period columns to the final features that will be used by the models.
If specified, this pipeline step can also resample the records and/or labels to new sampling periods.
"""
import os

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from utils.common import PIPELINE_SET_NAMES, CHOICES, parsers, get_output_path
from data.helpers import (
    load_files, save_files, extract_save_labels,
    get_resampled, get_numpy_from_dfs, get_dfs_from_numpy
)
from features.alteration import get_altered_features, get_bundles_lists
from features.transformers import transformation_classes


if __name__ == '__main__':
    # parse and get command-line arguments
    args = parsers['build_features'].parse_args()

    # get input and output paths
    INPUT_DATA_PATH = get_output_path(args, 'make_datasets')
    OUTPUT_DATA_PATH = get_output_path(args, 'build_features', 'data')
    OUTPUT_MODELS_PATH = get_output_path(args, 'build_features', 'models')

    # load datasets
    datasets = load_files(INPUT_DATA_PATH, PIPELINE_SET_NAMES, 'pickle')

    # extract, optionally downsample, and save labels from the `Anomaly` columns of the dataset periods
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
    labels_downsampling = args.labels_sampling_period != args.pre_sampling_period
    labels_sampling_period = args.labels_sampling_period if labels_downsampling else None
    print(f'{"downsampling and " if labels_downsampling else ""}saving labels to {OUTPUT_DATA_PATH}')
    for k in datasets:
        datasets[k] = extract_save_labels(
            datasets[k], f'y_{k}', OUTPUT_DATA_PATH, sampling_period=labels_sampling_period,
            pre_sampling_period=args.pre_sampling_period
        )

    # resample periods before doing anything if specified and relevant
    data_downsampling = args.data_sampling_period != args.pre_sampling_period
    downsampling_pos = {
        p: data_downsampling and args.data_downsampling_position == p
        for p in CHOICES['build_features']['data_downsampling_position']
    }
    if downsampling_pos['first']:
        for k in datasets:
            datasets[k] = get_resampled(
                datasets[k], args.data_sampling_period, anomaly_col=False,
                pre_sampling_period=args.pre_sampling_period
            )
    # optional features alteration bundle
    if args.alter_bundles != '.':
        print(f'altering features using bundle #{args.alter_bundle_idx} of {args.alter_bundles}')
        datasets = get_altered_features(datasets, get_bundles_lists()[args.alter_bundles][args.alter_bundle_idx])

    # resample periods after alteration but before transformation if specified and relevant
    if downsampling_pos['middle']:
        for k in datasets:
            datasets[k] = get_resampled(
                datasets[k], args.data_sampling_period, anomaly_col=False,
                pre_sampling_period=args.pre_sampling_period
            )

    # turn datasets to `(n_periods, period_size, n_features)` ndarrays
    print('converting datasets to numpy arrays...', end=' ', flush=True)
    for k in datasets:
        datasets[k] = get_numpy_from_dfs(datasets[k])
    print('done.')

    # optional features transformation chain
    datasets_info = load_files(
        INPUT_DATA_PATH, [f'{n}_info' for n in PIPELINE_SET_NAMES], 'pickle', drop_info_suffix=True
    )
    for transform_step in [ts for ts in args.transform_chain.split('.') if len(ts) > 0]:
        args_text = ''
        if 'scaling' in transform_step:
            args_text = f'{args.scaling_method}_'
        if 'pca' in transform_step:
            args_text = f'{args.pca_n_components}_{args.pca_kernel}_'
        if 'fa' in transform_step:
            args_text = f'{args.fa_n_components}_'
        if 'head' in transform_step:
            args_text = f'{args.head_size}_{args_text}'
        print(f'applying `{args_text}{transform_step}` to period features...', end=' ', flush=True)
        transformer = transformation_classes[transform_step](args, OUTPUT_MODELS_PATH)
        datasets = transformer.fit_transform_datasets(datasets, datasets_info)
        print('done.')

    # resample periods after every transformations if specified and relevant
    if downsampling_pos['last']:
        for k in datasets:
            datasets[k] = get_numpy_from_dfs(
                get_resampled(
                    get_dfs_from_numpy(datasets[k], args.pre_sampling_period),
                    args.data_sampling_period, anomaly_col=False,
                    pre_sampling_period=args.pre_sampling_period
                )
            )

    # save periods with updated features
    save_files(OUTPUT_DATA_PATH, datasets, 'numpy')
