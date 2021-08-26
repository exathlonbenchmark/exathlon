"""Features building module.

Turns the raw period columns to the final features that will be used by the models.
If specified, this pipeline step can also resample the periods to a new sampling period.
"""
import os

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from utils.common import PIPELINE_SET_NAMES, parsers, get_output_path
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

    # resample periods before doing anything if specified and relevant
    downsampling = (args.sampling_period != args.pre_sampling_period)
    if downsampling and args.downsampling_position == 'first':
        for k in datasets:
            datasets[k] = get_resampled(datasets[k], args.sampling_period, anomaly_col=True)
    # optional features alteration bundle
    if args.alter_bundles != '.':
        print(f'altering features using bundle #{args.alter_bundle_idx} of {args.alter_bundles}')
        datasets = get_altered_features(datasets, get_bundles_lists()[args.alter_bundles][args.alter_bundle_idx])

    # resample periods after alteration but before transformation if specified and relevant
    if downsampling and args.downsampling_position == 'middle':
        for k in datasets:
            datasets[k] = get_resampled(datasets[k], args.sampling_period, anomaly_col=True)

    # turn `Anomaly` columns to `(n_periods, period_size,)` ndarrays, dropping them from the DataFrames
    # (`period_size` can depend on the period)
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
    # downsample the labels before saving them if downsampling should be applied at the end
    end_downsampling = (downsampling and args.downsampling_position == 'last')
    sampling_period = args.sampling_period if end_downsampling else None
    print(f'{"downsampling and " if end_downsampling else ""}saving labels to {OUTPUT_DATA_PATH}')
    for k in datasets:
        datasets[k] = extract_save_labels(
            datasets[k], f'y_{k}', OUTPUT_DATA_PATH,
            sampling_period=sampling_period, pre_sampling_period=args.pre_sampling_period
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

    # resample periods of all datasets according to the downsampling position if relevant
    if end_downsampling:
        for k in datasets:
            datasets[k] = get_numpy_from_dfs(
                get_resampled(
                    get_dfs_from_numpy(datasets[k], args.pre_sampling_period),
                    args.sampling_period, anomaly_col=False
                )
            )

    # save periods with updated features
    save_files(OUTPUT_DATA_PATH, datasets, 'numpy')
