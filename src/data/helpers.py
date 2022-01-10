"""Data helpers module.

Gathers functions for loading and manipulating period DataFrames and numpy arrays.
"""
import os
import json
import pickle

import numpy as np
import pandas as pd


def is_string_type(str_, type_=float):
    """Returns True if `str_` can be converted to type `type_`, False otherwise.

    Args:
        str_ (str): input string whose type to test.
        type_ (type): standard python type (e.g. `int` or `float`).

    Returns:
        bool: True if `str_` can be converted to type `type_`, False otherwise.
    """
    try:
        type_(str_)
        return True
    except ValueError:
        return False


def get_resampled(period_dfs, sampling_period, agg='mean', anomaly_col=False):
    """Returns the period DataFrames resampled to `sampling_period` using the
        provided aggregation function.

    Args:
        period_dfs (list): the period DataFrames to resample.
        sampling_period (str): the new sampling period, as a valid argument to `pd.Timedelta`.
        agg (str): the aggregation function defining the resampling (e.g. `mean`, `median` or `max`).
        anomaly_col (bool): whether the provided DataFrames have an `Anomaly` column, to resample differently.

    Returns:
        list: the same periods with resampled records.
    """
    resampled_dfs, sampling_p = [], pd.Timedelta(sampling_period)
    feature_cols = [c for c in period_dfs[0].columns if c != 'Anomaly']
    print(f'resampling periods applying records `{agg}` every {sampling_period}...', end=' ', flush=True)
    for df in period_dfs:
        resampled_df = df[feature_cols].resample(sampling_p).agg(agg).ffill().bfill()
        if anomaly_col:
            # if no records during `sampling_p`, we choose to simply repeat the label of the last one here
            resampled_df['Anomaly'] = df['Anomaly'].resample(sampling_p).agg('max').ffill().bfill()
        resampled_dfs.append(resampled_df)
    print('done.')
    return resampled_dfs


def load_files(input_path, file_names, file_format, *,
               drop_info_suffix=False, drop_labels_prefix=False, parse_keys=True):
    """Loads and returns the provided `file_names` files from `input_path`.

    Args:
        input_path (str): path to load the files from.
        file_names (list): list of file names to load (without file extensions).
        file_format (str): format of the files to load, must be either `pickle`, `json` or `numpy`.
        drop_info_suffix (bool): if True, drop any `_info` string from the output dict keys.
        drop_labels_prefix (bool): if True, drop any `y_` string from the output dict keys.
        parse_keys (bool): if True and loading `json` format, parse keys to type `int` or `float` if relevant.

    Returns:
        dict: the loaded files of the form `{file_name: file_content}`.
    """
    a_t = 'supported formats only include `pickle`, `json` and `numpy`'
    assert file_format in ['pickle', 'json', 'numpy'], a_t
    if file_format == 'pickle':
        ext = 'pkl'
    else:
        ext = 'json' if file_format == 'json' else 'npy'
    files_dict = dict()
    print(f'loading {file_format} files from {input_path}')
    for fn in file_names:
        print(f'loading `{fn}.{ext}`...', end=' ', flush=True)
        if file_format == 'pickle':
            files_dict[fn] = pickle.load(open(os.path.join(input_path, f'{fn}.{ext}'), 'rb'))
        elif file_format == 'json':
            files_dict[fn] = json.load(open(os.path.join(input_path, f'{fn}.{ext}')))
            # parse keys to type float or int if specified and relevant
            if parse_keys:
                if all([is_string_type(k, float) for k in files_dict[fn]]):
                    if all([is_string_type(k, int) for k in files_dict[fn]]):
                        files_dict[fn] = {int(k): v for k, v in files_dict[fn].items()}
                    else:
                        files_dict[fn] = {float(k): v for k, v in files_dict[fn].items()}
        else:
            files_dict[fn] = np.load(os.path.join(input_path, f'{fn}.{ext}'), allow_pickle=True)
        print('done.')
    if not (drop_info_suffix or drop_labels_prefix):
        return files_dict
    if drop_info_suffix:
        files_dict = {k.replace('_info', ''): v for k, v in files_dict.items()}
    if drop_labels_prefix:
        return {k.replace('y_', ''): v for k, v in files_dict.items()}
    return files_dict


def save_files(output_path, files_dict, file_format):
    """Saves files from the provided `files_dict` to `output_path` in the relevant format.

    Args:
        output_path (str): path to save the files to.
        files_dict (dict): dictionary of the form `{file_name: file_content}` (file names without extensions).
        file_format (str): format of the files to save, must be either `pickle`, `json` or `numpy`.
    """
    a_t = 'supported formats only include `pickle`, `json` and `numpy`'
    assert file_format in ['pickle', 'json', 'numpy'], a_t
    if file_format == 'pickle':
        ext = 'pkl'
    else:
        ext = 'json' if file_format == 'json' else 'npy'
    print(f'saving {file_format} files to {output_path}')
    os.makedirs(output_path, exist_ok=True)
    for fn in files_dict:
        print(f'saving `{fn}.{ext}`...', end=' ', flush=True)
        if file_format == 'pickle':
            with open(os.path.join(output_path, f'{fn}.{ext}'), 'wb') as pickle_file:
                pickle.dump(files_dict[fn], pickle_file)
        elif file_format == 'json':
            with open(os.path.join(output_path, f'{fn}.{ext}'), 'w') as json_file:
                json.dump(files_dict[fn], json_file, separators=(',', ':'), sort_keys=True, indent=4)
        else:
            np.save(os.path.join(output_path, f'{fn}.{ext}'), files_dict[fn], allow_pickle=True)
        print('done.')


def load_mixed_formats(file_paths, file_names, file_formats):
    """Loads and returns `file_names` stored as `file_formats` files at `file_paths`.

    Args:
        file_paths (list): list of paths for each file name.
        file_names (list): list of file names to load (without file extensions).
        file_formats (list): list of file formats for each name, must be either `pickle`, `json` or `numpy`.

    Returns:
        dict: the loaded files, with as keys the file names.
    """
    assert len(file_names) == len(file_paths) == len(file_formats), 'the provided lists must be of same lengths'
    files = dict()
    for name, path, format_ in zip(file_names, file_paths, file_formats):
        files[name] = load_files(path, [name], format_)[name]
    return files


def load_datasets_data(input_path, info_path, dataset_names):
    """Returns the periods records, labels and information for the provided dataset names.

    Args:
        input_path (str): path from which to load the records and labels.
        info_path (str): path from which to load the periods information.
        dataset_names (list): list of dataset names.

    Returns:
        dict: the datasets data, with keys of the form `{n}`, `y_{n}` and `{n}_info` (`n` the dataset name).
    """
    file_names = [fn for n in dataset_names for fn in [n, f'y_{n}', f'{n}_info']]
    n_sets = len(dataset_names)
    file_paths, file_formats = n_sets * (2 * [input_path] + [info_path]), n_sets * (2 * ['numpy'] + ['pickle'])
    return load_mixed_formats(file_paths, file_names, file_formats)


def extract_save_labels(period_dfs, labels_file_name, output_path, sampling_period=None, pre_sampling_period=None):
    """Extracts and saves labels from the `Anomaly` columns of the provided period DataFrames.

    If labels are resampled before being saved, their indices are first reset to start from 0 second.
    We do this to keep them aligned with any subsequently resampled periods data (which would then be
    resampled after having been converted to numpy arrays, hence without their datetime index).

    Args:
        period_dfs (list): list of period pd.DataFrame.
        labels_file_name (str): name of the numpy labels file to save (without ".npy" extension).
        output_path (str): path to save the labels to, as a numpy array of shape `(n_periods, period_length)`.
            Where `period_length` depends on the period.
        sampling_period (str|None): if specified, period to resample the labels to before saving them
            (as a valid argument to `pd.Timedelta`).
        pre_sampling_period (str|None): original sampling period of the DataFrames.
            If None and `sampling_period` is specified, the original sampling period is inferred from
            the first two records of the first DataFrame.

    Returns:
        list: the input periods without their `Anomaly` columns.
    """
    new_periods, labels_list = [period_df.copy() for period_df in period_dfs], []
    sampling_p, pre_sampling_p = None, None
    if sampling_period is not None:
        sampling_p = pd.Timedelta(sampling_period)
        if pre_sampling_period is None:
            pre_sampling_p = pd.Timedelta(np.diff(period_dfs[0].index[:2])[0])
        else:
            pre_sampling_p = pd.Timedelta(pre_sampling_period)
    for i, period_df in enumerate(period_dfs):
        if sampling_period is None:
            labels_list.append(period_df['Anomaly'].values.astype(int))
        else:
            # resample labels (after resetting their indices to start from a round date)
            labels_series = period_df[['Anomaly']].set_index(
                pd.date_range('01-01-2000', periods=len(period_df), freq=pre_sampling_p)
            )['Anomaly']
            labels_list.append(labels_series.resample(sampling_p).agg('max').ffill().bfill().values.astype(int))
        new_periods[i].drop('Anomaly', axis=1, inplace=True)

    # save labels and return the periods without their `Anomaly` columns
    print(f'saving {labels_file_name} labels file...', end=' ', flush=True)
    np.save(os.path.join(output_path, f'{labels_file_name}.npy'), get_numpy_from_numpy_list(labels_list))
    print('done.')
    return new_periods


def get_numpy_from_numpy_list(numpy_list):
    """Returns the equivalent numpy array for the provided list of numpy arrays.

    Args:
        numpy_list (list): the list of numpy arrays to turn into a numpy array.

    Returns:
        ndarray: corresponding numpy array of shape `(list_length, ...)`. If the arrays
            in the list have different shapes, the final array is returned with dtype object.
    """
    # if the numpy list contains a single ndarray or all its ndarrays have the same shape
    if len(numpy_list) == 1 or len(set([a.shape for a in numpy_list])) == 1:
        # return with original data type
        return np.array(numpy_list)
    # else return with data type "object"
    return np.array(numpy_list, dtype=object)


def get_numpy_from_dfs(period_dfs):
    """Returns the equivalent numpy 3d-array for the provided list of period DataFrames.

    Args:
        period_dfs (list): the list of period DataFrames to turn into a numpy array.

    Returns:
        ndarray: corresponding numpy array of shape `(n_periods, period_size, n_features)`.
            Where `period_size` depends on the period.
    """
    return get_numpy_from_numpy_list([period_df.values for period_df in period_dfs])


def get_dfs_from_numpy(periods, sampling_period):
    """Returns the equivalent list of period DataFrames for the provided numpy 3d-array.

    Args:
        periods (ndarray): numpy array of shape `(n_periods, period_size, n_features)` (where
            `period_size` depends on the period), to turn into a list of DataFrames.
        sampling_period (str): time resolution to use for the output DataFrames (starting from Jan. 1st, 2000).

    Returns:
        list: corresponding list of DataFrames with their time resolution set to `sampling_period`.
    """
    return [pd.DataFrame(p, pd.date_range('01-01-2000', periods=len(p), freq=sampling_period)) for p in periods]


def get_aligned_shuffle(array_1, array_2=None):
    """Returns `array_1` and `array_2` randomly shuffled, preserving alignments between the array elements.

    If `array_2` is None, simply return shuffled `array_1`.
    If it is not, the provided arrays must have the same number of elements.

    Args:
        array_1 (ndarray): first array to shuffle.
        array_2 (ndarray|None): optional second array to shuffle accordingly.

    Returns:
        (ndarray, ndarray)|ndarray: the shuffled array(s).
    """
    assert array_2 is None or len(array_1) == len(array_2), 'arrays to shuffle must be have the same lengths'
    mask = np.random.permutation(len(array_1))
    if array_2 is None:
        return array_1[mask]
    return array_1[mask], array_2[mask]


def get_sliding_windows(array, window_size, window_step, include_remainder=False, dtype=np.float64):
    """Returns sliding windows of `window_size` elements extracted every `window_step` from `array`.

    Args:
        array (ndarray): input ndarray whose first axis will be used for extracting windows.
        window_size (int): number of elements of the windows to extract.
        window_step (int): step size between two adjacent windows to extract.
        include_remainder (bool): whether to include any remaining window, with a different step size.
        dtype (type): optional data type to enforce for the window elements (default np.float64).

    Returns:
        ndarray: windows array of shape `(n_windows, window_size, *array.shape[1:])`.
    """
    windows, start_idx = [], 0
    while start_idx <= array.shape[0] - window_size:
        windows.append(array[start_idx:(start_idx + window_size)])
        start_idx += window_step
    if include_remainder and start_idx - window_step + window_size != array.shape[0]:
        windows.append(array[-window_size:])
    return np.array(windows, dtype=dtype)


def get_nansum(array, **kwargs):
    """Returns the sum of array elements over a given axis treating NaNs as zero.

    Same as `np.nansum`, expect NaN is returned instead of zero if all array elements are NaN.

    Args:
        array (ndarray): array whose sum of elements to compute.
        **kwargs: optional keyword arguments to pass to `np.nansum`.

    Returns:
        float: the sum of array elements (or NaN if all NaNs).
    """
    if np.isnan(array).all():
        return np.nan
    return np.nansum(array, **kwargs)
