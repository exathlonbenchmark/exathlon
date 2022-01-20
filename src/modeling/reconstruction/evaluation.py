"""Reconstruction evaluation module.
"""
import os

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.losses import mean_squared_error, mean_absolute_error
from tensorflow.keras.losses import binary_crossentropy

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from utils.common import MODELING_SET_NAMES
from modeling.helpers import prepend_training_info, get_test_batch_size


def save_reconstruction_evaluation(data_dict, reconstructor, modeling_task_string, config_name, spreadsheet_path):
    """Adds the reconstruction evaluation of this configuration to a sorted comparison spreadsheet.

    Args:
        data_dict (dict): train, val and test windows, as `{X_(modeling_set_name): ndarray}`.
            ndarrays of shapes `(n_windows, window_size, n_features).
        reconstructor (Reconstructor): the Reconstructor object to evaluate.
        modeling_task_string (str): formatted modeling task arguments to compare models under the same task.
        config_name (str): unique configuration identifier serving as an index in the spreadsheet.
        spreadsheet_path (str): comparison spreadsheet path.

    Returns:
        pd.DataFrame: the 1-row evaluation DataFrame holding the computed metrics.
    """
    # set the full path for the comparison spreadsheet
    full_spreadsheet_path = os.path.join(spreadsheet_path, f'{modeling_task_string}_comparison.csv')

    # compute and add reconstruction metrics
    column_names, metrics = [], []
    for set_name in MODELING_SET_NAMES:
        print(f'evaluating reconstruction metrics on the {set_name} set...', end=' ', flush=True)
        set_metrics = reconstructor.evaluate(data_dict[f'X_{set_name}'])
        for metric_name in set_metrics:
            metrics.append(set_metrics[metric_name])
            column_names.append((set_name + '_' + metric_name).upper())
        print('done.')
    # prepend training time information
    metrics, column_names = prepend_training_info(reconstructor, metrics, column_names)
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


def get_mean_squared_error(X, reconstructed, sample_wise=False):
    """Returns the coordinate-wise MSE of the model reconstructions.

    Args:
        X (ndarray): original samples of shape `(n_samples, window_size, n_features)`.
        reconstructed (ndarray): reconstructed samples of the same shape.
        sample_wise (bool): return sample-wise MSE if True, else return average batch MSE.

    Returns:
        ndarray|float: the MSE score for each sample or the average MSE across the batch.
    """
    sample_mse = np.mean(mean_squared_error(X, reconstructed).numpy(), axis=1)
    return sample_mse if sample_wise else np.mean(sample_mse)


def get_mean_absolute_error(X, reconstructed, sample_wise=False):
    """Returns the coordinate-wise MAE of the model reconstructions."""
    sample_mae = np.mean(mean_absolute_error(X, reconstructed).numpy(), axis=1)
    return sample_mae if sample_wise else np.mean(sample_mae)


def get_discriminator_loss(rec_x, rec_z, discriminator, sample_wise=False):
    """Returns the discriminator loss of the model reconstructions.

    This loss measures how wrong would the discriminator be in classifying the provided
    data as real.
    => The higher it is, the more the data is likely to come from another distribution
    than the one seen in training, and hence to be abnormal.

    Args:
        rec_x (ndarray): batch of reconstructed windows produced by an encoder-decoder model.
        rec_z (ndarray): corresponding latent representations of the reconstructed windows.
        discriminator (keras.model): discriminator network as a keras model.
        sample_wise (bool): return sample-wise losses if True, else return average batch loss.

    Returns:
        float: average discriminator loss across the provided batch.
    """
    batch_size = get_test_batch_size()
    preds = discriminator.predict([rec_x, rec_z], batch_size=batch_size)
    labels = tf.constant([[1.]] * rec_x.shape[0])
    sample_dis_loss = binary_crossentropy(labels, preds).numpy()
    return sample_dis_loss if sample_wise else np.mean(sample_dis_loss)


def get_feature_loss(x, z, rec_x, rec_z, discriminator, sample_wise=False):
    """Returns the feature loss of the model reconstructions.

    This loss is inversely proportional to the feature matching of the original
    and reconstructed windows.
    => The higher it is, the less their features match. Such features, having been extracted
    by the discriminator network, were found relevant to distinguish real from fake windows
    and should hence capture the essence of what constitutes a normal window.

    Args:
        x (ndarray): batch of original windows.
        z (ndarray): latent representation of original windows.
        rec_x (ndarray): batch of reconstructed windows.
        rec_z (ndarray): latent representation of reconstructed windows.
        discriminator (keras.model): discriminator network as a keras model.
        sample_wise (bool): return sample-wise losses if True, else return average batch loss.

    Returns:
        float: average feature loss across the provided batch.
    """
    batch_size = get_test_batch_size()
    dis_features = Model(inputs=discriminator.input, outputs=discriminator.layers[-2].output)
    real_features = dis_features.predict([x, z], batch_size=batch_size)
    rec_features = dis_features.predict([rec_x, rec_z], batch_size=batch_size)
    sample_ft_loss = np.sum(np.abs(real_features - rec_features), axis=1)
    return sample_ft_loss if sample_wise else np.mean(sample_ft_loss)
