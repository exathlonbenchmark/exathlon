"""Model building helpers module.

Gathers elements used for building and compiling keras models.
"""
import time
from argparse import Namespace

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU
from tensorflow.keras.optimizers import SGD, Adam, Nadam, RMSprop, Adadelta
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.callbacks import Callback


class EpochTimesCallback(Callback):
    """Keras callback recording each training epoch time.

    Args:
        epoch_times (list): reference to the list of epoch times.
            The times will be appended to this list inplace (emptying it first).
    """
    def __init__(self, epoch_times):
        super().__init__()
        self.epoch_times = epoch_times
        self.epoch_start = None

    def on_train_begin(self, logs=None):
        del self.epoch_times[:]

    def on_epoch_begin(self, batch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, batch, logs=None):
        self.epoch_times.append(time.time() - self.epoch_start)


def prepend_training_info(model, metrics, column_names):
    """Returns the metrics and column names with prepended training information.

    Here the training information contains:
    - Number of training epochs.
    - Average epoch time in seconds.
    - Total training time in seconds.

    Args:
        model (Forecaster|Reconstructor): the object used for modeling.
            Must have an `epoch_times` attribute storing the list of training epoch times.
        metrics (list): list of evaluation metrics to which to prepend the training info.
        column_names (list): list of corresponding metric names.

    Returns:
        list, list: the updated metrics and column names.
    """
    extended_metrics, extended_columns = metrics.copy(), column_names.copy()
    n_epochs = len(model.epoch_times)
    if n_epochs == 0:
        n_epochs, avg_epoch_time, train_time = np.nan, np.nan, np.nan
    else:
        train_time = sum(model.epoch_times)
        avg_epoch_time = train_time / n_epochs
    for column, metric in zip(
            ['TOT_TRAIN_TIME', 'AVG_EPOCH_TIME', 'N_EPOCHS'],
            [train_time, avg_epoch_time, n_epochs]
    ):
        extended_metrics = [metric] + extended_metrics
        extended_columns = [column] + extended_columns
    return extended_metrics, extended_columns


# keras classes corresponding to string parameters (i.e. "parameter classes")
PC = {
    'units': {
        'rnn': SimpleRNN,
        'lstm': LSTM,
        'gru': GRU
    },
    'opt': {
        'sgd': SGD,
        'adam': Adam,
        'nadam': Nadam,
        'rmsprop': RMSprop,
        'adadelta': Adadelta
    },
    'loss': {
        'bce': BinaryCrossentropy,
        'mse': MeanSquaredError
    }
}
PC = Namespace(**PC)


def get_test_batch_size():
    """Returns the batch size to use at test time to prevent memory overflow.

    This is especially relevant when using the memory of a GPU, which is usually more
    limited.

    Returns:
        int: the batch size to use at test time.
    """
    n_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
    return 128 if n_gpus > 0 else None
