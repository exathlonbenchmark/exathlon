"""RNN building and compilation module.

Gathers functions for building and compiling recurrent neural network models.
"""
import os

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from modeling.helpers import PC


def build_rnn(n_back, n_features, n_hidden_neurons, unit_type, dropout, rec_dropout):
    """Returns the RNN model matching the specified architecture hyperparameters.

    Args:
        n_back (int): number of records used to predict ahead (input sequence length).
        n_features (int): number of features for each record in the sequence.
        n_hidden_neurons (list): number of units for each recurrent layer before regression.
        unit_type (str): hidden unit type (simple RNN, LSTM or GRU).
        dropout (float): dropout rate for feed-forward layers.
        rec_dropout (float): dropout rate for recurrent layers.

    Returns:
        keras.model: the RNN keras model.
    """
    model = Sequential()
    n_hidden = len(n_hidden_neurons)
    for layer, n_neurons in enumerate(n_hidden_neurons):
        shape_arg = {'input_shape': (n_back, n_features)} if layer == 0 else {}
        model.add(PC.units[unit_type](
            n_neurons, dropout=dropout, recurrent_dropout=rec_dropout,
            return_sequences=(layer != n_hidden - 1), **shape_arg)
        )
    model.add(Dense(n_features, activation='linear'))
    return model


def compile_rnn(model, optimizer, learning_rate):
    """Compiles the provided RNN inplace using the specified optimization hyperparameters.

    Args:
        model (keras.model): keras RNN model to compile.
        optimizer (str): optimization algorithm used for training the RNN.
        learning_rate (float): learning rate used by the optimization algorithm.
    """
    model.compile(loss='mse', optimizer=PC.opt[optimizer](lr=learning_rate))
