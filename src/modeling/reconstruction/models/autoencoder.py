"""Autoencoder building and compilation module.

Gathers functions for building and compiling autoencoder models.
"""
import os

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Reshape, Dropout
from tensorflow.keras.layers import RepeatVector, TimeDistributed
from tensorflow.keras.layers import Dense

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from modeling.helpers import PC


def build_autoencoder(window_size, n_features, latent_dim=10, type_='dense',
                      enc_n_hidden_neurons=None, dense_layers_activation='relu', dec_last_activation='linear',
                      dropout=0.0, rec_unit_type='lstm', rec_dropout=0.0):
    """Returns an autoencoder network with the specified architecture hyperparameters.

    Note: the decoder's architecture is always set to a mirrored version of the encoder.

    Args:
        window_size (int): size of input samples in number of records.
        n_features (int): number of input features.
        latent_dim (int): dimension of the latent vector representation (coding).
        type_ (str): type of autoencoder to build.
        enc_n_hidden_neurons (list): number of units for each hidden layer before the coding.
        dec_last_activation (str): last decoder layer activation (either `linear` or `sigmoid`).
        dropout (float): dropout rate for feed-forward layers.
        dense_layers_activation (str): intermediate layers activation for dense architectures.
        rec_unit_type (str): type of recurrent unit (either "lstm" or "gru").
        rec_dropout (float): recurrent dropout rate.

    Returns:
        keras.model: the autoencoder keras model.
    """
    if enc_n_hidden_neurons is None:
        enc_n_hidden_neurons = []
    a_t = 'only dense and recurrent autoencoders are supported for now'
    assert type_ in ['dense', 'rec'], a_t
    if type_ == 'dense':
        # encoder network
        encoder = Sequential(name='Encoder')
        encoder.add(Flatten(input_shape=(window_size, n_features)))
        for layer, n_neurons in enumerate(enc_n_hidden_neurons):
            encoder.add(Dense(n_neurons, activation=dense_layers_activation))
            encoder.add(Dropout(dropout))
        encoder.add(Dense(latent_dim, activation=dense_layers_activation))
        # decoder network
        decoder = Sequential(name='Decoder')
        for layer, n_neurons in enumerate(reversed(enc_n_hidden_neurons)):
            shape_arg = {'input_shape': [latent_dim]} if layer == 0 else {}
            decoder.add(Dense(n_neurons, activation=dense_layers_activation, **shape_arg))
            decoder.add(Dropout(dropout))
        decoder.add(Dense(window_size * n_features, activation=dec_last_activation))
        decoder.add(Reshape([window_size, n_features]))
    else:
        # encoder network
        encoder = Sequential(name='Encoder')
        for layer, n_neurons in enumerate(enc_n_hidden_neurons):
            shape_arg = {'input_shape': (window_size, n_features)} if layer == 0 else {}
            encoder.add(
                PC.units[rec_unit_type](
                    n_neurons, dropout=dropout, recurrent_dropout=rec_dropout,
                    return_sequences=True, **shape_arg
                )
            )
        encoder.add(
            PC.units[rec_unit_type](
                latent_dim, dropout=dropout, recurrent_dropout=rec_dropout
            )
        )
        # decoder network
        decoder = Sequential(name='Decoder')
        decoder.add(RepeatVector(window_size, input_shape=[latent_dim]))
        for layer, n_neurons in enumerate(reversed(enc_n_hidden_neurons)):
            decoder.add(
                PC.units[rec_unit_type](
                    n_neurons, dropout=dropout, recurrent_dropout=rec_dropout,
                    return_sequences=True
                )
            )
        decoder.add(TimeDistributed(Dense(n_features, activation=dec_last_activation)))
    return Sequential([encoder, decoder], name='Autoencoder')


def compile_autoencoder(model, loss, optimizer, learning_rate):
    """Compiles the autoencoder inplace using the specified optimization hyperparameters.

    Args:
        model (keras.model): autoencoder to compile as a keras model.
        loss (str): loss function to optimize (either `mse` or `bce`).
        optimizer (str): optimization algorithm used for training the network.
        learning_rate (float): learning rate used by the optimization algorithm.
    """
    optimizer = PC.opt[optimizer](lr=learning_rate)
    model.compile(loss=PC.loss[loss](), optimizer=optimizer)
