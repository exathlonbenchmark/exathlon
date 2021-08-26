"""BiGAN architectures module.

Gathers functions for returning hardcoded architecture lists for encoder, generator
and discriminator networks.
"""
import os

import numpy as np
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    Input, Concatenate, Flatten, Reshape, Dropout, ReLU, LeakyReLU,
    Dense, Conv2D, Conv2DTranspose, Cropping2D, BatchNormalization,
    LSTM, RepeatVector, TimeDistributed
)
from tensorflow.keras.initializers import RandomNormal

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from modeling.reconstruction.models.helpers import get_shape_and_cropping


def get_encoder_architectures(window_size, n_features, latent_dim, type_):
    """Returns the hardcoded encoder architectures for the provided type.

    The first hardcoded convolutional architecture is from "Efficient
    GAN-based Anomaly Detection" (https://arxiv.org/pdf/1802.06222.pdf).
    => This architecture was used to detect anomalies in the MNIST dataset
    """
    model_name = 'Encoder'
    if type_ == 'conv':
        # default parameters
        kernel_init = RandomNormal(mean=0.0, stddev=0.02)
        leaky_alpha = 0.1
        return [
            Sequential([
                # first hidden layer
                Conv2D(
                    32, kernel_size=3, strides=1, padding='same',
                    kernel_initializer=kernel_init,
                    activation=LeakyReLU(alpha=leaky_alpha),
                    input_shape=[window_size, n_features, 1]
                ),
                # second hidden layer
                Conv2D(
                    64, kernel_size=3, strides=2, padding='same',
                    kernel_initializer=kernel_init
                ),
                BatchNormalization(),
                LeakyReLU(alpha=leaky_alpha),
                # third hidden layer
                Conv2D(
                    128, kernel_size=3, strides=2, padding='same',
                    kernel_initializer=kernel_init
                ),
                BatchNormalization(),
                LeakyReLU(alpha=leaky_alpha),
                # flattening layer
                Flatten(),
                # output layer
                Dense(latent_dim, kernel_initializer=kernel_init, activation='linear')
            ], name=model_name)
        ]
    if type_ == 'rec':
        return [
            # first architecture
            Sequential([
                LSTM(100, input_shape=[window_size, n_features]),
                Dense(latent_dim, activation='linear')
            ], name=model_name)
        ]
    return []


def get_generator_architectures(window_size, n_features, latent_dim, type_, last_activation):
    """Returns the hardcoded generator architectures for the provided type.

    The first hardcoded convolutional architecture is from "Efficient
    GAN-based Anomaly Detection" (https://arxiv.org/pdf/1802.06222.pdf).
    => This architecture was used to detect anomalies in the MNIST dataset, with
        pixels rescaled to [-1, 1], therefore using "tanh" for `last_activation`.
    """
    model_name = 'Generator'
    if type_ == 'conv':
        # default parameters, four-fold reduction shape and cropping
        kernel_init = RandomNormal(mean=0.0, stddev=0.02)
        reduced_shape, cropping = get_shape_and_cropping(window_size, n_features, 4.)
        reduced_shape = [*reduced_shape, 128]
        return [
            Sequential([
                # first hidden layer
                Dense(
                    1024, input_shape=[latent_dim], kernel_initializer=kernel_init
                ),
                BatchNormalization(),
                ReLU(),
                # second hidden layer
                Dense(np.prod(reduced_shape), kernel_initializer=kernel_init),
                BatchNormalization(),
                ReLU(),
                # reshaping layer
                Reshape(reduced_shape),
                # third hidden layer
                Conv2DTranspose(
                    64, kernel_size=4, strides=2, padding='same',
                    kernel_initializer=kernel_init
                ),
                BatchNormalization(),
                ReLU(),
                # output layer
                Conv2DTranspose(
                    1, kernel_size=4, strides=2, padding='same',
                    kernel_initializer=kernel_init, activation=last_activation
                ),
                Cropping2D(cropping=cropping)
            ], name=model_name)
        ]
    if type_ == 'rec':
        return [
            # first architecture
            Sequential([
                RepeatVector(window_size, input_shape=[latent_dim]),
                LSTM(100, return_sequences=True),
                TimeDistributed(Dense(n_features, activation=last_activation))
            ], name=model_name)
        ]
    return []


def get_discriminator_architectures(window_size, n_features, latent_dim, type_):
    """Returns the hardcoded discriminator architectures for the provided type.

    The first hardcoded architectures correspond to the ones tried out as part
    of the first set of experiments.

    The second hardcoded convolutional architecture is from "Efficient
    GAN-based Anomaly Detection" (https://arxiv.org/pdf/1802.06222.pdf).
    => This architecture was used to detect anomalies in the MNIST dataset.
    """
    model_name, architectures = 'Discriminator', []
    if type_ == 'rec':
        # FIRST RECURRENT ARCHITECTURE
        # inputs
        x_input = Input(shape=[window_size, n_features])
        z_input = Input(shape=[latent_dim])

        # path of x
        x_h1 = LSTM(100)(x_input)
        x_output = Dense(100, activation=LeakyReLU(alpha=0.2))(x_h1)

        # path of z
        z_output = z_input
        for n_neurons in [32, 32]:
            z_output = Dense(n_neurons, activation=LeakyReLU(alpha=0.2))(z_output)

        # common path
        concat = Concatenate()([x_output, z_output])
        output = Dense(1, activation='sigmoid')(concat)
        model = Model(inputs=[x_input, z_input], outputs=[output], name='Discriminator')
        architectures.append(model)

    if type_ == 'conv':
        # FIRST CONVOLUTIONAL ARCHITECTURE
        # inputs
        x_input = Input(shape=[window_size, n_features])
        z_input = Input(shape=[latent_dim])

        # path of x
        x_h1 = Conv2D(16, kernel_size=3, strides=1, padding='same')(
            Reshape((window_size, n_features, 1))(x_input)
        )
        x_h1_output = LeakyReLU()(BatchNormalization()(x_h1))
        x_h2 = Conv2D(16 * 2, kernel_size=3, strides=2, padding='same')(x_h1_output)
        x_h2_output = LeakyReLU()(BatchNormalization()(x_h2))
        x_h3 = Conv2D(16 * 4, kernel_size=3, strides=2, padding='same')(x_h2_output)
        x_h3_output = LeakyReLU()(BatchNormalization()(x_h3))
        x_h4 = Conv2D(16 * 4, kernel_size=3, strides=4, padding='same')(x_h3_output)
        x_h4_output = LeakyReLU(alpha=0.2)(BatchNormalization()(x_h4))
        x_output = Flatten()(x_h4_output)

        # path of z
        z_output = z_input
        for n_neurons in [32, 32]:
            z_output = Dense(n_neurons, activation=LeakyReLU(alpha=0.2))(z_output)

        # common path
        concat = Concatenate()([x_output, z_output])
        output = Dense(1, activation='sigmoid')(concat)
        model = Model(inputs=[x_input, z_input], outputs=[output], name='Discriminator')
        architectures.append(model)

        # SECOND CONVOLUTIONAL ARCHITECTURE
        kernel_init = RandomNormal(mean=0.0, stddev=0.02)
        leaky_alpha, dropout_rate = 0.1, 0.5

        # inputs
        x_input = Input(shape=[window_size, n_features])
        z_input = Input(shape=[latent_dim])

        # path of x
        # first hidden layer
        x_h1 = Conv2D(
            64, kernel_size=4, strides=2, padding='same',
            kernel_initializer=kernel_init, activation=LeakyReLU(alpha=leaky_alpha)
        )(Reshape((window_size, n_features, 1))(x_input))
        x_h1_drop = Dropout(dropout_rate)(x_h1)
        # second hidden layer
        x_h2 = Conv2D(
            64, kernel_size=4, strides=2, padding='same',
            kernel_initializer=kernel_init
        )(x_h1_drop)
        x_h2_norm = BatchNormalization()(x_h2)
        x_h2_norm_act = LeakyReLU(alpha=leaky_alpha)(x_h2_norm)
        x_h2_norm_act_drop = Dropout(dropout_rate)(x_h2_norm_act)
        # flattening layer
        x_h2_norm_act_drop_flat = Flatten()(x_h2_norm_act_drop)

        # path of z
        # first hidden layer
        z_h1 = Dense(
            512, kernel_initializer=kernel_init,
            activation=LeakyReLU(alpha=leaky_alpha)
        )(z_input)
        z_h1_drop = Dropout(dropout_rate)(z_h1)

        # common path
        # concatenation layer
        concat = Concatenate()([x_h2_norm_act_drop_flat, z_h1_drop])
        # first hidden layer
        common_h1 = Dense(1024, kernel_initializer=kernel_init, activation=LeakyReLU(alpha=leaky_alpha))(concat)
        common_h1_drop = Dropout(dropout_rate)(common_h1)
        # output layer
        output = Dense(1, kernel_initializer=kernel_init, activation='sigmoid')(common_h1_drop)
        model = Model(inputs=[x_input, z_input], outputs=[output], name='Discriminator')
        architectures.append(model)

        # THIRD CONVOLUTIONAL ARCHITECTURE
        # inputs
        x_input = Input(shape=[window_size, n_features])
        z_input = Input(shape=[latent_dim])

        # path of x
        x_h1 = Conv2D(
            64, kernel_size=3, strides=1, padding='same'
        )(Reshape((window_size, n_features, 1))(x_input))
        x_h1_output = LeakyReLU(0.01)(BatchNormalization()(x_h1))
        x_h2 = Conv2D(
            64 * 2, kernel_size=3, strides=2, padding='same'
        )(x_h1_output)
        x_h2_output = LeakyReLU(0.01)(BatchNormalization()(x_h2))
        x_h3 = Conv2D(
            64 * 3, kernel_size=3, strides=1, padding='same'
        )(x_h2_output)
        x_h3_output = LeakyReLU(0.01)(BatchNormalization()(x_h3))
        x_h4 = Conv2D(
            64 * 4, kernel_size=3, strides=2, padding='same'
        )(x_h3_output)
        x_h4_output = LeakyReLU(0.01)(BatchNormalization()(x_h4))
        x_output = Flatten()(x_h4_output)

        # path of z
        z_h1 = Dense(64, activation=LeakyReLU(alpha=0.2))(z_input)
        z_output = Dense(10, activation=LeakyReLU(alpha=0.2))(z_h1)

        # common path
        output = Dense(1, activation='sigmoid')(Concatenate()([x_output, z_output]))
        model = Model(inputs=[x_input, z_input], outputs=[output], name='Discriminator')
        architectures.append(model)
    return architectures
