"""BiGAN building and compilation module.

Gathers functions for building, chaining and compiling E, G and D networks.
"""
import os

import numpy as np
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    Input, Concatenate, Flatten, Reshape, Dropout, ReLU, LeakyReLU,
    Dense, Conv2D, Conv2DTranspose, Cropping2D, BatchNormalization,
    RepeatVector, TimeDistributed
)
# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from modeling.helpers import PC
from modeling.reconstruction.models.helpers import get_shape_and_cropping
from modeling.reconstruction.models.bigan_architectures import (
    get_encoder_architectures, get_generator_architectures, get_discriminator_architectures
)


def build_encoder(window_size, n_features, latent_dim, type_, arch_idx,
                  rec_n_hidden_neurons, rec_unit_type, conv_n_filters,
                  dropout, rec_dropout):
    """Returns an encoder network built from the specified architecture hyperparameters.

    If `arch_idx` is not -1, the returned architecture will be the hardcoded
    architecture at index `arch_idx` for the encoder type, and all parameters
    except `window_size`, `n_features` and `latent_dim` will be ignored.

    Args:
        window_size (int): size of input samples in number of records.
        n_features (int): number of input features.
        latent_dim (int): dimension of the latent vector representations.
        type_ (str): type of encoder ("rec" or "conv").
        arch_idx (int): if not -1, index of the hardcoded architecture to
            use in the provided type.
        rec_n_hidden_neurons (list): number of recurrent units for
            each hidden layer before the coding.
        rec_unit_type (str): recurrent unit type (simple RNN, LSTM or GRU).
        conv_n_filters (int): initial number of filters of the first
            convolutional layer. This number then typically doubles everytime
            spatial dimensions are halved.
        dropout (float): dropout rate for feed-forward layers.
        rec_dropout (float): recurrent dropout rate.

    Returns:
        keras.model: the encoder keras model.
    """
    a_text = 'only recurrent and convolutional encoders are supported for now'
    assert type_ in ['rec', 'conv'], a_text
    if arch_idx != -1:
        # return an hardcoded architecture, ignoring parameters
        return get_encoder_architectures(
            window_size, n_features, latent_dim, type_
        )[arch_idx]

    # return a custom architecture based on the relevant parameters
    if type_ == 'rec':
        model = Sequential(name='Encoder')
        n_hidden = len(rec_n_hidden_neurons)
        for layer, n_neurons in enumerate(rec_n_hidden_neurons):
            shape_arg = {'input_shape': (window_size, n_features)} if layer == 0 else {}
            model.add(
                PC.units[rec_unit_type](
                    n_neurons, dropout=dropout, recurrent_dropout=rec_dropout,
                    return_sequences=(layer != n_hidden - 1), **shape_arg
                )
            )
        # the produced latent vectors are not bounded
        model.add(Dense(latent_dim, activation='linear'))
        return model
    if type_ == 'conv':
        return Sequential([
            Reshape((window_size, n_features, 1), input_shape=[window_size, n_features]),
            Conv2D(
                conv_n_filters, kernel_size=3, strides=1, padding='same'
            ),
            BatchNormalization(),
            LeakyReLU(),
            Conv2D(
                conv_n_filters * 2, kernel_size=3, strides=2, padding='same',
            ),
            BatchNormalization(),
            LeakyReLU(),
            Conv2D(
                conv_n_filters * 4, kernel_size=3, strides=2, padding='same',
            ),
            BatchNormalization(),
            LeakyReLU(),
            Flatten(),
            Dense(latent_dim, activation='linear')
        ], name='Encoder')


def build_generator(window_size, n_features, latent_dim, type_, last_activation,
                    arch_idx, rec_n_hidden_neurons, rec_unit_type, conv_n_filters,
                    dropout, rec_dropout):
    """Returns a generator network built from the specified hyperparameters.

    If `arch_idx` is not -1, the returned architecture will be the hardcoded
    architecture at index `arch_idx` for the generator type, and all parameters
    except `window_size`, `n_features`, `latent_dim` and `last_activation` will be ignored.

    Returns:
        keras.model: the generator keras model.
    """
    a_text = 'only recurrent and convolutional generators are supported for now'
    assert type_ in ['rec', 'conv'], a_text
    if arch_idx != -1:
        # return an hardcoded architecture, ignoring parameters
        return get_generator_architectures(
            window_size, n_features, latent_dim, type_, last_activation
        )[arch_idx]

    # return a custom architecture based on the relevant parameters
    if type_ == 'rec':
        model = Sequential(name='Generator')
        model.add(RepeatVector(window_size, input_shape=[latent_dim]))
        for layer, n_neurons in enumerate(rec_n_hidden_neurons):
            model.add(
                PC.units[rec_unit_type](
                    n_neurons, dropout=dropout, recurrent_dropout=rec_dropout,
                    return_sequences=True
                )
            )
        model.add(TimeDistributed(Dense(n_features, activation=last_activation)))
        return model
    if type_ == 'conv':
        reduced_shape, cropping = get_shape_and_cropping(window_size, n_features, 4.)
        # add channel dimension to the reduced_shape
        reduced_shape = [*reduced_shape, 4. * conv_n_filters]
        return Sequential([
            Dense(1024, input_shape=[latent_dim]),
            BatchNormalization(),
            ReLU(),
            Dense(np.prod(reduced_shape)),
            BatchNormalization(),
            ReLU(),
            # reduced windows
            Reshape(reduced_shape),
            Conv2DTranspose(
                conv_n_filters * 2, kernel_size=4, strides=2, padding='same'
            ),
            BatchNormalization(),
            ReLU(),
            Conv2DTranspose(
                conv_n_filters, kernel_size=4, strides=2, padding='same'
            ),
            BatchNormalization(),
            ReLU(),
            # output layer
            Conv2DTranspose(
                1, kernel_size=4, strides=2, padding='same', activation=last_activation
            ),
            Cropping2D(cropping=cropping)
        ], name='Generator')


def build_discriminator(window_size, n_features, latent_dim, type_, arch_idx,
                        x_rec_n_hidden_neurons, x_rec_unit_type,
                        x_conv_n_filters, x_dropout, x_rec_dropout,
                        z_n_hidden_neurons, z_dropout):
    """Returns a discriminator network built from the specified architecture hyperparameters.

    If `arch_idx` is not -1, the returned architecture will be the hardcoded
    architecture at index `arch_idx` for the discriminator type, and all parameters
    except `window_size`, `n_features` and `latent_dim` will be ignored.

    Returns:
        keras.model: the discriminator keras model.
    """
    a_text = 'only recurrent and convolutional discriminators are supported for now'
    assert type_ in ['rec', 'conv'], a_text
    if arch_idx != -1:
        # return an hardcoded architecture, ignoring parameters
        return get_discriminator_architectures(
            window_size, n_features, latent_dim, type_
        )[arch_idx]

    # return a custom architecture considering the relevant parameters
    # the latent vector path is common to all discriminator types
    z_input = Input(shape=[latent_dim])
    z_output = z_input
    for layer, n_neurons in enumerate(z_n_hidden_neurons):
        z_output = Dense(n_neurons, activation=LeakyReLU(alpha=0.2))(z_output)
        z_output = Dropout(z_dropout)(z_output)
    if type_ == 'rec':
        # use the recurrent parameters to build a recurrent X path
        x_input = Input(shape=[window_size, n_features])
        x_output = x_input
        x_n_hidden = len(x_rec_n_hidden_neurons)
        for layer, n_neurons in enumerate(x_rec_n_hidden_neurons):
            x_output = PC.units[x_rec_unit_type](
                n_neurons, dropout=x_dropout, recurrent_dropout=x_rec_dropout,
                return_sequences=(layer != x_n_hidden - 1)
            )(x_output)
    else:
        # use the convolutional parameters to build a convolutional X path
        x_input = Input(shape=[window_size, n_features])
        x_h1 = Conv2D(
            x_conv_n_filters, kernel_size=3, strides=1, padding='same'
        )(Reshape((window_size, n_features, 1))(x_input))
        x_h1_output = LeakyReLU()(BatchNormalization()(x_h1))
        x_h2 = Conv2D(
            x_conv_n_filters * 2, kernel_size=3, strides=2, padding='same'
        )(x_h1_output)
        x_h2_output = LeakyReLU()(BatchNormalization()(x_h2))
        x_h3 = Conv2D(
            x_conv_n_filters * 4, kernel_size=3, strides=2, padding='same'
        )(x_h2_output)
        x_h3_output = LeakyReLU()(BatchNormalization()(x_h3))
        x_output = Flatten()(x_h3_output)
    # the common path is also independent from the discriminator type
    concat = Concatenate()([x_output, z_output])
    output = Dense(1, activation='sigmoid')(concat)
    return Model(inputs=[x_input, z_input], outputs=[output], name='Discriminator')


def build_bigan(encoder, generator, discriminator):
    """Returns a full "BiGAN" model from the provided E, G and D networks.

    Full "BiGAN" architecture (the D network is shared):

    |======================> |
    |                       |D| => real path's output
    X ==> |E| ==> z_real ==> |
    ####
    z ==> |G| ==> X_fake ==> |
    |                       |D| => fake path's output
    |======================> |

    The discriminator network has 2 inputs and 2 outputs:
    * The first output is for the "real path", taking a real image and its encoded
    vector as inputs, that E and G will try to make D classify as fake.
    * The second output is for the "fake path", taking a generated image and its
    corresponding random vector as inputs, that E and G will try to make D classify as real.

    Args:
        encoder (keras.model): encoder keras model.
        generator (keras.model): generator keras model.
        discriminator (keras.model): discriminator keras model.

    Returns:
        keras.model: the full BiGAN keras model.
    """
    # get window size, number of features and latent dimension from network inputs
    window_size, n_features = encoder.layers[0].input_shape[1:3]
    latent_dim = generator.layers[0].input_shape[1]
    # build real data path
    real_x = Input(shape=[window_size, n_features])
    real_z = encoder(real_x)
    # build fake data path
    fake_z = Input(shape=[latent_dim])
    fake_x = generator(fake_z)
    # create a discriminator output for each path
    dis_real_output = discriminator([real_x, real_z])
    dis_fake_output = discriminator([fake_x, fake_z])
    # the BiGAN model takes real windows and fake noise and outputs D's decision for each
    return Model([real_x, fake_z], [dis_real_output, dis_fake_output], name='BiGAN')


def build_encoder_decoder(encoder, generator):
    """Returns the encoder-decoder architecture chaining `encoder` and `generator`.
    """
    return Sequential([encoder, generator], name='Encoder-Decoder')


def compile_models(discriminator, bigan,
                   dis_optimizer, dis_learning_rate, enc_gen_optimizer, enc_gen_learning_rate):
    """Compiles the provided `discriminator` and `bigan` models inplace.

    Args:
        discriminator (keras.model): discriminator model to compile.
        bigan (keras.model): full bigan model to compile.
        dis_optimizer (str): optimization algorithm used for training D.
        dis_learning_rate (float): learning rate used by D's optimization algorithm.
        enc_gen_optimizer (str): optimization algorithm used for training E and G.
        enc_gen_learning_rate (float): learning rate used by E and G's optimization algorithm.
    """
    dis_optimizer = PC.opt[dis_optimizer](lr=dis_learning_rate)
    bigan_optimizer = PC.opt[enc_gen_optimizer](lr=enc_gen_learning_rate)
    discriminator.compile(loss='binary_crossentropy', optimizer=dis_optimizer, metrics=['acc'])
    discriminator.trainable = False
    bigan.compile(loss=['binary_crossentropy', 'binary_crossentropy'], optimizer=bigan_optimizer)
