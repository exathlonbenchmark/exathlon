"""Module gathering various reconstruction methods to compare.
"""
import os
import time
import argparse
from abc import abstractmethod

import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, TerminateOnNaN

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from utils.common import MODELING_TRAIN_NAME, MODELING_VAL_NAME, get_output_path
from modeling.helpers import EpochTimesCallback, get_test_batch_size
from modeling.reconstruction.models.autoencoder import build_autoencoder, compile_autoencoder
from modeling.reconstruction.models.bigan import (
    build_discriminator, build_encoder, build_generator,
    build_bigan, build_encoder_decoder, compile_models
)
from modeling.reconstruction.evaluation import (
    get_mean_squared_error, get_mean_absolute_error,
    get_discriminator_loss, get_feature_loss
)
# try to import keras tuner if available in the environment
try:
    from keras_tuner import HyperModel
    from modeling.hp_tuners import BayesianTuner
    KT_AVAILABLE = True
except ModuleNotFoundError:
    print('No keras tuner in the current environment.')
    KT_AVAILABLE = False


class Reconstructor:
    """Reconstruction model base class. Aiming to reconstruct `window_size`-windows.

    Args:
        args (argparse.Namespace): parsed command-line arguments.
        output_path (str): path to save the model and training information to.
        model (keras.model): optional reconstruction model initialization.
    """
    def __init__(self, args, output_path, model=None):
        # number of records of the windows to reconstruct
        self.window_size = args.window_size
        self.output_path = output_path
        self.model = model
        # model hyperparameters (used along with the timestamp as the model id)
        self.hp = dict()
        # model training epoch times
        self.epoch_times = []

    @classmethod
    def from_file(cls, args, model_root_path):
        """Returns a Reconstructor object with its parameters initialized from an existing model file.

        Args:
            args (argparse.Namespace): parsed command-line arguments.
            model_root_path (str): root path to the keras model file (assumed named "model.h5").

        Returns:
            Reconstructor: pre-initialized Reconstructor object.
        """
        full_model_path = os.path.join(model_root_path, 'model.h5')
        print(f'loading reconstruction model file {full_model_path}...', end=' ', flush=True)
        model = load_model(full_model_path, custom_objects={'LeakyReLU': LeakyReLU})
        print('done.')
        return cls(args, '', model)

    @abstractmethod
    def fit(self, X_train, X_val):
        """Fits the reconstruction model to `X_train` samples, validating on `X_val` samples.

        Args:
            X_train (ndarray): training samples of shape `(n_samples, window_size, n_features)`.
            X_val (ndarray): validation samples of shape shape `(n_samples, window_size, n_features)`.
        """

    @abstractmethod
    def reconstruct(self, X):
        """Returns the reconstructions of the samples of `X` by the model.

        Args:
            X (ndarray): samples to reconstruct of shape `(n_samples, window_size, n_features)`.

        Returns:
            ndarray: reconstructed samples of shape `(n_samples, window_size, n_features)`.
        """

    def evaluate(self, X):
        """Returns a dictionary with relevant metrics evaluating the reconstruction of `X` samples.

        This method centralizes every evaluation metrics that can be computed across
        the various reconstruction methods, including the ones that only apply to some.
        This enables comparing all methods with respect to the metrics they share.

        Args:
            X (ndarray): samples to reconstruct of shape `(n_samples, window_size, n_features)`.

        Returns:
            dict: metrics of interest for evaluating the model's reconstruction ability.
        """
        reconstructed = self.reconstruct(X)
        metrics = {
            # common reconstruction-based metrics
            'mse': get_mean_squared_error(X, reconstructed),
            'mae': get_mean_absolute_error(X, reconstructed),
            # additional discriminator-based metrics used by some subclasses
            'dis_loss': np.nan,
            'ft_loss': np.nan
        }
        self.complete_evaluation(metrics, X, reconstructed)
        return metrics

    def complete_evaluation(self, metrics, X, reconstructed):
        """Completes the evaluation metrics dictionary inplace.

        The metrics completion will typically involve computing some of the metrics filled
        with NaN in the class's `evaluate` method.

        Args:
            metrics (dict): metrics of interest for evaluating the model's reconstructions.
            X (ndarray): original samples of shape `(n_samples, window_size, n_features)`.
            reconstructed (ndarray): reconstructed samples of the same shape.
        """
        pass


class Autoencoder(Reconstructor, HyperModel if KT_AVAILABLE else object):
    """Autoencoder-based reconstructor.

    This method uses a stacked Autoencoder network to learn to reconstruct the data.
    """
    def __init__(self, args, output_path, model=None):
        super().__init__(args, output_path, model)
        # architecture hyperparameters
        self.arch_hp = {
            'latent_dim': args.ae_latent_dim,
            'type_': args.ae_type,
            # number of units for each encoder hidden layer before the coding
            'enc_n_hidden_neurons': args.ae_enc_n_hidden_neurons,
            # activation function for the last decoder layer
            'dec_last_activation': args.ae_dec_last_activation,
            'dropout': args.ae_dropout,
            'dense_layers_activation': args.ae_dense_layers_activation,
            'rec_unit_type': args.ae_rec_unit_type,
            'rec_dropout': args.ae_rec_dropout,
        }
        # optimization hyperparameters
        self.opt_hp = {
            'loss': args.ae_loss,
            'optimizer': args.ae_optimizer,
            'learning_rate': args.ae_learning_rate
        }
        # training hyperparameters
        self.train_hp = {
            'n_epochs': args.ae_n_epochs,
            'batch_size': args.ae_batch_size
        }
        # overall hyperparameters
        self.hp = dict(**self.arch_hp, **self.opt_hp, **self.train_hp)
        # placeholder for the number of features (needed for the hp tuning function)
        self.n_features = None
        # full command-line arguments updated and used for saving models when tuning hp
        self.args = args

    def build(self, hp):
        """Defines the hp names and scopes and return the model in terms of them.

        This method is required for all `HyperModel` objects and will be used
        by keras tuners.

        Args:
            hp (kerastuner.engine.hyperparameters.HyperParameters): API for defining
                hyperparameters.

        Returns:
            model: built and compiled AE model in terms of the hyperparameters to tune.
        """
        # parameters initialization (to be overridden by search spaces)
        rec_dropout, rec_unit_type = self.arch_hp['rec_dropout'], self.arch_hp['rec_unit_type']
        enc_n_hidden_neurons, dropout = self.arch_hp['enc_n_hidden_neurons'], self.arch_hp['dropout']
        optimizer, learning_rate = self.opt_hp['optimizer'], self.opt_hp['learning_rate']

        # dense autoencoders search space
        if self.arch_hp['type_'] == 'dense':
            # architecture hyperparameters
            n_layers = hp.Choice('n_layers', values=[1, 2])
            enc_n_hidden_neurons = [hp.Int('n_units_1', min_value=72, max_value=200, step=16)]
            if n_layers == 2:
                # number of units of layer #2 is always half of layer #1's if it exists
                enc_n_hidden_neurons.append(int(enc_n_hidden_neurons[0] / 2))
            dropout = hp.Choice('dropout', values=[0.0, 0.25, 0.5])
            # optimization hyperparameters
            optimizer = hp.Choice('optimizer', values=['adam', 'nadam'])
            learning_rate = hp.Float('learning_rate', min_value=10**-6, max_value=0.1, sampling='log')

        # recurrent autoencoders search space
        if self.arch_hp['type_'] == 'rec':
            # architecture hyperparameters
            n_layers = hp.Choice('n_layers', values=[1, 2])
            enc_n_hidden_neurons = [hp.Int('n_units_1', min_value=72, max_value=200, step=16)]
            if n_layers == 2:
                # number of units of layer #2 is always half of layer #1's if it exists
                enc_n_hidden_neurons.append(int(enc_n_hidden_neurons[0] / 2))
            rec_unit_type = hp.Choice('unit_type', values=['lstm', 'gru'])
            dropout = hp.Choice('dropout', values=[0.0, 0.25, 0.5])
            # optional - put recurrent dropout to zero to support GPU acceleration
            rec_dropout = dropout
            # optimization hyperparameters
            optimizer = hp.Choice('optimizer', values=['adam', 'nadam', 'rmsprop'])
            learning_rate = hp.Float('learning_rate', min_value=10**-6, max_value=0.1, sampling='log')

        # hyperparameters assignment
        self.arch_hp = {
            'latent_dim': self.arch_hp['latent_dim'],
            'type_': self.arch_hp['type_'],
            'enc_n_hidden_neurons': enc_n_hidden_neurons,
            'dec_last_activation': self.arch_hp['dec_last_activation'],
            'dropout': dropout,
            'rec_unit_type': rec_unit_type,
            'rec_dropout': rec_dropout,
        }
        self.opt_hp = {
            'loss': self.opt_hp['loss'],
            'optimizer': optimizer,
            'learning_rate': learning_rate
        }
        # model building and compilation
        model = build_autoencoder(self.window_size, self.n_features, **self.arch_hp)
        compile_autoencoder(model, **self.opt_hp)
        # overall hp and args update according to the trial's hyperparameters
        args_dict = vars(self.args)
        for k, v in dict(**self.arch_hp, **self.opt_hp).items():
            args_dict[f'ae_{k}'] = v
            self.hp[k] = v
        self.args = argparse.Namespace(**args_dict)
        # root output path update for saving the trial's model
        self.output_path = get_output_path(self.args, 'train_model', 'model')
        return model

    def tune_hp(self, data):
        """Tunes the model's hyperparameters defined in the `build` method.

        For now, the `BayesianOptimization` tuner is used with hardcoded maximum
        trials and number of instances per trial.

        Args:
            data (dict): training, validation and test windows at keys of the
                form `X_{modeling_set_name}`.
        """
        # window size and number of input features
        _, self.window_size, self.n_features = data[f'X_{MODELING_TRAIN_NAME}'].shape
        # paths to the tuning's results and tensorboard logs
        tuning_root = os.path.sep.join(self.output_path.split(os.path.sep)[:-1])
        tuning_path = os.path.join(tuning_root, f'{self.arch_hp["type_"]}_ae_tuning')
        tensorboard = TensorBoard(tuning_path)
        # path to the performance comparison spreadsheet for the task on the 3 datasets
        comparison_path = get_output_path(self.args, 'train_model', 'comparison')
        # we still use the early stopping callback for hyperparameters tuning
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=10, mode='min', restore_best_weights=True
        )
        # we add a `TerminateOnNaN` callback for hyperparameters tuning to skip bad configurations
        terminate_on_nan = TerminateOnNaN()
        tuner = BayesianTuner(
            self,
            data=data,
            comparison_path=comparison_path,
            objective='val_loss',
            # alternatively, set to a large number and stop after a given time
            max_trials=2,
            executions_per_trial=1,
            directory=tuning_path,
            project_name='tuning'
        )
        X_train, X_val = data[f'X_{MODELING_TRAIN_NAME}'], data[f'X_{MODELING_VAL_NAME}']
        n_epochs, batch_size = self.train_hp['n_epochs'], self.train_hp['batch_size']
        tuner.search(
            X_train, X_train, epochs=n_epochs, batch_size=batch_size, verbose=1,
            validation_data=(X_val, X_val), callbacks=[tensorboard, early_stopping, terminate_on_nan]
        )

    def fit(self, X_train, X_val):
        _, window_size, n_features = X_train.shape
        print('building and compiling autoencoder network...', end=' ', flush=True)
        self.model = build_autoencoder(window_size, n_features, **self.arch_hp)
        compile_autoencoder(self.model, **self.opt_hp)
        print('done.')
        # training and validation loss monitoring
        logging_path = os.path.join(self.output_path, time.strftime('%Y_%m_%d-%H_%M_%S'))
        tensorboard = TensorBoard(logging_path)
        # main checkpoint (one per hyperparameters set)
        checkpoint_a = ModelCheckpoint(os.path.join(self.output_path, 'model.h5'), save_best_only=True)
        # backup checkpoint (one per run instance)
        checkpoint_b = ModelCheckpoint(os.path.join(logging_path, 'model.h5'), save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
        # training epoch times monitoring
        epoch_times_cb = EpochTimesCallback(self.epoch_times)
        n_epochs, batch_size = self.train_hp['n_epochs'], self.train_hp['batch_size']
        self.model.fit(
            X_train, X_train, epochs=n_epochs,
            batch_size=batch_size, verbose=1, validation_data=(X_val, X_val),
            callbacks=[tensorboard, checkpoint_a, checkpoint_b, early_stopping, epoch_times_cb]
        )

    def reconstruct(self, X):
        return self.model.predict(X, batch_size=get_test_batch_size())


class BiGAN(Reconstructor):
    """BiGAN-based reconstructor class.

    This method uses a bidirectional GAN model to learn to reconstruct the data.
    => https://arxiv.org/pdf/1605.09782.pdf.

    A discriminator D is trained to distinguish real from fake (X, z) pairs,
    where X is a window and z its latent representation.
    An encoder (E : X -> z) and generator (G : z -> X) are jointly trained to fool
    D into believing that (G(z), z) pairs are real and (X, E(X)) pairs are fake.

    Args:
        args (argparse.Namespace): parsed command-line arguments.
        output_path (str): path to save the models and training information to.
        model (keras.model): optional reconstruction model initialization.
        bigan (keras.model): optional BiGAN model used to initialize E, G and D.
    """
    def __init__(self, args, output_path, model=None, bigan=None):
        super().__init__(args, output_path, model)
        # encoder, generator and discriminator networks
        self.encoder, self.generator, self.discriminator = None, None, None
        # initialize them from the loaded bigan network if provided
        if bigan is not None:
            self.encoder = bigan.get_layer('Encoder')
            self.generator = bigan.get_layer('Generator')
            self.discriminator = bigan.get_layer('Discriminator')
        # architecture hp (hardcoded architectures can be used within a network type)
        self.latent_dim = args.bigan_latent_dim
        self.arch_hp = {
            'encoder': {
                'type_': args.bigan_enc_type,
                'arch_idx': args.bigan_enc_arch_idx,
                'rec_n_hidden_neurons': args.bigan_enc_rec_n_hidden_neurons,
                'rec_unit_type': args.bigan_enc_rec_unit_type,
                'conv_n_filters': args.bigan_enc_conv_n_filters,
                'dropout': args.bigan_enc_dropout,
                'rec_dropout': args.bigan_enc_rec_dropout
            },
            'generator': {
                'type_': args.bigan_gen_type,
                # activation function for the last generator layer
                'last_activation': args.bigan_gen_last_activation,
                'arch_idx': args.bigan_gen_arch_idx,
                'rec_n_hidden_neurons': args.bigan_gen_rec_n_hidden_neurons,
                'rec_unit_type': args.bigan_gen_rec_unit_type,
                'conv_n_filters': args.bigan_gen_conv_n_filters,
                'dropout': args.bigan_gen_dropout,
                'rec_dropout': args.bigan_gen_rec_dropout
            },
            'discriminator': {
                'type_': args.bigan_dis_type,
                'arch_idx': args.bigan_dis_arch_idx,
                'x_rec_n_hidden_neurons': args.bigan_dis_x_rec_n_hidden_neurons,
                'x_rec_unit_type': args.bigan_dis_x_rec_unit_type,
                'x_conv_n_filters': args.bigan_dis_x_conv_n_filters,
                'x_dropout': args.bigan_dis_x_dropout,
                'x_rec_dropout': args.bigan_dis_x_rec_dropout,
                'z_n_hidden_neurons': args.bigan_dis_z_n_hidden_neurons,
                'z_dropout': args.bigan_dis_z_dropout
            }
        }
        flat_arch_hp = {'latent_dim': self.latent_dim}
        flat_arch_hp = dict(flat_arch_hp, **{
            f'{k[:3]}_{p}': v for k in self.arch_hp for p, v in self.arch_hp[k].items()
        })
        # optimization hyperparameters
        self.opt_hp = {
            'dis_optimizer': args.bigan_dis_optimizer,
            'dis_learning_rate': args.bigan_dis_learning_rate,
            'enc_gen_optimizer': args.bigan_enc_gen_optimizer,
            'enc_gen_learning_rate': args.bigan_enc_gen_learning_rate,
        }
        # training hyperparameters
        self.train_hp = {
            'n_epochs': args.bigan_n_epochs,
            'batch_size': args.bigan_batch_size,
            'dis_loss_threshold': args.bigan_dis_threshold
        }
        # overall hyperparameters
        self.hp = dict(flat_arch_hp, **self.opt_hp, **self.train_hp)

    @classmethod
    def from_file(cls, args, root_model_path):
        """Overrides the file loading method to also load the full BiGAN model.

        This model will further be used to initialize the E, G and D networks.
        """
        models = dict()
        print(f'loading BiGAN model files at {root_model_path}...', end=' ', flush=True)
        obj = {'LeakyReLU': LeakyReLU}
        for n in ['model', 'bigan']:
            models[n] = load_model(os.path.join(root_model_path, f'{n}.h5'), custom_objects=obj)
        print('done.')
        return cls(args, '', model=models['model'], bigan=models['bigan'])

    def complete_evaluation(self, metrics, X, reconstructed):
        """Completes evaluation metrics by computing the discriminator and feature losses.
        """
        # compute the latent representation of the original and reconstructed batch
        z = dict()
        for windows, k in zip([X, reconstructed], ['X', 'reconstructed']):
            z[k] = self.encoder.predict(windows)
        # add the discriminator loss
        metrics['dis_loss'] = get_discriminator_loss(
            reconstructed, z['reconstructed'], self.discriminator
        )
        # add the feature loss (to evaluate feature matching)
        metrics['ft_loss'] = get_feature_loss(
            X, z['X'], reconstructed, z['reconstructed'], self.discriminator
        )

    def fit(self, X_train, X_val):
        """Model fitting implementation.

        The discriminator network is trained to distinguish real from fake windows.
        The full "BiGAN" model is trained to make the discriminator believe that data
        going through its "real" path is fake and that data going through its "fake"
        path is real.

        Comments regarding the training loop:
        - The training batches shuffling will be re-performed for each epoch.
        - We always fetch one batch ahead to increase efficiency.
        - The `trainable` attribute is only explicitly set and reset to remove
            warnings, since it is only considered before compiling the models.
        """
        n_train, window_size, n_features = X_train.shape
        print('building encoder, generator and discriminator networks...', end=' ', flush=True)
        self.encoder = build_encoder(
            window_size, n_features, self.latent_dim, **self.arch_hp['encoder']
        )
        self.generator = build_generator(
            window_size, n_features, self.latent_dim, **self.arch_hp['generator']
        )
        self.discriminator = build_discriminator(
            window_size, n_features, self.latent_dim, **self.arch_hp['discriminator']
        )
        print('done.')
        print('building full BiGAN and encoder-decoder networks...', end=' ', flush=True)
        bigan = build_bigan(self.encoder, self.generator, self.discriminator)
        self.model = build_encoder_decoder(self.encoder, self.generator)
        print('done.')
        print('compiling discriminator and full BiGAN models...', end=' ', flush=True)
        compile_models(self.discriminator, bigan, **self.opt_hp)
        print('done.')

        # constitute shuffled training batches
        batch_size = self.train_hp['batch_size']
        dataset = tf.data.Dataset.from_tensor_slices(
            X_train.astype(np.float32)
        ).shuffle(n_train, seed=21)
        dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)

        # initialize recorded metrics and tensorboard logger
        logging_path = os.path.join(self.output_path, time.strftime('%Y_%m_%d-%H_%M_%S'))
        tensorboard = TensorBoard(log_dir=logging_path, histogram_freq=0, write_graph=True)
        tensorboard.set_model(bigan)
        metrics, min_val_mse = dict(), np.inf

        # main training loop
        dis_loss, latent_dim = np.inf, self.latent_dim
        for epoch in range(self.train_hp['n_epochs']):
            print(f'Epoch {epoch + 1}/{self.train_hp["n_epochs"]}')
            # average discriminator's accuracy for this epoch
            metrics['dis_acc'] = []
            # train models
            epoch_start = time.time()
            for batch in tqdm(dataset):
                # prevent D from being too confident using one-sided label smoothing
                one_labels, zero_labels = tf.constant([[0.9]] * batch_size), tf.constant([[0.]] * batch_size)
                # fake z and X
                fake_z = tf.random.normal(shape=[batch_size, latent_dim])
                fake_x = self.generator.predict(fake_z)
                # real z
                real_z = self.encoder.predict(batch)
                # discriminator training (only update if previous loss was above threshold)
                dis_batch = [tf.concat([batch, fake_x], axis=0), tf.concat([real_z, fake_z], axis=0)]
                dis_labels = tf.concat([one_labels, zero_labels], axis=0)
                if dis_loss > self.train_hp['dis_loss_threshold']:
                    self.discriminator.trainable = True
                    self.discriminator.train_on_batch(dis_batch, dis_labels)
                    self.discriminator.trainable = False
                # encoder and generator training
                bigan.train_on_batch([batch, fake_z], [zero_labels, one_labels])
                # get discriminator loss and append its batch accuracy
                dis_loss, dis_acc = self.discriminator.test_on_batch(dis_batch, dis_labels)
                metrics['dis_acc'].append(dis_acc)
            # compute and log epoch metrics
            self.epoch_times.append(time.time() - epoch_start)
            metrics['dis_acc'] = sum(metrics['dis_acc']) / len(metrics['dis_acc'])
            for i, X in enumerate([X_train, X_val]):
                set_metrics = self.evaluate(X)
                # add validation prefix to metrics computed on the validation set
                prefix = f'{MODELING_VAL_NAME}_' if i == 1 else ''
                for k in set_metrics:
                    metrics[f'{prefix}{k}'] = set_metrics[k]
            tensorboard.on_epoch_end(epoch, metrics)
            # save full BiGAN and encoder-decoder models if best validation MSE was reached
            if metrics[f'{MODELING_VAL_NAME}_mse'] < min_val_mse:
                min_val_mse = metrics[f'{MODELING_VAL_NAME}_mse']
                # main and backup checkpoints (one per set of hp and one per run instance)
                for path in [self.output_path, logging_path]:
                    for m, model_name in zip([self.model, bigan], ['model', 'bigan']):
                        m.save(os.path.join(path, f'{model_name}.h5'))

    def reconstruct(self, X):
        return self.model.predict(X, batch_size=get_test_batch_size())


# dictionary gathering references to the defined reconstruction methods
reconstruction_classes = {
    'ae': Autoencoder,
    'bigan': BiGAN
}
