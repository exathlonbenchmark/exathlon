"""Module gathering various forecasting methods to compare.
"""
import os
import time
import argparse
from abc import abstractmethod

import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, TerminateOnNaN

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from utils.common import MODELING_TRAIN_NAME, MODELING_VAL_NAME, get_output_path
from modeling.helpers import EpochTimesCallback
from modeling.forecasting.models.rnn import build_rnn, compile_rnn
from modeling.forecasting.evaluation import get_mean_squared_error, get_mean_absolute_error
# try to import keras tuner if available in the environment
try:
    from keras_tuner import HyperModel
    from modeling.hp_tuners import BayesianTuner
    KT_AVAILABLE = True
except ModuleNotFoundError:
    print('No keras tuner in the current environment.')
    KT_AVAILABLE = False


class Forecaster:
    """Forecasting model base class. Performing `n_forward`-ahead forecasts from `n_back` records.

    Args:
        args (argparse.Namespace): parsed command-line arguments.
        output_path (str): path to save the model and training information to.
        model (keras_model|None): if not None, the forecaster will be initialized using the provided keras model.
    """
    def __init__(self, args, output_path, model=None):
        # number of records used to forecast and number of records to forecast ahead
        self.n_back = args.n_back
        self.n_forward = args.n_forward
        self.output_path = output_path
        self.model = model
        # model hyperparameters (used along with the timestamp as the model id)
        self.hp = dict()
        # model training epoch times
        self.epoch_times = []

    @classmethod
    def from_file(cls, args, model_root_path):
        """Returns a Forecaster object with its parameters initialized from an existing model file.

        Args:
            args (argparse.Namespace): parsed command-line arguments.
            model_root_path (str): root path to the keras model file (assumed named "model.h5").

        Returns:
            Forecaster: pre-initialized Forecaster object.
        """
        full_model_path = os.path.join(model_root_path, 'model.h5')
        print(f'loading forecasting model file {full_model_path}...', end=' ', flush=True)
        model = load_model(full_model_path)
        print('done.')
        return cls(args, '', model)

    @abstractmethod
    def fit(self, X_train, y_train, X_val, y_val):
        """Fits the forecasting model to `(X_train, y_train)`, validating on `(X_val, y_val)`.

        Args:
            X_train (ndarray): shape `(n_samples, n_back, n_features)`.
            y_train (ndarray): shape `(n_samples, n_features)` or `(n_samples, n_forward, n_features)`.
            X_val (ndarray): shape `(n_samples, n_back, n_features)`.
            y_val (ndarray): shape `(n_samples, n_features)` or `(n_samples, n_forward, n_features)`.
        """

    @abstractmethod
    def predict(self, X):
        """Returns the forecast values for the provided input batch.

        Args:
            X (ndarray): shape `(n_samples, n_back, n_features)`.

        Returns:
            ndarray: shape `(n_samples, n_features)` or `(n_samples, n_forward, n_features)`.
        """

    def evaluate(self, X, y):
        """Returns a dictionary with relevant metrics evaluating the forecasting of `y` given `X`.

        Args:
            X (ndarray): model input of shape `(n_samples, n_back, n_features)`.
            y (ndarray): labels of shape `(n_samples, n_features)` or `(n_samples, n_forward, n_features)`.

        Returns:
            dict: regression metrics of interest for evaluating the model's forecasting ability.
        """
        y_pred = self.predict(X)
        # return the MSE and MAE by default
        return {
            'mse': get_mean_squared_error(y, y_pred),
            'mae': get_mean_absolute_error(y, y_pred)
        }


class NaiveForecaster(Forecaster):
    """Naive forecaster class (non-parametric). The forecasts are just the last observed records.
    """
    def __init__(self, args, output_path, model=None):
        super().__init__(args, output_path, model)

    def fit(self, X_train, y_train, X_val, y_val):
        """This method does not require any fitting.
        """
        pass

    def predict(self, X):
        y_pred = []
        for sample in X:
            y_pred.append(
                 sample[-1, :] if self.n_forward == 1 else np.array([sample[-1, :] for _ in range(self.n_forward)])
            )
        return np.array(y_pred)


class RNN(Forecaster, HyperModel if KT_AVAILABLE else object):
    """Recurrent Neural Network-based forecaster.
    """
    def __init__(self, args, output_path, model=None):
        super().__init__(args, output_path, model)
        # architecture hyperparameters
        self.arch_hp = {
            # number of units for each recurrent layer before regression
            'n_hidden_neurons': args.rnn_n_hidden_neurons,
            'unit_type': args.rnn_unit_type,
            'dropout': args.rnn_dropout,
            'rec_dropout': args.rnn_rec_dropout,
        }
        # optimization hyperparameters
        self.opt_hp = {
            'optimizer': args.rnn_optimizer,
            'learning_rate': args.rnn_learning_rate,
        }
        # training hyperparameters
        self.train_hp = {
            'n_epochs': args.rnn_n_epochs,
            'batch_size': args.rnn_batch_size
        }
        # overall hyperparameters
        self.hp = dict(**self.arch_hp, **self.opt_hp, **self.train_hp)
        # placeholder for the number of features (needed for the hp tuning function)
        self.n_features = None
        # full command-line arguments updated and used for saving models when tuning hp
        self.args = args

    def build(self, hp):
        """Defines the hp names and scopes and returns the corresponding model.

        This method is required for all `HyperModel` objects and will be used
        by keras tuners.

        Args:
            hp (kerastuner.engine.hyperparameters.HyperParameters): API for defining
                hyperparameters.

        Returns:
            model: built and compiled RNN model in terms of the hyperparameters to tune.
        """
        # architecture hyperparameters
        n_layers = hp.Choice('n_layers', values=[1, 2])
        n_hidden_neurons = [hp.Int('n_units_1', min_value=32, max_value=160, step=16)]
        if n_layers == 2:
            n_units_1 = n_hidden_neurons[0]
            n_hidden_neurons.append(hp.Int('n_units_2', min_value=16, max_value=n_units_1, step=8))
        unit_type = hp.Choice('unit_type', values=['lstm', 'gru'])
        dropout = hp.Choice('dropout', values=[0.0, 0.25, 0.5])
        # optional - put recurrent dropout to zero to support GPU acceleration
        rec_dropout = dropout
        self.arch_hp = {
            'n_hidden_neurons': n_hidden_neurons,
            'unit_type': unit_type,
            'dropout': dropout,
            'rec_dropout': rec_dropout,
        }
        # optimization hyperparameters
        optimizer = hp.Choice('optimizer', values=['adam', 'nadam', 'rmsprop'])
        learning_rate = hp.Float('learning_rate', min_value=10**-6, max_value=0.1, sampling='log')
        self.opt_hp = {
            'optimizer': optimizer,
            'learning_rate': learning_rate
        }
        # model building and compilation
        model = build_rnn(self.n_back, self.n_features, **self.arch_hp)
        compile_rnn(model, **self.opt_hp)
        # overall hp and args update according to the trial's hyperparameters
        args_dict = vars(self.args)
        for k, v in dict(**self.arch_hp, **self.opt_hp).items():
            args_dict[f'rnn_{k}'] = v
            self.hp[k] = v
        self.args = argparse.Namespace(**args_dict)
        # root output path update for saving the trial's model
        self.output_path = get_output_path(self.args, 'train_model', 'model')
        return model

    def tune_hp(self, data):
        """Tunes the model's hyperparameters as defined in the `build` method.

        For now, the `BayesianOptimization` tuner is used with hardcoded maximum
        trials and number of instances per trial.

        Args:
            data (dict): training, validation and test (sequence, target) pairs at keys of
            the form `X|y_{modeling_set_name}`.
        """
        # number of records looked back and number of input features
        _, self.n_back, self.n_features = data[f'X_{MODELING_TRAIN_NAME}'].shape
        # paths to the tuning's results and tensorboard logs
        tuning_root = os.path.sep.join(self.output_path.split(os.path.sep)[:-1])
        tuning_path = os.path.join(tuning_root, 'rnn_tuning')
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
        X_train, y_train = data[f'X_{MODELING_TRAIN_NAME}'], data[f'y_{MODELING_TRAIN_NAME}']
        X_val, y_val = data[f'X_{MODELING_VAL_NAME}'], data[f'y_{MODELING_VAL_NAME}']
        n_epochs, batch_size = self.train_hp['n_epochs'], self.train_hp['batch_size']
        tuner.search(
            X_train, y_train, epochs=n_epochs, batch_size=batch_size, verbose=1,
            validation_data=(X_val, y_val), callbacks=[tensorboard, early_stopping, terminate_on_nan]
        )

    def fit(self, X_train, y_train, X_val, y_val):
        print('building and compiling RNN...', end=' ', flush=True)
        self.model = build_rnn(X_train.shape[1], X_train.shape[2], **self.arch_hp)
        compile_rnn(self.model, **self.opt_hp)
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
            X_train, y_train, epochs=n_epochs, batch_size=batch_size,
            verbose=1, validation_data=(X_val, y_val),
            callbacks=[tensorboard, checkpoint_a, checkpoint_b, early_stopping, epoch_times_cb]
        )

    def predict(self, X):
        return self.model.predict(X)


# dictionary gathering references to the defined forecasting methods
forecasting_classes = {
    'naive.forecasting': NaiveForecaster,
    'rnn': RNN,
}
