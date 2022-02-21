"""Custom keras tuners module.

Gathers classes adding custom model saving and evaluation to existing keras tuners.
"""
import os
import time

from keras_tuner.tuners import BayesianOptimization

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from utils.common import get_args_string, CHOICES
from modeling.forecasting.evaluation import save_forecasting_evaluation
from modeling.reconstruction.evaluation import save_reconstruction_evaluation


class BayesianTuner(BayesianOptimization):
    """Custom BayesianOptimization tuner.

    Note: the following parameters appear as no longer available for the base class.
    - alpha (float|array): value added to the diagonal of the kernel matrix during fitting.
    - beta (float): balancing factor of exploration and exploitation.
        The larger it is, the more explorative it gets.

    We cannot use the `EpochTimesCallback` here: a new callback object is created
    for each trial execution, and we cannot access it.
    => The only way to access it would be to copy the implementation of `run_trial`
    in https://bit.ly/2XZHKeN and make it return the callbacks, which is not advised since
    it would override any future update of keras tuner, possibly breaking compatibility.

    Args:
        normality_modeler (HyperModel): normality modeling object, for now either
            a `Forecaster` or `Reconstructor`. Must also be a `HyperModel`.
        objective (str): name of model metric to minimize or maximize, e.g. 'val_accuracy'.
        max_trials (int): total number of trials (model configurations) to test at most.
            Note that the oracle may interrupt the search before max_trial models have
            been tested if the search space has been exhausted.
        data (dict): training, validation and test data at keys of the form
            `X|y_{modeling_set_name}`.
        comparison_path (str): path to the performance comparison spreadsheet.
        num_initial_points (int): the number of randomly generated samples as initial
            training data for Bayesian optimization.
        seed (int): random seed.
        hyperparameters (HyperParameters): can be used to override (or register in advance)
            hyperparameters in the search space.
        tune_new_entries (bool): whether hyperparameter entries that are requested
            by the hypermodel but that were not specified in hyperparameters should
            be added to the search space, or not.
            If not, then the default value for these parameters will be used.
        allow_new_entries (bool): whether the hypermodel is allowed to request
            hyperparameter entries not listed in `hyperparameters`.
        **kwargs: keyword arguments relevant to all `Tuner` subclasses.
            Please see the docstring for `Tuner`.
    """
    def __init__(self, normality_modeler, objective, max_trials, data, comparison_path,
                 num_initial_points=2, seed=None,
                 hyperparameters=None, tune_new_entries=True, allow_new_entries=True,
                 **kwargs):
        super().__init__(
            normality_modeler, objective, max_trials,
            num_initial_points=num_initial_points, seed=seed, hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries, allow_new_entries=allow_new_entries,
            **kwargs
        )
        self.normality_modeler = normality_modeler
        self.data = data
        self.comparison_path = comparison_path

    def on_trial_end(self, trial):
        """Callback function called after each trial run.

        Note: for the model saving and performance recording to work, the `args`
        and `output_path` attributes of `normality_modeler` have to be updated within
        its `build` method, called at the beginning of every trial.

        Args:
            trial (Trial): the `Trial` instance corresponding to the trial that was run.
        """
        # call the base behavior of the hook
        super().on_trial_end(trial)
        # only load and save the best model for the run if the loss did not end up as NaN
        if trial.best_step is not None:
            # load the best model corresponding to the trial's run
            trial_model = self.load_model(trial)
            # save model to the main path (one per hp set) and a backup path (one per run)
            hp_root = self.normality_modeler.output_path
            run_root = os.path.join(hp_root, time.strftime('%Y_%m_%d-%H_%M_%S'))
            for root_path in [hp_root, run_root]:
                print(f'saving best trial model to {root_path}...', end=' ', flush=True)
                os.makedirs(root_path, exist_ok=True)
                trial_model.save(os.path.join(root_path, 'model.h5'))
                print('done.')

            # save model's training, validation and test performance for comparison
            args = self.normality_modeler.args
            a_t = 'model type must be a supported forecasting or reconstruction-based method'
            assert args.model_type in CHOICES['train_model']['forecasting'] + \
                   CHOICES['train_model']['reconstruction'], a_t
            modeling_task = get_args_string(args, 'modeling_task')
            config_name = get_args_string(args, 'model') + f'_{time.strftime("run.%Y.%m.%d.%H.%M.%S")}'
            print(f'saving model performance to {self.comparison_path} at index {config_name}')
            # set the trial's model as the model used by the normality modeler
            self.normality_modeler.model = trial_model
            f = save_forecasting_evaluation if args.model_type in CHOICES['train_model']['forecasting'] \
                else save_reconstruction_evaluation
            f(self.data, self.normality_modeler, modeling_task, config_name, self.comparison_path)
