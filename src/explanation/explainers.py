"""Explanation discovery classes.
"""
import os
from abc import abstractmethod

import numpy as np


class Explainer:
    """Base explanation discovery class.

    Gathers common functionalities of all explanation discovery methods.

    To be evaluated by an EDEvaluator, explainers must provide their explanation as a dictionary
    with a key called "important_fts", providing a sequence of "important" explanatory features.

    Args:
        args (argparse.Namespace): parsed command-line arguments.
        output_path (str): path to save the explanation model and information to.
    """
    def __init__(self, args, output_path):
        self.output_path = output_path

    @abstractmethod
    def explain_sample(self, sample, sample_labels=None):
        """Returns the explanation and explanation time for the provided `sample`.

        Args:
            sample (ndarray): sample to explain of shape `(sample_length, n_features)`.
            sample_labels (ndarray|None): optional sample labels of shape `(sample_length,)`.

        Returns:
            dict, float: sample explanation dictionary and explanation time (in seconds).
        """

    def explain_each_sample(self, samples, samples_labels=None):
        """Returns the explanation and explanation time for each of the provided `samples`.

        Args:
            samples (ndarray): samples to explain of shape `(n_samples, sample_length, n_features)`.
                With `sample_length` possibly depending on the sample.
            samples_labels (ndarray): optional samples labels of shape `(n_samples, sample_length)`.
                With `sample_length` possibly depending on the sample.

        Returns:
            list, list: list of explanation dictionaries and corresponding times.
        """
        explanations, times = [], []
        if samples_labels is None:
            samples_labels = np.full(len(samples), None)
        for sample, sample_labels in zip(samples, samples_labels):
            explanation, time = self.explain_sample(sample, sample_labels)
            explanations.append(explanation)
            times.append(time)
        return explanations, times


# use a getter function to access references to ED classes to solve cross-import issues
def get_explanation_classes():
    """Returns a dictionary gathering references to the defined explanation discovery classes.
    """
    # add absolute src directory to python path to import other project modules
    import sys
    src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
    sys.path.append(src_path)
    from explanation.model_free.model_free_explainers import get_model_free_explanation_classes
    from explanation.model_dependent.model_dependent_explainers import get_model_dependent_explanation_classes
    return dict(get_model_free_explanation_classes(), **get_model_dependent_explanation_classes())
