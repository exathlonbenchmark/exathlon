"""Model-free explanation discovery classes.
"""
import os
from abc import abstractmethod

import numpy as np

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from explanation.explainers import Explainer
from explanation.model_free.helpers import get_split_sample


class ModelFreeExplainer(Explainer):
    """Base model-free explanation discovery class.

    Gathers common functionalities of all model-free explanation discovery methods.

    These methods try to explain differences between a set of normal records and a set of
    anomalous records, independently from any anomaly detection model.

    Args:
        args (argparse.Namespace): parsed command-line arguments.
        output_path (str): path to save the explanation model and information to.
    """
    def __init__(self, args, output_path):
        super().__init__(args, output_path)

    def explain_sample(self, sample, sample_labels=None):
        """Model-free implementation.

        Model-free methods try to explain differences between the normal and anomalous
        records of the provided sample, providing `sample_labels` is therefore mandatory.
        """
        assert sample_labels is not None, 'sample labels must be provided for model-free ED methods'
        return self.explain_split_sample(*get_split_sample(sample, sample_labels))

    @abstractmethod
    def explain_split_sample(self, normal_records, anomalous_records):
        """Returns the explanation and explanation time for the provided normal and anomalous records.

        Args:
            normal_records (ndarray): "normal" records of shape `(n_normal_records, n_features)`.
            anomalous_records: "anomalous" records of shape `(n_anomalous_records, n_features)`.

        Returns:
            dict, float: explanation dictionary and explanation time (in seconds).
        """

    def classify_sample(self, explanation, sample):
        """Returns binary predictions for the `sample` records using `explanation` as an AD rule.

        Args:
            explanation (dict): explanation dictionary used to classify the sample records.
            sample (ndarray): sample records to classify of shape `(sample_length, n_features)`.

        Returns:
            ndarray: binary predictions for the sample records of shape `(sample_length,)`.
        """
        return np.array([self.classify_record(explanation, record) for record in sample])

    @abstractmethod
    def classify_record(self, explanation, record):
        """Returns a binary prediction for `record` using `explanation` as an AD rule.

        Args:
            explanation (dict): explanation dictionary used to classify the record.
            record (ndarray): record to classify of shape `(n_features,)`.

        Returns:
            int: binary prediction for the record.
        """


# use a getter function to access references to model-free ED classes to solve cross-import issues
def get_model_free_explanation_classes():
    """Returns a dictionary gathering references to the defined model-free ED classes.
    """
    # add absolute src directory to python path to import other project modules
    from explanation.model_free.exstream.exstream import EXstream
    from explanation.model_free.macrobase import MacroBase
    return {'exstream': EXstream, 'macrobase': MacroBase}
