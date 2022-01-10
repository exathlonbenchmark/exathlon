"""LIME explanation discovery module.
"""
import os
import re
import time
import warnings

from lime import lime_tabular

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from explanation.model_dependent.model_dependent_explainers import ModelDependentExplainer


class LIME(ModelDependentExplainer):
    """Local Interpretable Model-agnostic Explanations (LIME) explanation discovery class.

    For this explainer, the `fit` method must be called before explaining samples.

    See https://arxiv.org/pdf/1602.04938.pdf for more details.
    """
    def __init__(self, args, output_path, ad_model):
        super().__init__(args, output_path, ad_model)
        # number of features to report in the explanations
        self.n_features = args.lime_n_features
        # LIME model
        self.lime_model = None

    def fit(self, training_samples):
        """Initializes and fits the LIME model to the provided `training_samples`.

        We use the implementation of the LIME package with continuous features discretized into
        deciles. `RecurrentTabularExplainer` is used to accommodate to the samples shape.

        Args:
            training_samples (ndarray): training samples of shape `(n_samples, sample_length, n_features)`.
        """
        # feature names must be consistent with the way we identify features in `explain_sample`
        feature_names = [f'ft_{i}' for i in range(training_samples[0].shape[1])]
        self.lime_model = lime_tabular.RecurrentTabularExplainer(
            training_samples, mode='regression', feature_names=feature_names,
            discretize_continuous=True, discretizer='decile'
        )

    def explain_sample(self, sample, sample_labels=None):
        """LIME implementation.

        An "explanation" is defined as a set of relevant features, it is derived from
        the LIME model. Sample labels are not relevant in this case.
        """
        start_t = time.time()
        try:
            explanation = self.lime_model.explain_instance(
                sample, self.ad_scoring_func, num_features=self.n_features
            )
        except AttributeError:
            print('No model initialization, please call the `fit` method before using this explainer')
            return {'important_fts': []}, 0
        # important feature indices are extracted from the feature names reported in the explanation
        important_fts = []
        for statement, weight in explanation.as_list():
            # "ft_{id}" names were used to identify numbers that relate to features in the strings
            ft_name = re.findall(r'ft_\d+', statement)[0]
            ft_index = int(ft_name[3:])
            important_fts.append(ft_index)
        # a same feature can occur in multiple statements
        returned_fts = sorted(list(set(important_fts)))
        if len(returned_fts) == 0:
            warnings.warn('No explanation found for the sample.')
        return {'important_fts': returned_fts}, time.time() - start_t
