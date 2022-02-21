"""Model-dependent explanation discovery classes.
"""
import os
from abc import abstractmethod

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from scoring.forecasting.forecasting_scorers import ForecastingScorer
from scoring.reconstruction.reconstruction_scorers import ReconstructionScorer
from explanation.explainers import Explainer


class ModelDependentExplainer(Explainer):
    """Base model-dependent explanation discovery class.

    Gathers common functionalities of all model-dependent explanation discovery methods.

    These methods try to explain predictions of an anomaly detection model, assumed
    to be of the form `sample -> outlier score`, where `sample` is of fixed length.

    Args:
        args (argparse.Namespace): parsed command-line arguments.
        output_path (str): path to save the explanation model and information to.
        ad_model (Scorer): AD model whose (outlier score) predictions to explain.
    """
    def __init__(self, args, output_path, ad_model):
        super().__init__(args, output_path)
        a_t = 'supported AD models to explain only include forecasting or reconstruction scorers'
        assert isinstance(ad_model, ForecastingScorer) or isinstance(ad_model, ReconstructionScorer), a_t
        if isinstance(ad_model, ForecastingScorer):
            self.sample_length = ad_model.normality_model.n_back + ad_model.normality_model.n_forward
        else:
            self.sample_length = ad_model.normality_model.window_size
        # prediction (i.e., outlier scoring) function of the AD model to explain
        self.ad_scoring_func = ad_model.score_windows

    def explain_all_samples(self, samples):
        """Returns the explanation and explanation time for all the provided `samples` together.

        We define the shared explanation of a set of samples as the (duplicate-free) union
        of the "important" explanatory features found for each sample.

        Indeed, for a single sample, these important features constitute those that affected
        the outlier score function the most locally. Taking the union of these features hence
        gives us a sense of the features that were found most relevant throughout the samples.

        Args:
            samples (ndarray): samples to explain of shape `(n_samples, sample_length, n_features)`.

        Returns:
            dict, float: explanation dictionary and explanation time (in seconds) for the samples.
        """
        samples_explanation, samples_time = {'important_fts': set()}, 0
        for sample in samples:
            explanation, time = self.explain_sample(sample)
            samples_explanation['important_fts'] = samples_explanation['important_fts'].union(
                explanation['important_fts']
            )
            samples_time += time
        return samples_explanation, samples_time

    @abstractmethod
    def explain_sample(self, sample, sample_labels=None):
        """Model-dependent implementation.
        """


# use a getter function to access references to model-dependent ED classes to solve cross-import issues
def get_model_dependent_explanation_classes():
    """Returns a dictionary gathering references to the defined model-dependent ED classes.
    """
    # add absolute src directory to python path to import other project modules
    from explanation.model_dependent.lime_method import LIME
    return {'lime': LIME}
