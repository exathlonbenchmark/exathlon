"""Explanation discovery metrics definition module.

We separate the notion of "evaluation instance" from the one of "explained sample":

- Evaluation instances are used to compute the ED metrics for anomaly instances.
- Explained samples are the samples directly explained by an ED method, through an explanation
    function of the form "sample -> explanation".

In practice, evaluation instances and explained samples can either be the same or not. When
computing ED metrics, explained samples are typically "extracted" from evaluation instances.
"""
import os
import random
import warnings
from math import floor, ceil
from abc import abstractmethod

import numpy as np
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from data.helpers import get_numpy_from_numpy_list, get_sliding_windows, get_nansum
from metrics.ad_evaluators import extract_binary_ranges_ids
from explanation.model_free.helpers import get_split_sample, get_merged_sample
from explanation.model_free.model_free_explainers import ModelFreeExplainer
from explanation.model_dependent.model_dependent_explainers import ModelDependentExplainer


class NoInstanceError(Exception):
    """Exception raised in case no evaluation instance could be extracted from the data.

    Args:
        message (str): description of the error.
    """
    def __init__(self, message):
        self.message = message


class EDEvaluator:
    """Explanation discovery evaluation class.

    Defines the base class for computing relevant ED metrics, including:

    - Proportion of "covered" instances (i.e., having sufficient length to be considered
        by the ED method / in the evaluation).
    - Among the covered instances, proportion of "explained" instances (i.e., for which the ED method
        found important features). Only those instances will be considered when computing other metrics.
    - Average inference (i.e., explanation) time.
    - ED1/2 conciseness.
    - ED1/2 normalized consistency.
    - For the methods supporting it, ED1/2 accuracy (i.e., point-based Precision, Recall and F1-score).

    * Proportion of covered/explained instances, inference time, ED1 consistency and ED1 accuracy
        are returned as dictionaries whose keys are:
        - `global`, considering all non-zero labels as a single positive anomaly class.
        - every distinct non-zero label encountered in the data, considering the corresponding class only.
        - `avg`, corresponding to the average scores across each positive class (excluding the `global` key).
    * ED2 metrics are returned as similar dictionaries, but without the `global` key.
        ED2 metrics are indeed only defined across anomalies of the same type.
    * ED1 conciseness is returned as a single value, corresponding to the `global` key only.
        Considering ED1 conciseness per class would indeed be redundant with ED2 conciseness.

    Args:
        args (argparse.Namespace): parsed command-line arguments.
        explainer (explanation.explainers.Explainer): explainer object used to derive sample explanations.
    """
    def __init__(self, args, explainer):
        self.explainer = explainer
        # minimum anomaly length for an instance to be considered in the evaluation
        self.min_anomaly_length = args.ed_eval_min_anomaly_length
        # number of "disturbances" to perform when computing ED1 consistency
        self.ed1_consistency_n_disturbances = args.ed1_consistency_n_disturbances
        # dictionary gathering references to the defined ED1 metric computation functions
        self.ed1_functions = {
            'conciseness': self.compute_conciseness,
            'consistency': self.compute_ed1_consistency,
            'accuracy': self.compute_ed1_accuracy
        }

    def compute_metrics(self, periods, periods_labels, periods_info, used_data, include_ed2=True):
        """Returns the relevant ED metrics and explanations for the provided `periods` and `periods_labels`.

        Metrics are returned as a single dictionary with as keys the metric names and as
        values the metric values, as described in the class documentation.

        Explanations are returned as a single dictionary whose format depends on the used data.
        For spark data, explanations are grouped by file name, and labeled as
        `{instance_idx}_T{instance_type}`, where `instance_idx` is the chronological rank of the
        corresponding instance in the file, and `instance_type` is its numerical anomaly type.

        Args:
            periods (ndarray): periods data of shape `(n_periods, period_length, n_features)`.
                With `period_length` depending on the period.
            periods_labels (ndarray): labels for each period of shape `(n_periods, period_length)`.
                With `period_length` depending on the period.
            periods_info (list): list of periods information.
            used_data (str): used data, used to interpret the periods information.
            include_ed2 (bool): whether to include ED2 metrics in the computation. We might
                want to ignore those metrics when anomaly types are unknown.

        Returns:
            dict, dict: relevant ED metrics at keys `prop_covered`, `prop_explained`, `time` and `ed{i}_{name}`;
                with `i` either 1 or 2 and `name` in {"conciseness", "norm_consistency", "precision",
                "recall", "f1_score"}, along with instances explanations.
        """
        # ED metrics initialization
        shared_m_names = ['prop_covered', 'prop_explained', 'time']
        common_ed_m_names = ['conciseness', 'norm_consistency']
        acc_m_names = ['precision', 'recall', 'f1_score']
        acc_metrics = {f'ed{i}_{m}': dict() for i in [1, 2] for m in acc_m_names}
        ed_metrics = dict(
            {m: dict() for m in shared_m_names},
            **{f'ed{i}_{m}': dict() for i in [1, 2] for m in common_ed_m_names},
            **acc_metrics
        )
        if not include_ed2:
            # only keep ED2 conciseness, since used to compute ED1 conciseness
            ed_metrics = {k: v for k, v in ed_metrics.items() if 'ed2' not in k or 'conciseness' in k}

        # instances explanations per anomaly type
        explanations_dict = dict()
        # evaluation instances, with corresponding labels and information per anomaly type
        instances_dict, instances_labels_dict, instances_info_dict, ed_metrics['prop_covered'] = \
            self.get_evaluation_instances(periods, periods_labels, periods_info)

        for type_, type_instances in instances_dict.items():
            print(f'computing metrics for instances of type {type_}...', end=' ', flush=True)
            # corresponding labels for the type instances
            type_instances_labels = instances_labels_dict[type_]

            # type instances explanations, average inference time and proportion of explained instances
            explanations_dict[type_], ed_metrics['time'][type_], ed_metrics['prop_explained'][type_] = \
                self.explain_instances(type_instances, type_instances_labels)

            if ed_metrics['prop_explained'][type_] == 0:
                # no instance could be explained for the whole type
                for m in ed_metrics.keys():
                    if m not in shared_m_names:
                        ed_metrics[m][type_] = np.nan
            else:
                # compute ED1 metrics, considering one instance at a time
                ed_metrics['ed1_norm_consistency'][type_] = self.compute_ed1_metric(
                    'consistency', {
                        'instance': type_instances, 'instance_labels': type_instances_labels,
                        'explanation': explanations_dict[type_]
                    }, {'normalized': True}
                )
                ed_metrics['ed1_precision'][type_], ed_metrics['ed1_recall'][type_], \
                    ed_metrics['ed1_f1_score'][type_] = self.compute_ed1_metric(
                        'accuracy', {'instance': type_instances, 'instance_labels': type_instances_labels}
                    )

                # compute ED2 metrics, considering instances of the same type together
                ed_metrics['ed2_conciseness'][type_] = self.compute_ed1_metric(
                    'conciseness', {'explanation': explanations_dict[type_]}
                )
                if include_ed2:
                    ed_metrics['ed2_norm_consistency'][type_] = self.compute_ed2_consistency(
                        explanations_dict[type_], normalized=True
                    )
                    ed_metrics['ed2_precision'][type_], ed_metrics['ed2_recall'][type_], \
                        ed_metrics['ed2_f1_score'][type_] = self.compute_ed2_accuracy(
                            explanations_dict[type_], type_instances, type_instances_labels
                        )
            print('done.')

        # average metrics across anomaly types
        print('averaging metrics across anomaly types...', end=' ', flush=True)
        for m_name in [m for m in ed_metrics if m != 'ed1_conciseness']:
            if m_name in acc_metrics and isinstance(self, ModelDependentEvaluator):
                # set to NaN without warning if accuracy metrics are all NaN for model-dependent methods
                ed_metrics[m_name]['avg'] = np.nan
            else:
                # will set to NaN with a warning if all values are NaN for a metric here
                ed_metrics[m_name]['avg'] = np.nanmean([ed_metrics[m_name][k] for k in ed_metrics[m_name]])
        print('done.')

        # deduce metrics "globally" (i.e., considering all anomaly types the same)
        print('deriving "global" metrics...', end=' ', flush=True)
        # number of covered and explained instances
        n_cov_instances_dict, n_explained_instances_dict = dict(), dict()
        for k, v in explanations_dict.items():
            n_cov_instances_dict[k], n_explained_instances_dict[k] = len(instances_dict[k]), len(v)
        n_cov_instances = sum(n_cov_instances_dict.values())
        n_explained_instances = sum(n_explained_instances_dict.values())
        # this "global" consideration does not makes sense for ED2 metrics
        for m_name in [m for m in ed_metrics if 'ed2' not in m]:
            # ED1 conciseness is only "global"
            if m_name == 'ed1_conciseness':
                if n_explained_instances > 0:
                    ed_metrics[m_name] = get_nansum([
                        n_explained_instances_dict[k] * ed_metrics['ed2_conciseness'][k] for k in instances_dict
                    ]) / n_explained_instances
                else:
                    # case of no explained instances at all
                    ed_metrics[m_name] = np.nan
            # it corresponds to the "global" key for other metrics
            else:
                # only consider explained instances for all metrics except the proportion of explained instances
                if m_name != 'prop_explained':
                    dict_, total = n_explained_instances_dict, n_explained_instances
                # for the proportion of explained instances, consider the number of covered instances
                else:
                    dict_, total = n_cov_instances_dict, n_cov_instances
                if total > 0:
                    ed_metrics[m_name]['global'] = get_nansum([
                        dict_[k] * ed_metrics[m_name][k] for k in dict_
                    ]) / total
                else:
                    # case of no explained instances at all
                    ed_metrics[m_name]['global'] = 0 if m_name == 'prop_covered' else np.nan
        print('done.')

        if not include_ed2:
            # remove ED2 conciseness (only used to computed ED1 conciseness)
            return {k: v for k, v in ed_metrics.items() if 'ed2_conciseness' not in k}

        # format type-wise explanations to include the instances information
        formatted_explanations_dict = dict()
        for type_ in explanations_dict:
            for i in range(len(explanations_dict[type_])):
                explanation, instance_info = explanations_dict[type_][i], instances_info_dict[type_][i]
                # the chronological rank of the instance in the period is always last
                instance_idx = instance_info[-1]
                if used_data == 'spark':
                    # for spark data, the first element of the period information is the file name
                    file_name = instance_info[0]
                    # group explanations by file name
                    if file_name not in formatted_explanations_dict:
                        formatted_explanations_dict[file_name] = dict()
                    # within a file, label each explanation with its instance rank and type
                    formatted_explanations_dict[file_name][f'{instance_idx}_T{type_}'] = explanation

        # return ED metrics and formatted explanations
        return ed_metrics, formatted_explanations_dict

    def get_evaluation_instances(self, periods, periods_labels, periods_info):
        """Returns evaluation instances, labels and information from the provided `periods`, `periods_labels`
            and `periods_info`.

        Raises `NoInstanceError` if no evaluation instance could be extracted from the
        provided data. Else, provides along with the instances and labels the proportion
        of anomalies that will be covered in the evaluation (grouped by type).

        Instances information is returned as lists of the form `[*period_info, instance_idx]`, where
        `period_info` is the information of the period the instance belongs to, and `instance_idx` is
        the chronological rank of the instance in that period.

        Args:
            periods (ndarray): periods records of shape `(n_periods, period_length, n_features)`.
                With `period_length` depending on the period.
            periods_labels (ndarray): multiclass periods labels of shape `(n_periods, period_length)`.
                With `period_length` depending on the period.
            periods_info (list): list of periods information.

        Returns:
            dict, dict, dict, dict: instances, corresponding labels and information, as well as proportions
                of anomalies covered per anomaly type, with as keys the relevant (numerical) anomaly types and
                as values ndarrays/lists/floats.
        """
        pos_classes = np.delete(np.unique(np.concatenate(periods_labels, axis=0)), 0)
        n_ranges_dict, prop_covered_dict = dict(), dict()
        instances_dict, instances_labels_dict, instances_info_dict = dict(), dict(), dict()
        for pc in pos_classes:
            n_ranges_dict[pc] = 0
            instances_dict[pc], instances_labels_dict[pc], instances_info_dict[pc] = [], [], []

        for period, period_labels, period_info in zip(periods, periods_labels, periods_info):
            # instances, labels and information grouped by anomaly type for the period
            period_pos_ranges = extract_binary_ranges_ids((period_labels > 0).astype(int))
            period_range_classes = np.array([period_labels[pos_range[0]] for pos_range in period_pos_ranges])
            for range_idx in range(len(period_pos_ranges)):
                instance, instance_labels = self.get_evaluation_instance(
                    period, period_labels, period_pos_ranges, range_idx
                )
                range_class = period_range_classes[range_idx]
                n_ranges_dict[range_class] += 1
                if instance is not None:
                    instances_dict[range_class].append(instance)
                    instances_labels_dict[range_class].append(instance_labels)
                    instances_info_dict[range_class].append([*period_info, range_idx])
        # turn data and labels lists to numpy arrays and return the results
        for pc in instances_dict:
            instances_dict[pc] = get_numpy_from_numpy_list(instances_dict[pc])
            instances_labels_dict[pc] = get_numpy_from_numpy_list(instances_labels_dict[pc])
            # proportion of positive ranges covered in the evaluation
            prop_covered_dict[pc] = instances_dict[pc].shape[0] / n_ranges_dict[pc]
        # raise error if no evaluation instance could be extracted
        if np.all([prop_covered == 0 for prop_covered in prop_covered_dict.values()]):
            raise NoInstanceError('No evaluation instance could be extracted from the periods.')
        return instances_dict, instances_labels_dict, instances_info_dict, prop_covered_dict

    @abstractmethod
    def get_evaluation_instance(self, period, period_labels, pos_ranges, range_idx):
        """Returns the evaluation instance and labels corresponding to the anomaly range
            `pos_ranges[range_idx]` in `period`.

        If the evaluation instance cannot be extracted, due to the positive range violating
        a given constraint (e.g., minimum length), then None values should be returned.

        Args:
            period (ndarray): period records of shape `(period_length, n_features)`.
            period_labels (ndarray): corresponding multiclass labels of shape `(period_length,)`.
            pos_ranges (ndarray): start and (excluded) end indices of every anomaly range in the
                period, of the form `[[start_1, end_1], [start_2, end_2], ...]`.
            range_idx (int): index in `pos_ranges` of the anomaly range for which to return the
                instance and labels.

        Returns:
            ndarray, ndarray: evaluation instance and labels corresponding to the anomaly range,
                of respective shapes `(instance_length, n_features)` and `(instance_length,)`.
        """
        pass

    def explain_instances(self, instances, instances_labels):
        """Returns the explanations of the provided `instances` along with the average
            explanation time and proportion of explained instances.

        If an instance could not be explained by the ED method (i.e., no "important features" were
        found for it), it will be ignored when computing ED metrics, but reflected in a metric called
        "proportion of explained instances".

        Args:
            instances (ndarray): evaluation instances of shape `(n_instances, instance_length, n_features)`.
                With `instance_length` depending on the instance.
            instances_labels (ndarray): instances labels of shape `(n_instances, instance_length)`.
                With `instance_length` depending on the instance.

        Returns:
            list, float, float: instances explanations (as a list of dicts), average explanation time,
                and proportion of instances explained by the ED method.
        """
        explanations, times_sum = [], 0
        for instance, instance_labels in zip(instances, instances_labels):
            explanation, explanation_time = self.explain_instance(instance, instance_labels)
            if len(explanation['important_fts']) > 0:
                explanations.append(explanation)
                times_sum += explanation_time
        n_explained = len(explanations)
        if n_explained > 0:
            avg_time = times_sum / n_explained
            prop_explained = n_explained / len(instances)
        else:
            avg_time, prop_explained = 0, 0
        return explanations, avg_time, prop_explained

    @abstractmethod
    def explain_instance(self, instance, instance_labels):
        """Returns the explanation of the provided `instance` along with the explanation time.

        According to the type of evaluated method and instance definition, the explanation
        of an *instance* might be derived from the explanations of *samples* in various ways.

        Args:
            instance (ndarray): evaluation instance of shape `(instance_length, n_features)`.
            instance_labels (ndarray): instance labels of shape `(instance_length,)`.

        Returns:
            dict, float: instance explanation and explanation time.
        """

    def compute_ed1_metric(self, metric_name, instances_args_dict, func_args_dict=None):
        """Returns the ED1 `metric_name`.

        Every ED1 metrics share the fact they are computed on each considered instance
        separately and then returned averaged.

        Args:
            metric_name (str): name of the ED1 metric to compute. Must be either `conciseness`,
                `consistency` or `accuracy`.
            instances_args_dict (dict): arguments to loop through when iterating through instances,
                of the form `{arg_key: arg_values_list}`, with value `i` corresponding to instance `i`.
            func_args_dict (dict|None): optional function arguments keeping the same values
                when iterating through instances, of the form `{arg_key: arg_value}`.

        Returns:
            float|list: single value for `conciseness` and `consistency`, list of values for `accuracy`,
                corresponding to the precision, recall and f1-score, respectively.
        """
        a_t = 'the provided metric name must be either `conciseness`, `consistency` or `accuracy`'
        assert metric_name in ['conciseness', 'consistency', 'accuracy'], a_t
        func_kwargs = dict() if func_args_dict is None else func_args_dict
        # compute the relevant metric per instance
        instance_metrics = []
        n_instance_args, instance_args_keys = len(instances_args_dict), list(instances_args_dict.keys())
        for instance_args_values in zip(*instances_args_dict.values()):
            # the orders of `keys` and `values` always correspond if the dict was not altered
            instance_kwargs = {instance_args_keys[i]: instance_args_values[i] for i in range(n_instance_args)}
            instance_metrics.append(
                self.ed1_functions[metric_name](**instance_kwargs, **func_kwargs)
            )
        # return average metric(s) across instances (multivalued metrics should be returned as ndarrays)
        if isinstance(instance_metrics[0], np.ndarray):
            instance_metrics = get_numpy_from_numpy_list(instance_metrics)
        return sum(instance_metrics) / len(instance_metrics)

    @staticmethod
    def compute_conciseness(explanation):
        """Returns the conciseness of the provided `explanation`.

        Args:
            explanation (dict): explanation dictionary, assumed to provide an array-like of
                "important" explanatory features at key "important_fts".

        Returns:
            int: the conciseness value, defined as the number of important features.
        """
        return len(explanation['important_fts'])

    def compute_ed1_consistency(self, instance, instance_labels, explanation=None, normalized=True):
        """Returns the ED1 consistency (i.e., stability) score of the provided `instance`.

        This metric is defined as the (possibly normalized) consistency of explanatory
        features across different "disturbances" of the explained instance.

        Args:
            instance (ndarray): instance data of shape `(instance_length, n_features)`.
            instance_labels (ndarray): instance labels of shape `(instance_length,)`.
            explanation (dict|None): optional instance explanation, that can be used to normalize
                the consistency score of the instance.
            normalized (bool): whether to "normalize" the consistency score with respect to the
                original explanation's conciseness (if True, `explanation` must be provided).

        Returns:
            float: the ED1 consistency score of the instance.
        """
        if normalized:
            a_t = 'normalizing consistency requires providing the instance explanation'
            assert explanation is not None, a_t
        # consistency of explanatory features across different instance "disturbances"
        fts_consistency = self.compute_features_consistency(
            self.get_disturbances_features(instance, instance_labels, explanation)
        )
        if not normalized:
            return fts_consistency
        return (2 ** fts_consistency) / len(explanation['important_fts'])

    @abstractmethod
    def get_disturbances_features(self, instance, instance_labels, explanation=None):
        """Returns the explanatory features found for different "disturbances" of `instance`.

        The number of disturbances to perform is given by the value of
        `self.ed1_consistency_n_disturbances`, which might include the original instance or not.

        Args:
            instance (ndarray): instance data of shape `(instance_length, n_features)`.
            instance_labels (ndarray): instance labels of shape `(instance_length,)`.
            explanation (dict|None): optional, pre-computed, explanation of `instance`.

        Returns:
            list: the list of explanatory features lists for each instance "disturbance".
        """

    @staticmethod
    def compute_features_consistency(features_lists):
        """Returns the "consistency" of the provided features lists.

        This "consistency" aims to capture the degree of agreement between the features lists.
        We define it as the entropy of the lists' duplicate-preserving union (i.e., turning the
        features lists into a features bag).

        The smaller the entropy value, the less uncertain will be the outcome of randomly
        drawing an explanatory feature from the bag, and hence the more the lists of features
        will agree with each other.

        Args:
            features_lists (list): list of features lists whose consistency to compute.

        Returns:
            float: consistency of the features lists.
        """
        features_bag = [ft for features_list in features_lists for ft in features_list]
        # unnormalized probability distribution of feature ids
        p_features = []
        for feature_id in set(features_bag):
            p_features.append(features_bag.count(feature_id))
        return entropy(p_features, base=2)

    @abstractmethod
    def compute_ed1_accuracy(self, instance, instance_labels):
        """Returns the ED1 accuracy metrics of the provided `instance`.

        These metrics aim to capture the local predictive power of the explanations derived by the
        ED method. This predictive power is measured as the point-wise Precision, Recall and F1-score
        achieved by explanations when used as anomaly detection rules around the explained anomaly.

        Args:
            instance (ndarray): instance data of shape `(instance_length, n_features)`.
            instance_labels (ndarray): instance labels of shape `(instance_length,)`.

        Returns:
            ndarray: Precision, Recall and F1-score for the instance, respectively.
        """

    def compute_ed2_consistency(self, explanations, normalized=True):
        """Returns the ED2 consistency (i.e., concordance) score of the provided `explanations`.

        This metric is defined as the (possibly normalized) consistency of explanatory features across
        the provided explanations, it is therefore only defined if multiple explanations are provided.

        Args:
            explanations (list): list of instance explanations dicts whose consistency to compute.
            normalized (bool): whether to "normalize" the consistency score with respect to the
                average explanation conciseness.

        Returns:
            float: the ED2 consistency score of the explanations (or NaN if a single explanation
                was provided).
        """
        if len(explanations) == 1:
            return np.nan
        explanations_fts = [explanation['important_fts'] for explanation in explanations]
        fts_consistency = self.compute_features_consistency(explanations_fts)
        if not normalized:
            return fts_consistency
        avg_explanation_length = sum([len(fts) for fts in explanations_fts]) / len(explanations)
        return (2 ** fts_consistency) / avg_explanation_length

    @abstractmethod
    def compute_ed2_accuracy(self, explanations, instances, instances_labels):
        """Returns the ED2 accuracy metrics for the provided `explanations` and `instances`.

        These metrics aim to capture the "global" predictive power of the explanations derived
        by the ED method. This predictive power is measured as the point-wise Precision, Recall and
        F1-score achieved by explanations when used as anomaly detection rules around other anomalies
        of the same type.

        Like ED2 consistency, this metric is only defined across multiple evaluated instances.

        Args:
            explanations (list): list of instance explanations dicts.
            instances (ndarray): corresponding instances, assumed to be of the same anomaly type,
                of shape `(n_instances, instance_length, n_features)`. With `instance_length`
                depending on the instance.
            instances_labels (ndarray): instances labels of shape `(n_instances, instance_length)`.
                With `instance_length` depending on the instance.

        Returns:
            ndarray: Precision, Recall and F1-score, respectively (or NaNs if a single
                instance was provided).
        """


class ModelFreeEvaluator(EDEvaluator):
    """Model-free evaluation.

    Evaluates model-free explanation discovery methods, trying to explain differences
    between a set of normal records and a set of anomalous records.

    For such methods, evaluation instances are the same as explained samples: data intervals
    comprised of a "normal" interval followed by an "anomalous" interval: [N; A].

    The explanation of an instance/sample is defined by the explanation discovery method,
    and must contain a set of "explanatory" or "important" feature indices.
    """
    def __init__(self, args, explainer):
        super().__init__(args, explainer)
        # check parameters
        a_t = 'the provided explanation discovery method must be model-free'
        assert isinstance(explainer, ModelFreeExplainer), a_t

        # a minimum length of 2 points per sample part is needed for computing ED1 consistency and accuracy
        self.min_anomaly_length = max(2, self.min_anomaly_length)
        self.min_normal_length = max(2, args.mf_eval_min_normal_length)

        # proportion of records to sample when computing ED1 consistency
        self.ed1_consistency_sampled_prop = args.mf_ed1_consistency_sampled_prop

        # number of random splits and proportion of records to use as test when computing ED1 accuracy
        self.ed1_accuracy_n_splits = args.mf_ed1_accuracy_n_splits
        self.ed1_accuracy_test_prop = args.mf_ed1_accuracy_test_prop

    def get_evaluation_instance(self, period, period_labels, pos_ranges, range_idx):
        """Model-free implementation.

        For model-free explainers, evaluation instances are the full anomaly ranges prepended
        with (all) their preceding normal data.
        """
        instance, instance_labels, pos_range = None, None, pos_ranges[range_idx]
        range_class = period_labels[pos_range[0]]
        normal_data = (pos_range[0] != 0) if range_idx == 0 else (pos_range[0] != pos_ranges[range_idx-1][1])
        if normal_data:
            instance_start = 0 if range_idx == 0 else pos_ranges[range_idx-1][1]
            normal_length, anomaly_length = pos_range[0] - instance_start, pos_range[1] - pos_range[0]
            if normal_length >= self.min_normal_length and anomaly_length >= self.min_anomaly_length:
                instance = period[instance_start:pos_range[1]]
                instance_labels = np.concatenate([
                    np.zeros(normal_length), range_class * np.ones(anomaly_length)
                ])
            else:
                warnings.warn(
                    f'Instance of type {range_class} dropped due to insufficient normal and/or anomaly length'
                )
        else:
            warnings.warn(f'Instance of type {range_class} dropped due to absence of preceding normal data')
        return instance, instance_labels

    def explain_instance(self, instance, instance_labels):
        """Model-free implementation.

        For model-free explainers, evaluation instances are the same as explained samples.
        """
        return self.explainer.explain_sample(instance, instance_labels)

    def get_disturbances_features(self, instance, instance_labels, explanation=None):
        """Model-free implementation.

        For model-free explainers, "disturbances" are defined as random samples separately
        drawn from the instance's normal and anomalous data.

        The explanatory features of the original instance are therefore not included in the
        consistency computation.
        """
        # separate normal from anomalous records
        normal_records, anomalous_records = get_split_sample(instance, instance_labels)
        n_normal_records, n_anomalous_records = len(normal_records), len(anomalous_records)
        # important (explanatory) features found for `ed1_consistency_n_disturbances` random samples
        samples_fts = []
        # round below except if the resulting size is zero
        normal_sample_size = max(int(self.ed1_consistency_sampled_prop * n_normal_records), 1)
        anomalous_sample_size = max(int(self.ed1_consistency_sampled_prop * n_anomalous_records), 1)
        for _ in range(self.ed1_consistency_n_disturbances):
            # draw a random sample from the normal and anomalous records
            normal_ids = random.sample(range(n_normal_records), normal_sample_size)
            anomalous_ids = random.sample(range(n_anomalous_records), anomalous_sample_size)
            # get important explanatory features for the sample
            sample_explanation, _ = self.explainer.explain_split_sample(
                normal_records[normal_ids], anomalous_records[anomalous_ids]
            )
            samples_fts.append(sample_explanation['important_fts'])
        return samples_fts

    def compute_ed1_accuracy(self, instance, instance_labels):
        """Model-free implementation.

        For model-free explainers, the normal and anomalous records of the instance are
        randomly split into a training and a test set (according to `self.ed1_accuracy_test_prop`).

        The explanation is derived on the training set and evaluated on the test set. The
        final performance is returned averaged across `self.ed1_accuracy_n_splits` random splits.
        """
        # extract the normal and anomalous records from the instance
        normal_records, anomalous_records = get_split_sample(instance, instance_labels)
        # get accuracy scores averaged across `ed1_accuracy_n_splits` random splits
        accuracy_scores = np.zeros(3)
        for _ in range(self.ed1_accuracy_n_splits):
            # get training and test instances
            normal_train, normal_test, = train_test_split(normal_records, test_size=self.ed1_accuracy_test_prop)
            anomalous_train, anomalous_test, = train_test_split(
                anomalous_records, test_size=self.ed1_accuracy_test_prop
            )
            test_instance, test_instance_binary_labels = get_merged_sample(normal_test, anomalous_test, 1)
            # derive explanation for the training instance
            train_explanation, _ = self.explainer.explain_split_sample(normal_train, anomalous_train)
            # evaluate classification performance on the test instance
            test_instance_preds = self.explainer.classify_sample(train_explanation, test_instance)
            accuracy_scores += precision_recall_fscore_support(
                test_instance_binary_labels, test_instance_preds, average='binary', zero_division=0
            )[:3]
        # return the average precision, recall and f1-score across splits
        return accuracy_scores / self.ed1_accuracy_n_splits

    def compute_ed2_accuracy(self, explanations, instances, instances_labels):
        """Model-free implementation.

        Since instances of model-free explainers contain both normal and anomalous records,
        an explanation derived for an instance can directly be evaluated on others.
        """
        if len(instances) == 1:
            return np.full(3, np.nan)
        accuracy_scores, n_explanations = np.zeros(3), len(explanations)
        for i, explanation in enumerate(explanations):
            explanation_accuracy_scores = np.zeros(3)
            for k in [j for j in range(n_explanations) if j != i]:
                y_preds = self.explainer.classify_sample(explanation, instances[k])
                explanation_accuracy_scores += precision_recall_fscore_support(
                    (instances_labels[k] > 0).astype(int), y_preds, average='binary', zero_division=0
                )[:3]
            # add average performance of the explanation across test instances
            accuracy_scores += (explanation_accuracy_scores / (n_explanations - 1))
        # return average performance across explanations
        return accuracy_scores / n_explanations


class ModelDependentEvaluator(EDEvaluator):
    """Model-dependent evaluation.

    Evaluates model-dependent explanation discovery methods, trying to explain predictions of an AD model.

    Such AD models and corresponding explainers are assumed to rely on a fixed input "sample length",
    which will typically differ from most anomaly lengths.

    - Anomalies smaller than the sample are either dropped or expanded with neighboring data,
        according to the value of `args.md_eval_small_anomalies_expansion`.
    - Anomalies larger than the sample are either "fully" or partially covered,
        according to the value of `args.md_eval_large_anomalies_coverage`.
    """
    def __init__(self, args, explainer):
        super().__init__(args, explainer)
        # check parameters
        a_t = 'the provided explanation discovery method must be model-dependent'
        assert isinstance(explainer, ModelDependentExplainer), a_t
        expansion_choices, coverage_choices = ['none', 'before', 'after', 'both'], ['all', 'center', 'end']
        for arg, t1, t2, choices in zip(
            [args.md_eval_small_anomalies_expansion, args.md_eval_large_anomalies_coverage],
            ['small', 'large'], ['expansion', 'coverage'], [expansion_choices, coverage_choices]
        ):
            assert arg in choices, f'{t1} anomalies {t2} policy must be in {choices}'

        # minimum evaluation instance length required to both explain it and compute ED1 consistency
        self.min_instance_length = self.explainer.sample_length + self.ed1_consistency_n_disturbances - 1

        # expansion policy for anomalies smaller than sample length ("none" for dropping them)
        self.small_anomalies_expansion = args.md_eval_small_anomalies_expansion
        # coverage policy for anomalies larger than sample length
        self.large_anomalies_coverage = args.md_eval_large_anomalies_coverage

        # label for the target anomaly in an evaluation instance, in case neighboring data is not fully normal
        self.target_anomaly_class = -1

    def get_evaluation_instance(self, period, period_labels, pos_ranges, range_idx):
        """Model-dependent implementation.

        For model-dependent explainers, evaluation instances are the full anomaly ranges, with instances
        smaller than the sample length either being ignored or expanded with neighboring data, according
        to the value of `self.small_anomalies_expansion`.
        """
        pos_range = pos_ranges[range_idx]
        anomaly_length, range_class = pos_range[1] - pos_range[0], period_labels[pos_range[0]]
        # insufficient anomaly length or dropping policy of anomalies smaller than the sample length
        if anomaly_length < self.min_anomaly_length or \
                (anomaly_length < self.explainer.sample_length and self.small_anomalies_expansion == 'none'):
            warnings.warn(f'Instance of type {range_class} dropped due to insufficient anomaly length')
            return None, None
        # initialize the instance and labels to the anomaly range
        instance, instance_labels = period[slice(*pos_range)], self.target_anomaly_class * np.ones(anomaly_length)
        prepended_start, appended_end = pos_range
        if anomaly_length < self.min_instance_length:
            # prepend and/or append some neighboring records
            period_end = len(period)
            n_added = self.min_instance_length - anomaly_length
            if self.small_anomalies_expansion != 'both':
                # try to prepend *or* append records
                expansion_type = 'prepend' if self.small_anomalies_expansion == 'before' else 'append'
                prepended_start, appended_end = self.get_expanded_indices(
                    expansion_type, n_added, prepended_start, appended_end, range_class, period_end
                )
            else:
                # try to prepend *and* append records (arbitrarily append more than prepended if odd `n_added`)
                half_n_added = n_added / 2
                for expansion_type, n_records in zip(
                        ['prepend', 'append'], [floor(half_n_added), ceil(half_n_added)]
                ):
                    if not (prepended_start is None or appended_end is None):
                        prepended_start, appended_end = self.get_expanded_indices(
                            expansion_type, n_records, prepended_start, appended_end, range_class, period_end
                        )
            if prepended_start is None or appended_end is None:
                # instance expansion was needed and failed: drop the instance
                return None, None
        # return instance and instance labels
        prep_slice, app_slice = slice(prepended_start, pos_range[0]), slice(pos_range[1], appended_end)
        return np.concatenate([period[prep_slice], instance, period[app_slice]]), \
            np.concatenate([period_labels[prep_slice], instance_labels, period_labels[app_slice]])

    @staticmethod
    def get_expanded_indices(expansion_type, n_added, initial_start, initial_end, range_class, period_end):
        """Returns the expanded indices of the evaluation instance initially at `[initial_start, initial_end]`
            inside a `[0, period_length]` period, according to `expansion_type`.

        Note: The added neighboring data is not required to be fully normal.

        Note: If no records are available before/after the original instance bounds, they will
        be taken on the "other side" instead, if possible, no matter `expansion_type`.

        Args:
            expansion_type (str): type of expansion to perform (either "prepend" or "append")
            n_added (int): number of neighboring records to append/prepend.
            initial_start (int): initial (included) start index of the instance.
            initial_end (int): initial (excluded) end index of the instance.
            range_class (int): anomaly type of the instance, only used when displaying warnings.
            period_end (int): (excluded) end index of the period the instance belongs to.

        Returns:
            (int, int)|(None, None): the updated instance's (included) start and (excluded) end indices
                if the instance could be expanded, (None, None) otherwise.
        """
        a_t = 'the provided expansion type must be either `prepend` or `append`'
        assert expansion_type in ['prepend', 'append'], a_t
        prepended_start, appended_end = initial_start, initial_end
        if expansion_type == 'prepend':
            prepended_start -= n_added
            if prepended_start < 0:
                n_appended = -prepended_start
                prepended_start = 0
                warnings.warn(
                    f'Failed to fully prepend {n_added} records to type {range_class} instance, '
                    f'trying to append the remaining {n_appended}...'
                )
                appended_end += n_appended
                if appended_end > period_end:
                    warnings.warn(f'Failed. Instance dropped.')
                    return None, None
                warnings.warn(f'Succeeded.')
        else:
            appended_end += n_added
            if appended_end > period_end:
                n_prepended = appended_end - period_end
                appended_end = period_end
                warnings.warn(
                    f'Failed to fully append {n_added} records to type {range_class} instance, '
                    f'trying to prepend the remaining {n_prepended}...'
                )
                prepended_start -= n_prepended
                if prepended_start < 0:
                    warnings.warn(f'Failed. instance dropped.')
                    return None, None
                warnings.warn(f'Succeeded.')
        return prepended_start, appended_end

    def explain_instance(self, instance, instance_labels):
        """Model-dependent implementation.

        Model-dependent explainers can only explain samples of fixed length.

        For them, evaluation instances are the anomaly ranges, possibly expanded with some neighboring
        data for ranges that were smaller than `self.min_instance_length`.

        The explanation of an instance is defined depending of the length of its anomaly:

        - If it equals the sample length, it is the sample explanation.
        - If it is smaller than the sample length, it is the explanation of the sample obtained before
            adding extra records for ED1 consistency computation, when expanding the anomaly.
        - If it is larger than the sample length, it is defined according to the value of
            `self.large_anomalies_coverage`.
        """
        instance_length, sample_length = len(instance), self.explainer.sample_length

        # the whole instance is the target anomaly (and larger than the sample length)
        if instance_length > self.min_instance_length:
            if self.large_anomalies_coverage in ['center', 'end']:
                if self.large_anomalies_coverage == 'center':
                    # lean more towards the instance start if different parities
                    sample_start = floor((instance_length - sample_length) / 2)
                else:
                    sample_start = instance_length - sample_length
                samples = np.array([instance[sample_start:sample_start+sample_length]])
            else:
                # cover jumping samples spanning the whole anomaly range
                samples = get_sliding_windows(instance, sample_length, sample_length, include_remainder=True)
            # single explanation for all the provided samples
            return self.explainer.explain_all_samples(samples)

        # the instance contains the target anomaly
        anomaly_start, anomaly_end = extract_binary_ranges_ids(
            (instance_labels == self.target_anomaly_class).astype(int)
        )[0]
        anomaly_length = anomaly_end - anomaly_start

        # equal anomaly and sample lengths (expansion was only performed for ED1 consistency computation)
        if anomaly_length == sample_length:
            return self.explainer.explain_sample(instance[anomaly_start:anomaly_end])

        # anomaly is smaller than the sample: construct sample depending on the anomaly expansion
        n_prepended, n_appended = anomaly_start, instance_length - anomaly_end
        # remove the records only added for ED1 consistency computation
        n_removed = self.min_instance_length - sample_length
        if n_prepended == 0:
            sample_range = [0, instance_length - n_removed]
        elif n_appended == 0:
            sample_range = [n_removed, instance_length]
        else:
            half_n_removed = n_removed / 2
            # try removing half before and half after the anomaly (more before if odd)
            n_removed_before, n_removed_after = ceil(half_n_removed), floor(half_n_removed)
            # remove after if not enough prepended records
            if n_prepended - n_removed_before < 0:
                n_removed_after += (n_removed_before - n_prepended)
                n_removed_before = n_prepended
            sample_range = [n_removed_before, instance_length - n_removed_after]
        return self.explainer.explain_sample(instance[slice(*sample_range)])

    def get_disturbances_features(self, instance, instance_labels, explanation=None):
        """Model-dependent implementation.

        For model-dependent explainers, "disturbances" are defined differently depending on
        the instance length and anomaly coverage policy:

        - For instances of minimal length, disturbances are defined as 1-step sliding windows to the right.
        - For others, disturbances are defined depending on the anomaly coverage policy:
            - "Center" coverage: as 1-step sliding windows* alternately to the right and left.
            - "End" coverage: as 1-step sliding windows to the left.
            - "All" coverage: as random samples of `n` windows among all possible 1-step sliding ones
                in the instance. Where `n` is the number of windows used when explaining the original instance.

        *"windows" are all of size `self.explainer.sample_length`.

        In all above cases, the explanatory features of the original instance are (by convention)
        included in the consistency computation.
        """
        instance_length, sample_length = len(instance), self.explainer.sample_length
        n_disturbances = self.ed1_consistency_n_disturbances
        if instance_length == self.min_instance_length:
            # instance of minimal length: no initialization and slide through the whole instance
            samples_fts = []
            sub_instance = instance
        else:
            # larger instance: initialize with the explanation of the original instance
            e = explanation if explanation is not None else self.explain_instance(instance, instance_labels)[0]
            samples_fts = [e['important_fts']]
            if self.large_anomalies_coverage == 'end':
                # slide through the end of the instance if "end" anomaly coverage
                sub_instance = instance[(instance_length - sample_length - n_disturbances + 1):-1]
        if instance_length == self.min_instance_length or self.large_anomalies_coverage == 'end':
            samples = get_sliding_windows(sub_instance, sample_length, 1)
            samples_explanations, _ = self.explainer.explain_each_sample(samples)
            samples_fts += [s_e['important_fts'] for s_e in samples_explanations]
        elif self.large_anomalies_coverage == 'center':
            # alternately slide to the right and left of the anomaly center if "center" anomaly coverage
            center_start = floor((instance_length - sample_length) / 2)
            slide, slide_sign = 1, 1
            for _ in range(self.ed1_consistency_n_disturbances - 1):
                start = center_start + slide_sign * slide
                sample_explanation, _ = self.explainer.explain_sample(instance[start:(start + sample_length)])
                samples_fts.append(sample_explanation['important_fts'])
                slide_sign *= -1
                if slide_sign == 1:
                    slide += 1
        else:
            # construct every possible samples from the instance if "all" anomaly coverage
            samples_pool = get_sliding_windows(instance, sample_length, 1)
            # randomly sample the same number of samples as used for explaining the original instance
            for _ in range(n_disturbances):
                # any remaining sample was included when explaining the original instance
                samples = samples_pool[
                    random.sample(range(len(samples_pool)), ceil(instance_length / sample_length))
                ]
                samples_explanation, _ = self.explainer.explain_all_samples(samples)
                samples_fts.append(samples_explanation['important_fts'])
        return samples_fts

    def compute_ed1_accuracy(self, instance, instance_labels=None):
        """Model-dependent implementation.

        For model-dependent explainers, ED1 accuracy is not defined.
        """
        return np.full(3, np.nan)

    def compute_ed2_accuracy(self, explanations, instances, instances_labels=None):
        """Model-dependent implementation.

        For model-dependent explainers, ED2 accuracy is not defined.
        """
        return np.full(3, np.nan)


# dictionary gathering references to the defined evaluation methods
evaluation_classes = {
    'model_free': ModelFreeEvaluator,
    'model_dependent': ModelDependentEvaluator
}
