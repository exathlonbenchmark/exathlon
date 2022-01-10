"""Anomaly detection metrics definition module.
"""
import os
from abc import abstractmethod

import numpy as np
from sklearn.metrics import auc, recall_score, precision_recall_fscore_support

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from detection.helpers import threshold_scores

# evaluation parameters corresponding to each AD level
AD_LEVEL_PARAMS = [
    # AD1
    {
        'recall_alpha': 1.0,
        'recall_omega': 'default', 'recall_delta': 'flat', 'recall_gamma': 'dup',
        'precision_omega': 'default', 'precision_delta': 'flat', 'precision_gamma': 'dup'
    },
    # AD2
    {
        'recall_alpha': 0.0,
        'recall_omega': 'default', 'recall_delta': 'flat', 'recall_gamma': 'dup',
        'precision_omega': 'default', 'precision_delta': 'flat', 'precision_gamma': 'dup'
    },
    # AD3
    {
        'recall_alpha': 0.0,
        'recall_omega': 'flat.normalized', 'recall_delta': 'front', 'recall_gamma': 'dup',
        'precision_omega': 'default', 'precision_delta': 'flat', 'precision_gamma': 'dup'
    },
    # AD4
    {
        'recall_alpha': 0.0,
        'recall_omega': 'flat.normalized', 'recall_delta': 'front', 'recall_gamma': 'no.dup',
        'precision_omega': 'default', 'precision_delta': 'flat', 'precision_gamma': 'no.dup'
    }
]


def get_range_pr_params(args):
    """Returns the range-based Precision and Recall parameters from `args`.

    Args:
        args (argparse.Namespace): parsed command-line arguments.

    Returns:
        dict: parameters dictionary of the form `{parameter_name: parameter_value}`.
    """
    a_t = 'the provided evaluation type must be either `range` or a valid AD level'
    assert args.evaluation_type in ['range'] + [f'ad{i+1}' for i in range(len(AD_LEVEL_PARAMS))], a_t
    if args.evaluation_type == 'range':
        # simply return the parameters of the command-line arguments
        args_dict = vars(args)
        return {k: args_dict[k] for k in AD_LEVEL_PARAMS[0]}
    # else return the hardcoded parameters corresponding to the AD level
    return AD_LEVEL_PARAMS[int(args.evaluation_type[2:])-1]


def get_f_beta_score(precision, recall, beta):
    """Returns the F_{`beta`}-score for the provided `precision` and `recall`.
    """
    # if both precision and recall are 0 we define the F-score as 0
    if precision == recall == 0:
        return 0
    beta_squared = beta ** 2
    return (1 + beta_squared) * precision * recall / (beta_squared * precision + recall)


def get_auc(x, y):
    """Returns the Area Under the Curve (AUC) for the provided `(x, y)` points.

    Prior to computing the AUC, the (x, y) points are sorted by `x` values to guarantee their monotonicity.

    Args:
        x (array-like): 1d-array of x-values.
        y (array-like): 1d-array of y-values.

    Returns:
        float: the AUC for the `(x, y)` points.
    """
    sorting_ids = np.argsort(x)
    return auc(x[sorting_ids], y[sorting_ids])


def extract_binary_ranges_ids(y):
    """Returns the start and (excluded) end indices of all contiguous ranges of 1s in the binary array `y`.

    Args:
        y (ndarray): 1d-array of binary elements.

    Returns:
        ndarray: array of `start, end` indices: `[[start_1, end_1], [start_2, end_2], ...]`.
    """
    y_diff = np.diff(y)
    start_ids = np.concatenate([[0] if y[0] == 1 else np.array([], dtype=int), np.where(y_diff == 1)[0] + 1])
    end_ids = np.concatenate([np.where(y_diff == -1)[0] + 1, [len(y)] if y[-1] == 1 else np.array([], dtype=int)])
    return np.array(list(zip(start_ids, end_ids)))


def extract_multiclass_ranges_ids(y):
    """Returns the start and (excluded) end ids of all contiguous ranges for each non-zero label in `y`.

    The lists of ranges are returned as a dictionary with as keys the positive labels in `y`
    and as values the corresponding range tuples.

    Args:
        y (ndarray): 1d-array of multiclass labels.

    Returns:
        dict: for each non-zero label, array of `start, end` indices:
            => `[[start_1, end_1], [start_2, end_2], ...]`.
    """
    # distinct non-zero labels in `y` (i.e. "positive classes")
    y_classes = np.unique(y)
    pos_classes = y_classes[y_classes != 0]
    ranges_ids_dict = dict()
    for pc in pos_classes:
        # extract binary ranges setting the class label to 1 and all others to 0
        ranges_ids_dict[pc] = extract_binary_ranges_ids((y == pc).astype(int))
    return ranges_ids_dict


def get_overlapping_ids(target_range, ranges, only_first=False):
    """Returns either the minimum or the full list of overlapping indices between `target_range` and `ranges`.

    If there are no overlapping indices between `target_range` and `ranges`, None is returned if
    `only_first` is True, else an empty list is returned.

    Note: the end indices can be either included or not, what matters here is the overlap
    between the ids, and not what they represent.

    Args:
        target_range (ndarray): target range to match as a `[start, end]` array.
        ranges (ndarray): candidate ranges, as a 2d `[[start_1, end_1], [start_2, end_2], ...]` array.
        only_first (bool): whether to return only the first (i.e. minimum) overlapping index.

    Returns:
        int|list: the first overlapping index if `only_first` is True, else the full list of overlapping indices.
    """
    target_set, overlapping_ids = set(range(target_range[0], target_range[1] + 1)), set()
    for range_ in ranges:
        overlapping_ids = overlapping_ids | (set(range(range_[0], range_[1] + 1)) & target_set)
    if only_first and len(overlapping_ids) == 0:
        return None
    else:
        return min(overlapping_ids) if only_first else list(overlapping_ids)


def get_overlapping_range(range_1, range_2):
    """Returns the start and end indices of the overlap between `range_1` and `range_2`.

    If `range_1` and `range_2` do not overlap, None is returned.

    Args:
        range_1 (ndarray): `[start, end]` indices of the first range.
        range_2 (ndarray): `[start, end]` indices of the second range.

    Returns:
        ndarray|None: `[start, end]` indices of the overlap between the 2 ranges, None if none.
    """
    overlapping_ids = get_overlapping_ids(range_1, np.array([range_2]), only_first=False)
    if len(overlapping_ids) == 0:
        return None
    return np.array([min(overlapping_ids), max(overlapping_ids)])


def get_overlapping_ranges(target_range, ranges):
    """Returns the start and end indices of all overlapping ranges between `target_range` and `ranges`.

    If none of the ranges overlap with `target_range`, an empty ndarray is returned.

    Args:
        target_range (ndarray): target range to overlap as a `[start, end]` array.
        ranges (ndarray): candidate ranges, as a 2d `[[start_1, end_1], [start_2, end_2], ...]` array.

    Returns:
        ndarray: all overlaps as a `[[start_1, end_1], [start_2, end_2], ...]` array.
    """
    overlaps = []
    for range_ in ranges:
        overlap = get_overlapping_range(target_range, range_)
        if overlap is not None:
            overlaps.append(overlap)
    return np.array(overlaps)


class ADEvaluator:
    """Anomaly detection evaluation base class.

    Computes the Precision, Recall and F_{beta}-score for the predicted anomalies on a dataset.

    While Precision is always computed globally and returned as a single value, Recall and
    F-score are returned as dictionaries whose keys are:
    - `global`, considering all non-zero labels as a single positive anomaly class.
    - every distinct non-zero label encountered in the data, considering the corresponding class only.
    - `avg`, corresponding to the average scores across each positive class (excluding the `global` key).

    Args:
        args (argparse.Namespace): parsed command-line arguments.
    """
    def __init__(self, args):
        # assign `beta` times more importance to Recall than Precision
        self.beta = args.f_score_beta

    def precision_recall_curves(self, periods_labels, periods_scores, n_thresholds=500, return_f_scores=False):
        """Returns the evaluator's precisions and recalls for a linear range of `n_thresholds` thresholds.

        A Precision score is returned for each threshold. Recalls and F-scores follow the same format,
        except the lists are grouped inside dictionaries with keys described in the class documentation.

        Args:
            periods_labels (ndarray): periods record-wise anomaly labels of shape `(n_periods, period_length)`
                Where `period_length` depends on the period.
            periods_scores (ndarray): periods record-wise outlier scores of the same shape.
            n_thresholds (int): number of threshold values at which to measure (precision, recall) pairs.
            return_f_scores (bool): whether to also return F-beta scores derived from the precisions and recalls.

        Returns:
            (dict, )ndarray, dict, ndarray: the corresponding (F-scores,) Precisions, Recalls and
                evaluated thresholds.
        """
        flattened_scores = np.concatenate(periods_scores, axis=0)
        min_score, max_score = np.min(flattened_scores), np.max(flattened_scores)
        thresholds = np.linspace(start=min_score, stop=max_score, num=n_thresholds)
        # add a threshold above the maximum score (Recalls are 0 and Precisions should be defined to 1)
        thresholds = np.concatenate([thresholds, [np.inf]])
        # list of global Precision scores, lists of global and type-wise Recall and F-scores
        precisions, recalls_dict, f_scores_dict = [], dict(), dict()
        for threshold in thresholds:
            # compute predictions corresponding to the threshold and get metrics
            periods_preds = threshold_scores(periods_scores, threshold)
            t_f_scores, t_precision, t_recalls = self.compute_metrics(periods_labels, periods_preds)
            # append the type-wise and global Recall and F-scores for this threshold
            for k in t_f_scores:
                # not the first occurrence of the key: append it to the existing list
                if k in f_scores_dict:
                    recalls_dict[k].append(t_recalls[k])
                    f_scores_dict[k].append(t_f_scores[k])
                # first occurrence of the key: initialize the list with the score value
                else:
                    recalls_dict[k] = [t_recalls[k]]
                    f_scores_dict[k] = [t_f_scores[k]]
            precisions.append(t_precision)
        # convert all lists to numpy arrays and return the PR curves
        precisions = np.array(precisions)
        recalls_dict = {k: np.array(v) for k, v in recalls_dict.items()}
        if return_f_scores:
            f_scores_dict = {k: np.array(v) for k, v in f_scores_dict.items()}
            return f_scores_dict, precisions, recalls_dict, thresholds
        return precisions, recalls_dict, thresholds

    @abstractmethod
    def compute_metrics(self, periods_labels, periods_preds):
        """Returns the F-score, Precision and Recall for the provided `periods_labels` and `periods_preds`.

        Recalls and F-scores are returned as dictionaries like described in the class documentation.

        Args:
            periods_labels (ndarray): labels for each period in the dataset, of shape
                `(n_periods, period_length)`. With `period_length` depending on the period.
            periods_preds (ndarray): binary predictions for each period, in the same format.

        Returns:
            dict, float, dict: F_{beta}-scores, Precision and Recalls, respectively.
        """


class RangeEvaluator(ADEvaluator):
    """Range-based evaluation.

    The Precision and Recall metrics are defined for range-based time series anomaly detection.

    They can reward existence, ranges cardinality, range overlaps size and position, like introduced
    in https://papers.nips.cc/paper/7462-precision-and-recall-for-time-series.pdf.
    """
    def __init__(self, args):
        super().__init__(args)
        params_dict = get_range_pr_params(args)
        # Recall specifications
        self.recall_alpha = params_dict['recall_alpha']
        self.recall_omega = self.omega_functions[params_dict['recall_omega']]
        self.recall_delta = self.delta_functions[params_dict['recall_delta']]
        self.recall_gamma = self.gamma_functions[params_dict['recall_gamma']]
        # Precision specifications
        self.precision_omega = self.omega_functions[params_dict['precision_omega']]
        self.precision_delta = self.delta_functions[params_dict['precision_delta']]
        self.precision_gamma = self.gamma_functions[params_dict['precision_gamma']]

    def compute_metrics(self, periods_labels, periods_preds):
        """Returns range-based metrics for the provided `periods_labels` and `periods_preds`.
        """
        precisions, recalls_dict = [], dict()
        for y_true, y_pred in zip(periods_labels, periods_preds):
            # add the period's Precision and Recall scores to the full lists
            p_precisions, p_recalls = self.compute_period_metrics(y_true, y_pred, as_list=True)
            precisions += p_precisions
            # the Recall scores are added per key
            for k in p_recalls:
                if k in recalls_dict:
                    recalls_dict[k] += p_recalls[k]
                else:
                    recalls_dict[k] = p_recalls[k]
        # compute global Precision (define it to 1 if no positive predictions)
        n_precisions = len(precisions)
        precision = sum(precisions) / n_precisions if n_precisions > 0 else 1

        # compute Recall and F-score globally and per label key (define Recall to 1 if no positive labels)
        returned_recalls, returned_f_scores = dict(), dict()
        for k, recalls_list in dict(recalls_dict, **{'global': sum(recalls_dict.values(), [])}).items():
            n_recalls = len(recalls_list)
            returned_recalls[k] = sum(recalls_list) / n_recalls if n_recalls > 0 else 1
            returned_f_scores[k] = get_f_beta_score(precision, returned_recalls[k], self.beta)

        # compute average Recall and F-score across label keys
        label_recalls = {k: v for k, v in returned_recalls.items() if k != 'global'}.values()
        returned_recalls['avg'] = sum(label_recalls) / len(label_recalls)
        returned_f_scores['avg'] = get_f_beta_score(precision, returned_recalls['avg'], self.beta)
        return returned_f_scores, precision, returned_recalls

    def compute_period_metrics(self, y_true, y_pred, as_list=False):
        """Returns the F-score, Precision and Recall for the provided `y_true` and `y_pred` arrays.

        If `as_list` is True, then the Precision and Recall scores for each predicted/real anomaly range
        are returned instead.

        Recalls and F-scores are returned for each non-zero labels encountered in the period.
        => As dictionaries whose keys are the labels and values are the scores.

        Args:
            y_true (ndarray): 1d-array of multiclass anomaly labels.
            y_pred (ndarray): corresponding array of binary anomaly predictions.
            as_list (bool): the F-score, average Precision and average Recall are returned if True, else the list
                of precisions and recalls are returned.

        Returns:
            (dict, float, dict)|(list, dict): F-scores, average Precision and average Recalls if
                `as_list` is False, else lists of Precision and Recall scores.
        """
        # extract contiguous real and predicted anomaly ranges from the period's arrays
        real_ranges_dict = extract_multiclass_ranges_ids(y_true)
        predicted_ranges = extract_binary_ranges_ids(y_pred)
        # compute the Recall and Precision score for each real and predicted anomaly range, respectively
        recalls_dict, precisions = {k: [] for k in real_ranges_dict}, []
        for k, real_ranges in real_ranges_dict.items():
            for real_range in real_ranges:
                recalls_dict[k].append(self.compute_range_recall(real_range, predicted_ranges))
        # consider anomalous ranges of all classes when computing Precision
        r_ranges_values = list(real_ranges_dict.values())
        all_real_ranges = np.concatenate(r_ranges_values, axis=0) if len(r_ranges_values) > 0 else np.array([])
        for predicted_range in predicted_ranges:
            precisions.append(self.compute_range_precision(predicted_range, all_real_ranges))
        # return the full lists if specified
        if as_list:
            return precisions, recalls_dict
        # else return the overall F-score, Precision and Recall
        precision = sum(precisions) / len(precisions)
        returned_recalls, returned_f_scores = dict(), dict()
        for k in recalls_dict:
            returned_recalls[k] = sum(recalls_dict[k]) / len(recalls_dict[k])
            returned_f_scores[k] = get_f_beta_score(precision, returned_recalls[k], self.beta)
        return returned_f_scores, precision, returned_recalls

    def compute_range_precision(self, predicted_range, real_ranges):
        """Returns the Precision score for the provided `predicted_range`, considering `real_ranges`.

        Args:
            predicted_range (ndarray): start and end indices of the predicted anomaly to score, `[start, end]`.
            real_ranges (ndarray): 2d-array for the start and end indices of all the real anomalies.

        Returns:
            float: Precision score of the predicted anomaly range.
        """
        return self.overlap_reward(
            predicted_range, real_ranges, self.precision_omega, self.precision_delta, self.precision_gamma
        )

    def compute_range_recall(self, real_range, predicted_ranges):
        """Returns the Recall score for the provided `real_range`, considering `predicted_ranges`.

        Args:
            real_range (ndarray): start and end indices of the real anomaly to score, `[start, end]`.
            predicted_ranges (ndarray): 2d-array for the start and end indices of all the predicted anomalies.

        Returns:
            float: Recall score of the real anomaly range.
        """
        alpha, omega, delta, gamma = self.recall_alpha, self.recall_omega, self.recall_delta, self.recall_gamma
        return alpha * RangeEvaluator.existence_reward(real_range, predicted_ranges) + \
            (1 - alpha) * RangeEvaluator.overlap_reward(real_range, predicted_ranges, omega, delta, gamma)

    @staticmethod
    def existence_reward(range_, other_ranges):
        """Returns the existence reward of `range_` with respect to `other_ranges`.

        Args:
            range_ (ndarray): start and end indices of the range whose existence reward to compute.
            other_ranges (ndarray): 2d-array for the start and end indices of the other ranges to test
                overlapping with.

        Returns:
            int: 1 if `range_` overlaps with at least one record of `other_ranges`, 0 otherwise.
        """
        return 0 if get_overlapping_ids(range_, other_ranges, True) is None else 1

    @staticmethod
    def overlap_reward(range_, other_ranges, omega_f, delta_f, gamma_f):
        """Returns the overlap reward of `range_` with respect to `other_ranges` and
            the provided functions.

        Args:
            range_ (ndarray): start and end indices of the range whose overlap reward to compute.
            other_ranges (ndarray): 2d-array for the start and end indices of the "target" ranges.
            omega_f (func): size function.
            delta_f (func): positional bias.
            gamma_f (func): cardinality function.

        Returns:
            float: the overlap reward of `range_`, between 0 and 1.
        """
        size_rewards = 0
        for other_range in other_ranges:
            size_rewards += omega_f(range_, get_overlapping_range(range_, other_range), delta_f)
        return RangeEvaluator.cardinality_factor(range_, other_ranges, gamma_f) * size_rewards

    """Omega (size) functions.
    
    Return the size reward of the overlap based on the positional bias of the target range.
    """
    @staticmethod
    def default_size_function(range_, overlap, delta_f):
        """Returns the reward as the overlap size weighted by the positional bias.
        """
        if overlap is None:
            return 0
        # normalized rewards of the range's indices
        range_rewards = delta_f(range_[1] - range_[0])
        # return the total normalized reward covered by the overlap
        return sum(range_rewards[slice(*(overlap - range_[0]))])

    @staticmethod
    def flat_normalized_size_function(range_, overlap, delta_f):
        """Returns the overlap reward normalized so as not to exceed what it would be under a flat bias.
        """
        if overlap is None:
            return 0
        # normalized rewards of the range's indices under the provided positional bias
        range_rewards = delta_f(range_[1] - range_[0])
        # normalized rewards of the range's indices under a flat positional bias
        flat_rewards = RangeEvaluator.delta_functions['flat'](range_[1] - range_[0])
        # total normalized rewards covered by the overlap for both the provided and flat biases
        overlap_slice = slice(*(overlap - range_[0]))
        original_reward, flat_reward = sum(range_rewards[overlap_slice]), sum(flat_rewards[overlap_slice])
        # best achievable reward given the overlap size (under an "ideal" position)
        max_reward = sum(sorted(range_rewards, reverse=True)[:(overlap[1]-overlap[0])])
        # return the original reward normalized so that the maximum is now the flat reward
        return (flat_reward * original_reward / max_reward) if max_reward != 0 else 0

    # dictionary gathering references to the defined `omega` size functions
    omega_functions = {
        'default': default_size_function.__func__,
        'flat.normalized': flat_normalized_size_function.__func__
    }

    """Delta functions (positional biases). 
    
    Return the normalized rewards for each relative index in a range of length `range_length`.
    """
    @staticmethod
    def flat_bias(range_length): return np.ones(range_length) / range_length

    @staticmethod
    def front_end_bias(range_length):
        """The index rewards linearly decrease as we move forward in the range.
        """
        raw_rewards = np.flip(np.array(range(range_length)))
        return raw_rewards / sum(raw_rewards)

    @staticmethod
    def back_end_bias(range_length):
        """The index rewards linearly increase as we move forward in the range.
        """
        raw_rewards = np.array(range(range_length))
        return raw_rewards / sum(raw_rewards)

    # dictionary gathering references to the defined `delta` positional biases
    delta_functions = {
        'flat': flat_bias.__func__,
        'front': front_end_bias.__func__,
        'back': back_end_bias.__func__
    }

    @staticmethod
    def cardinality_factor(range_, other_ranges, gamma_f):
        """Returns the cardinality factor of `range_` with respect to `other_ranges` and `gamma_f`.

        Args:
            range_ (ndarray): start and end indices of the range whose cardinality factor to compute.
            other_ranges (ndarray): 2d-array for the start and end indices of the "target" ranges.
            gamma_f (func): cardinality function.

        Returns:
            float: the cardinality factor of `range_`, between 0 and 1.
        """
        n_overlapping_ranges = len(get_overlapping_ranges(range_, other_ranges))
        if n_overlapping_ranges == 1:
            return 1
        return gamma_f(n_overlapping_ranges)

    """Gamma functions (cardinality)
    """
    @staticmethod
    def no_duplicates_cardinality(n_overlapping_ranges): return 0

    @staticmethod
    def allow_duplicates_cardinality(n_overlapping_ranges): return 1

    @staticmethod
    def inverse_polynomial_cardinality(n_overlapping_ranges): return 1 / n_overlapping_ranges
    # dictionary gathering references to the defined `gamma` cardinality functions
    gamma_functions = {
        'no.dup': no_duplicates_cardinality.__func__,
        'dup': allow_duplicates_cardinality.__func__,
        'inv.poly': inverse_polynomial_cardinality.__func__
    }


class PointEvaluator(ADEvaluator):
    """Point-based evaluation.

    The Precision and Recall metrics are defined for point-based anomaly detection.
    """
    def __init__(self, args):
        super().__init__(args)

    def compute_metrics(self, periods_labels, periods_preds):
        """Returns point-based metrics for the provided `periods_labels` and `periods_preds`.
        """
        flattened_preds = np.concatenate(periods_preds, axis=0)
        flattened_labels = np.concatenate(periods_labels, axis=0)
        flattened_binary = np.array(flattened_labels > 0, dtype=int)

        # global Precision, Recall and F-score (considering all anomaly types as one)
        recalls_dict, f_scores_dict = dict(), dict()
        # define the Precision/Recall as 1 if no positive predictions/labels
        precision, recalls_dict['global'], f_scores_dict['global'], _ = precision_recall_fscore_support(
            flattened_binary, flattened_preds, beta=self.beta, average='binary', zero_division=1
        )
        # a single anomaly type
        if (flattened_labels == flattened_binary).all():
            for k in [1, 'avg']:
                recalls_dict[k], f_scores_dict[k] = recalls_dict['global'], f_scores_dict['global']
        # multiple anomaly types
        else:
            # type-wise Recall and corresponding F-scores
            unique_labels = np.unique(flattened_labels)
            pos_classes = unique_labels[unique_labels != 0]
            for pc in pos_classes:
                # Recall and corresponding F-score setting the class label to 1 and all others to 0
                recalls_dict[pc] = recall_score(
                    np.array(flattened_labels == pc, dtype=int), flattened_preds, zero_division=1
                )
                f_scores_dict[pc] = get_f_beta_score(precision, recalls_dict[pc], self.beta)
            # average Recall across anomaly types and corresponding F-scores
            label_recalls = {k: v for k, v in recalls_dict.items() if k != 'global'}.values()
            recalls_dict['avg'] = sum(label_recalls) / len(label_recalls)
            f_scores_dict['avg'] = get_f_beta_score(precision, recalls_dict['avg'], self.beta)
        return f_scores_dict, precision, recalls_dict


# dictionary gathering references to the defined evaluation methods
evaluation_classes = dict(
    {
        'point': PointEvaluator,
        'range': RangeEvaluator
    },
    **{f'ad{i}': RangeEvaluator for i in range(1, 5)}
)
