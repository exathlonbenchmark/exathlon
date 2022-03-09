"""Period ndarray visualization module.

Gathers functions for visualizing periods represented as ndarrays.
"""
import os
import argparse
import importlib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from metrics.ad_evaluators import extract_multiclass_ranges_ids
from detection.threshold_selectors import IQRSelector


def period_wise_figure(plot_func, periods, periods_labels=None, periods_info=None, fig_title=None,
                       full_output_path=None, used_data=None, **func_args):
    """Plots period-wise items within the same figure using the provided plotting function for each axis.

    Args:
        plot_func (func): plotting function to call for each period in the figure.
        periods (ndarray): period-wise arrays.
        periods_labels (ndarray|None): optional periods labels.
        periods_info (ndarray|None): optional periods information.
        fig_title (str|None): optional figure title.
        full_output_path (str|None): optional output path to save the figure to (including file name and extension).
        used_data (str|None): used data (if relevant, used to derive a period's title from its information).
        **func_args: optional keyword arguments to pass to the plotting function.
    """
    # create and setup new figure
    n_periods, fontsizes = periods.shape[0], {'title': 25}
    fig, axs = plt.subplots(n_periods, 1, sharex='none')
    fig.set_size_inches(20, 5 * n_periods)
    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=fontsizes['title'], fontweight='bold')

    # define axes, period labels and period titles to loop through
    if n_periods == 1:
        axs = [axs]
    looped = dict()
    for k, items in zip(['labels', 'title'], [periods_labels, periods_info]):
        if items is None:
            looped[k] = np.repeat(None, n_periods)
        elif k == 'labels' or used_data is None:
            looped[k] = items
        else:
            get_title = importlib.import_module(f'visualization.helpers.{used_data}').get_period_title_from_info
            looped[k] = [get_title(info) for info in items]

    # call the plotting function for each period separately
    for i, ax in enumerate(axs):
        plot_func(
            periods[i], period_labels=looped['labels'][i], period_title=looped['title'][i], ax=ax, **func_args
        )

    # save the figure as an image if an output path was provided
    if full_output_path is not None:
        print(f'saving period-wise figure to {full_output_path}...', end=' ', flush=True)
        os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
        fig.savefig(full_output_path)
        plt.close()
        print('done.')


def plot_period_scores(period_scores, period_labels=None, period_title=None, *, ax=None, color='darkgreen'):
    """Plots the provided period outlier scores through time, highlighting anomalous ranges if specified and any.

    Args:
        period_scores (ndarray): 1d-array of outlier scores to plot.
        period_labels (ndarray|None): optional multiclass labels for the period's records.
        period_title (list|None): optional title to use for the period.
        ax (AxesSubplot|None): optional plt.axis to plot the scores on if not in a standalone figure.
        color (str): color to use for the curve of outlier scores.

    Returns:
        AxesSubplot: the axis the period was plotted on, to enable further usage.
    """
    # setup font sizes and title
    fontsizes = {'title': 25, 'axes': 25, 'legend': 25, 'ticks': 22}
    title = period_title if period_title is not None else 'Outlier Scores'

    # create standalone figure if needed and setup axis
    if ax is None:
        plt.figure(figsize=(20, 5))
        ax = plt.axes()
    ax.set_title(title, fontsize=fontsizes['title'], y=1.07)
    ax.set_xlabel('Time Index', fontsize=fontsizes['axes'])
    ax.set_ylabel('Outlier Score', fontsize=fontsizes['axes'])
    ax.tick_params(axis='both', which='major', labelsize=fontsizes['ticks'])
    ax.tick_params(axis='both', which='minor', labelsize=fontsizes['ticks'])

    # plot the period outlier scores
    ax.plot(period_scores, color=color)

    # highlight the anomalous ranges of the period if any
    if period_labels is not None:
        plot_period_anomalies(ax, period_labels)

    # remove duplicate labels for the legend
    handles, labels = ax.get_legend_handles_labels()
    label_dict = dict(zip(labels, handles))
    if len(label_dict) > 0:
        # show that anomalous ranges are represented as rectangles in the legend (color = red with alpha 0.05)
        red_patch = mpatches.Patch(facecolor='#FFF2F2', edgecolor='red', linewidth=1)
        ax.legend(
            handles=[red_patch], labels=label_dict.keys(),
            loc='upper left', prop={'size': fontsizes['legend']}
        )
    ax.grid()
    return ax


def plot_period_anomalies(ax, period_labels):
    """Plots any anomalous range(s) within a period on `ax` based on `periods_labels`.

    Args:
        ax (AxesSubplot): plt.axis on which to plot the anomalous ranges.
        period_labels (ndarray): 1d-array of multiclass labels for the period.
    """
    # extract contiguous ranges from the record-wise labels
    ranges_ids_dict = extract_multiclass_ranges_ids(period_labels)
    # plot anomalous ranges by type
    for class_label in ranges_ids_dict:
        anomalous_ranges = ranges_ids_dict[class_label]
        for range_ in anomalous_ranges:
            # end of the range is exclusive
            beg, end = (range_[0], range_[1] - 1)
            ax.axvspan(beg, end, color='r', alpha=0.05)
            for range_idx in [beg, end]:
                ax.axvline(range_idx, label='Real Anomaly Range', color='r')


def plot_scores_distributions(periods_scores, periods_labels, threshold=None, restricted_types=None,
                              fig_title=None, type_colors=None, anomaly_types=None, full_output_path=None):
    """Plots the distributions of the provided `scores` by record types.

    Args:
        periods_scores (ndarray): periods scores of shape `(n_periods, period_length)`.
            Where `period_length` depends on the period.
        periods_labels (ndarray): either binary or multiclass periods labels.
            With the same shape as `periods_scores`.
        threshold (float|None): optional outlier score threshold value to highlight in the figure.
        restricted_types (list|None): optional restriction of record types to plot.
            If not None, have to be either `normal` or `anomalous`, or either `normal` or in `anomaly_types`.
        fig_title (str|None): optional figure title.
        type_colors (dict|None): if multiple anomaly types, colors to use for each type.
            Every type in `anomaly_types` must then be present as a key in `type_colors`. The color for
            "normal" (label 0) is fixed to blue.
        anomaly_types (list|None): names of the anomaly types that we might encounter in the data (if relevant).
        full_output_path (str|None): optional output path to save the figure to (including file name and extension).
    """
    # check optional type restrictions
    if restricted_types is not None:
        if anomaly_types is None:
            a_t = 'restricted types have to be in `{normal, anomalous}`'
            assert len(set(restricted_types) - {'normal', 'anomalous'}) == 0, a_t
        else:
            a_t = 'restricted types have to be either `normal` or in `anomaly_types`'
            assert len(set(restricted_types) - set(['normal'] + anomaly_types)) == 0, a_t

    # label classes list, in which the index corresponds to the integer label
    label_class_names = ['normal']
    # label texts dictionary, mapping a label class to its displayed text
    label_texts_dict = {'normal': 'Normal'}
    if anomaly_types is None:
        label_class_names += ['anomalous']
        label_texts_dict.update({'anomalous': 'Anomalous'})
        # type colors map label classes to their corresponding color
        if type_colors is not None:
            print('Warning: type colors are only considered if multiple anomaly types are provided')
        type_colors = {'anomalous': 'orange'}
    else:
        label_class_names += anomaly_types
        label_texts_dict.update({a_t: f'T{i+1}' for i, a_t in enumerate(anomaly_types)})
        a_t = 'a color must be provided for every type in `anomaly_types`'
        assert len(set(anomaly_types) - set(type_colors.keys())) == 0, a_t
    colors = dict({label_class_names[0]: 'blue'}, **type_colors)
    alphas = dict({label_class_names[0]: 0.6}, **{k: 0.5 for k in type_colors})

    # histogram assignments
    flattened_scores = np.concatenate(periods_scores)
    flattened_labels = np.concatenate(periods_labels)
    # remove scores of the unknown class if any
    if anomaly_types is not None:
        try:
            unknown_class = anomaly_types.index('unknown') + 1
            type_mask = (flattened_labels != unknown_class)
            flattened_scores = flattened_scores[type_mask]
            flattened_labels = flattened_labels[type_mask]
        except ValueError:
            pass
    # integer label classes
    int_label_classes = np.unique(flattened_labels)

    # any values exceeding `thresholding_factor` * IQR are grouped in the last bin
    thresholding_args = argparse.Namespace(
        **{'thresholding_factor': 3, 'n_iterations': 1, 'removal_factor': 1}
    )
    selector = IQRSelector(thresholding_args, '')
    selector.select_threshold(flattened_scores)
    flattened_scores[flattened_scores >= selector.threshold] = selector.threshold
    capped = len(flattened_scores[flattened_scores >= selector.threshold]) != 0

    # put the "normal" class last if it is there so that it is shown above
    if 0 in int_label_classes:
        int_label_classes = [cl for cl in int_label_classes if cl != 0]
        int_label_classes.insert(len(int_label_classes), 0)
    scores_dict = dict()
    for int_class in int_label_classes:
        # only consider scores that are of the restricted types if relevant
        if restricted_types is None or label_class_names[int_class] in restricted_types:
            scores_dict[label_class_names[int_class]] = flattened_scores[flattened_labels == int_class]

    # setup figure and plot outlier scores histograms
    fontsizes_dict = {'title': 22, 'axes': 17, 'legend': 17, 'ticks': 17}
    n_bins = 30
    if capped:
        # plot the histograms on a wider left axis
        subplots_args = {'nrows': 1, 'ncols': 2, 'gridspec_kw': {'width_ratios': [3, 1]}}
        bins = np.linspace(0, np.max(flattened_scores), n_bins + 1)
    else:
        subplots_args = dict()
        bins = n_bins
    fig, axs = plt.subplots(figsize=(15, 5), **subplots_args)
    hist_ax = axs[0] if capped else axs
    fig.suptitle(fig_title, size=fontsizes_dict['title'], y=0.96)
    hist_labels = []
    for k, scores in scores_dict.items():
        uncapped_scores = scores[scores < selector.threshold]
        hist_labels.append(label_texts_dict[k])
        hist_ax.hist(
            uncapped_scores, bins=bins, label=hist_labels[-1], color=colors[k],
            weights=np.ones_like(uncapped_scores) / len(uncapped_scores),
            alpha=alphas[k], edgecolor='black', linewidth=1.2
        )
    # highlight the outlier score threshold value if provided
    if threshold is not None:
        threshold_pos = (threshold, 0.60 * hist_ax.get_ylim()[1])
        threshold_text_pos = (1.05 * threshold_pos[0], 1.05 * threshold_pos[1])
        hist_ax.axvline(threshold, color='r', lw=3, linestyle='--')
        hist_ax.annotate(
            'Threshold', threshold_pos,
            color='r', fontsize=fontsizes_dict['legend'], xytext=threshold_text_pos
        )
    plt.grid()
    # re-order the legend labels if relevant
    handles, legend_labels = hist_ax.get_legend_handles_labels()
    if anomaly_types is not None:
        # all possible anomaly labels might not be represented in the scores
        used_label_texts = {k: v for k, v in label_texts_dict.items() if v in legend_labels}
        for la, new_idx in zip(used_label_texts.values(), range(len(hist_labels))):
            current_idx = legend_labels.index(la)
            legend_labels[current_idx] = legend_labels[new_idx]
            handle_to_move = handles[current_idx]
            handles[current_idx] = handles[new_idx]
            legend_labels[new_idx] = la
            handles[new_idx] = handle_to_move
    hist_ax.legend(loc='best', prop={'size': fontsizes_dict['legend']}, labels=legend_labels, handles=handles)
    hist_ax.set_xlabel('Outlier Score', fontsize=fontsizes_dict['axes'])
    hist_ax.set_ylabel('Frequency', fontsize=fontsizes_dict['axes'])
    hist_ax.tick_params(axis='both', which='major', labelsize=fontsizes_dict['ticks'])
    hist_ax.tick_params(axis='both', which='minor', labelsize=fontsizes_dict['ticks'])
    hist_ax.grid(True)

    # add capped scores bar chart to a narrower right axis if relevant
    if capped:
        bar_ax, bar_labels = axs[1], []
        # put outlier scores of normal records back as the first key
        reordered_scores = dict(
            {label_class_names[0]: scores_dict[label_class_names[0]]},
            **{k: v for k, v in scores_dict.items() if k != label_class_names[0]}
        )
        i = 0
        for k, scores in reordered_scores.items():
            # only consider the first letter of the "normal" label due to space constraints
            bar_labels.append(label_texts_dict[k] if k != label_class_names[0] else label_texts_dict[k][0])
            capped_scores = scores[scores == selector.threshold]
            bar_ax.bar(
                [i], [len(capped_scores) / len(scores)], color=colors[k],
                alpha=alphas[k], edgecolor='black', linewidth=1.2
            )
            i += 1
        plt.xticks(
            range(len(reordered_scores)), bar_labels,
            fontweight='light', fontsize='x-large'
        )
        bar_ax.yaxis.set_label_position('right')
        bar_ax.yaxis.tick_right()
        bar_ax.tick_params(axis='both', which='major', labelsize=fontsizes_dict['ticks'])
        bar_ax.tick_params(axis='both', which='minor', labelsize=fontsizes_dict['ticks'])
        bar_ax.grid(True)

    # save the figure as an image if an output path was provided
    if full_output_path is not None:
        print(f'saving scores distributions figure to {full_output_path}...', end=' ', flush=True)
        os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
        fig.savefig(full_output_path)
        plt.close()
        print('done.')
