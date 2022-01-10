"""Evaluation metrics visualization module.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from metrics.ad_evaluators import get_auc


def plot_pr_curves(recalls, precisions, ts, *,
                   fig_title=None, colors=None, full_output_path=None):
    """Plots the Precision-Recall curve(s) corresponding to the provided `recalls` and `precisions`.

    If `recalls` is a dictionary, a curve will be plotted for each key, optionally using the colors
    specified by `colors` (if those 2 dictionaries are specified, their keys must hence match).
    The keys will also be used for the legend.

    If `recalls` is an ndarray (like `precisions`), a single curve will be plotted,
    optionally using the single color specified by `colors`.

    Args:
        recalls (ndarray|dict): recall(s) values for each threshold.
        precisions (ndarray): precision values for each threshold.
        ts (ndarray): threshold value for each (recall, precision) pair.
        fig_title (str|None): optional figure title.
        colors (dict|str|None): optional curve color(s).
        full_output_path (str|None): path to save the figure to if specified (with file name and extension).
    """
    # setup font sizes, create new figure, set title, labels and limits
    fontsizes_dict = {'title': 15, 'axes': 12, 'legend': 10, 'ticks': 12}
    fig, ax = plt.subplots()
    if fig_title is None:
        fig_title = 'Precision-Recall Curve'
    fig.suptitle(fig_title, size=fontsizes_dict['title'], y=0.96)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=fontsizes_dict['axes'])
    ax.set_ylabel('Precision', fontsize=fontsizes_dict['axes'])
    ax.tick_params(axis='both', which='major', labelsize=fontsizes_dict['ticks'])
    ax.tick_params(axis='both', which='minor', labelsize=fontsizes_dict['ticks'])

    # get minimum and maximum threshold values to highlight (second max if max is inf)
    min_ts, max_ts = ts[0], (ts[-1] if ts[-1] != np.inf else ts[-2])

    # if `recalls` is not a dict, plot a single PR curve
    if type(recalls) != dict:
        assert colors is None or type(colors) != dict, 'color must be passed as a single value if one curve'
        # show the minimum and maximum threshold values to highlight the "direction" of the curve
        ax.scatter([recalls[0], recalls[-1]], [precisions[0], precisions[-1]], color='r', s=75, marker='.')
        for index, text in zip([0, -1], [f'Threshold = {min_ts:.2f}', f'Threshold = {max_ts:.2f}']):
            ax.annotate(text, (recalls[index], precisions[index]), ha='center')
        # plot the PR curve
        c = colors if colors is not None else 'blue'
        ax.plot(recalls, precisions, color=c, label=f'(AUC = {get_auc(recalls, precisions):.2f})')
    # if `recalls` is a dict, plot one PR curve per key
    else:
        assert colors is None or type(colors) == dict, 'colors must be passed as a dict if multiple curves'
        if 'global' in recalls:
            # only show minimum and maximum thresholds for the "global" curve if it exists
            ax.scatter(
                [recalls['global'][0], recalls['global'][-1]], [precisions[0], precisions[-1]],
                color='r', s=75, marker='.'
            )
            for index, text in zip([0, -1], [f'Threshold = {min_ts:.2f}', f'Threshold = {max_ts:.2f}']):
                ax.annotate(
                    text, (recalls['global'][index], precisions[index]),
                    ha='center', bbox=dict(boxstyle='round', fc='w')
                )
        # plot the PR curves
        for k in recalls:
            c = colors[k] if colors is not None else None
            ax.plot(
                recalls[k], precisions, color=c,
                label=f'{k.replace("_", " ").upper()} (AUC = {get_auc(recalls[k], precisions):.2f})'
            )
    ax.legend(loc='best', prop={'size': fontsizes_dict['legend']})
    ax.grid()

    # save the figure as an image if an output path was specified
    if full_output_path is not None:
        print(f'saving PR curve figure to {full_output_path}...', end=' ', flush=True)
        os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
        fig.savefig(full_output_path)
        plt.close()
        print('done.')
