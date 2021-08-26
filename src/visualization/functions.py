"""Functions visualization module.
"""
import os

import numpy as np
import matplotlib.pyplot as plt


def plot_curves(x, y, *, fig_title=None, xlabel=None, ylabel=None,
                colors=None, show_max_values=True, full_output_path=None):
    """Utility function to plot (a) simple `(x, y)` curve(s) with a title and labels for the axes.

    We allow either `x` or `y` to be provided as a dictionary. In this case, a curve is drawn for each
    key of `x|y`, keeping `y|x` constant. Each key is assumed to represent the same quantity for `x|y`,
    so that the axis and axis label can be shared.

    The curve(s) colors can be specified using `colors`. If `x|y` is a dictionary and `colors` is specified,
    then it must also be a dictionary whose keys match the ones of `x|y`.

    Note: we do not allow both `x` and `y` to be dictionaries.

    Note: if `show_max_values` is True and there are multiple equal maxima for `y`, only the
    first one will be highlighted in the figure.

    Args:
        x (array-like|dict): x values.
        y (array-like|dict): y values. Must have the same number of elements as `x`(s).
        fig_title (str): optional title for the figure.
        xlabel (str): optional label for the x-axis.
        ylabel (str): optional label for the y-axis.
        colors (dict|str|None): optional curve color(s).
        show_max_values (bool): whether to highlight maximum `y` values for the curve(s).
        full_output_path (str|None): path to save the figure to if specified (with file name and extension).
    """
    # setup font sizes, create figure, set title and labels (no defaults)
    fontsizes_dict = {'title': 15, 'axes': 12, 'legend': 10, 'ticks': 12}
    fig, ax = plt.subplots()
    fig.suptitle(fig_title, size=fontsizes_dict['title'], y=0.96)
    ax.set_xlabel(xlabel, fontsize=fontsizes_dict['axes'])
    ax.set_ylabel(ylabel, fontsize=fontsizes_dict['axes'])
    ax.tick_params(axis='both', which='major', labelsize=fontsizes_dict['ticks'])
    ax.tick_params(axis='both', which='minor', labelsize=fontsizes_dict['ticks'])

    # coordinates of the maximum y value(s) to highlight if specified
    max_x, max_y = dict(), dict()

    # plot a single curve if `x` and `y` are simple arrays
    if type(x) != dict and type(y) != dict:
        assert colors is None or type(colors) != dict, 'color must be passed as a single value if one curve'
        if show_max_values:
            max_x['only_curve'], max_y['only_curve'] = x[np.argmax(y)], y.max()
        ax.plot(x, y, color=colors)
    # else plot one curve per dimension key, keeping the other constant
    else:
        assert colors is None or type(colors) == dict, 'colors must be passed as a dict if multiple curves'
        if type(x) == dict:
            assert type(y) != dict, 'x and y values cannot be both passed as dictionaries'
            for k in x:
                c = colors[k] if colors is not None else None
                if show_max_values:
                    max_x[k], max_y[k] = x[k][np.argmax(y)], y.max()
                ax.plot(x[k], y, color=c, label=k.replace('_', ' ').upper())
        else:
            assert type(x) != dict, 'x and y values cannot be both passed as dictionaries'
            for k in y:
                c = colors[k] if colors is not None else None
                if show_max_values:
                    max_x[k], max_y[k] = x[np.argmax(y[k])], y[k].max()
                ax.plot(x, y[k], color=c, label=k.replace('_', ' ').upper())
        # only plot a legend if multiple curves
        ax.legend(loc='best', prop={'size': fontsizes_dict['legend']})

    # show maximum coordinates pair(s) if specified
    for k in max_x:
        if k == 'only_curve':
            text, c = f'MAX = {max_y[k]:.2f}', colors
        else:
            text = f'{k.upper().replace("_", " ")} MAX = {max_y[k]:.2f}'
            c = colors[k] if colors is not None else None
        bbox_props = dict(boxstyle='square,pad=0.3', fc='w', ec=c, lw=0.72)
        arrow_props = dict(
            arrowstyle='->', connectionstyle='angle,angleA=0,angleB=60', color=c
        )
        kw = dict(
            xycoords='data', textcoords='offset points', ha='center', va='bottom',
            arrowprops=arrow_props, bbox=bbox_props
        )
        ax.annotate(text, color=c, xy=(max_x[k], max_y[k]), xytext=(80, 20), **kw)

    # add axis grid and save the figure as an image if specified
    ax.grid()
    if full_output_path is not None:
        print(f'saving figure to {full_output_path}...', end=' ', flush=True)
        os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
        fig.savefig(full_output_path)
        plt.close()
        print('done.')


def plot_curve(x, y, fig_title, xlabel, ylabel, color=None, full_output_path=None):
    """Utility function to plot a simple `(x, y)` curve with a title and labels for the axes.

    Args:
        x (array-like): x values as a 1d-array.
        y (array-like): y values as a 1d-array. Must have the same number of elements as `x`.
        fig_title (str): title for the figure.
        xlabel (str): label for the x-axis.
        ylabel (str): label for the y-axis.
        color (str|None): optional curve color
        full_output_path (str|None): path to save the figure to if specified (with file name and extension).
    """
    plot_curves(
        x, y, fig_title=fig_title, xlabel=xlabel, ylabel=ylabel,
        colors=color, full_output_path=full_output_path
    )
