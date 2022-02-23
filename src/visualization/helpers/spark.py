"""Spark-specific visualization helpers.
"""
import os

import matplotlib.pyplot as plt

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir))
sys.path.append(src_path)
from utils.spark import ANOMALY_TYPES

# colors used for representing each anomaly type
dark2_palette = plt.get_cmap('Dark2').colors
ANOMALY_COLORS = {type_: dark2_palette[i] for i, type_ in enumerate(ANOMALY_TYPES)}

# colors used for reporting performance globally and per anomaly type
METRICS_COLORS = dict({'global': 'blue'}, **ANOMALY_COLORS)


def get_period_title_from_info(period_info):
    """Returns the title to use to describe a specific period's plot from its information."""
    return f'{period_info[0]} ({period_info[1].title().replace("_", " ").replace("Cpu", "CPU")})'
