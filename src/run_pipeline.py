"""Module for running a full AD/AD+ED/ED pipeline.

Calls the main scripts in order with their corresponding arguments according to
the type of pipeline to run:

* COMMON PATH
- make_datasets.py.
- build_features.py.

* AD (+ED) PATH
- train_model.py
- train_scorer.py.
- train_detector.py.
(- train_explainer.py.)

* ED PATH
- train_explainer.py.

Except for the scoring and detection steps of the AD pipeline, we do not rerun a pipeline step
if it has been run already (existing output directory).
"""
import os
import warnings
from subprocess import call

from utils.common import parsers, get_output_path, get_command_line_string

# add absolute src directory to python path to import other project modules
import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(src_path)
from utils.common import get_args_string, get_best_thresholding_args

if __name__ == '__main__':
    # get command-line arguments and set up the scripts to call depending on the pipeline type
    args = parsers['run_pipeline'].parse_args()
    folders, scripts = ['data', 'features'], ['make_datasets', 'build_features']
    ad, ed = False, False
    if 'ad' in args.pipeline_type:
        ad = True
        folders += ['modeling', 'scoring', 'detection']
        scripts += ['train_model', 'train_scorer', 'train_detector']
    if 'ed' in args.pipeline_type:
        ed = True
        folders += ['explanation']
        scripts += ['train_explainer']
    if ad and ed and args.explained_predictions == 'ground.truth':
        warnings.warn('As specified, ED will be run after AD, but explaining ground truth labels')
    for folder, script in zip(folders, scripts):
        # if the output of a main pipeline step already exists, do not rerun it (except scoring and detection)
        if script not in ['train_scorer', 'train_detector']:
            if os.path.exists(get_output_path(args, script)):
                print(f'We do not run pipeline step "{script}" as it has been run already.')
                continue
        else:
            # the output of the detection step is defined as the method comparison file
            comparison_file = os.path.join(
                get_output_path(args, script, 'comparison'),
                f'{get_args_string(args, "ad_evaluation")}_detection_comparison.csv'
            )
            if os.path.isfile(comparison_file):
                print(f'We do not run pipeline step "{script}" as it has been run already.')
                continue
        # pick the "best" thresholding arguments if running ED after AD to explain model predictions
        if ad and ed and script == 'train_explainer' and args.explained_predictions == 'model':
            args = get_best_thresholding_args(args)
        # get the relevant arguments as a command-line string passed to the script call
        script_args = get_command_line_string(args, script)
        sts = call(f'python {os.path.join(folder, script + ".py")} {script_args}', shell=True)
