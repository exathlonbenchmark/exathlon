"""Module for running a full AD pipeline.

Calls the main scripts in order with their corresponding arguments.

We do not rerun a pipeline step if it has been run already (existing output directory).
"""
import os
from subprocess import call

from utils.common import parsers, get_output_path, get_command_line_string

if __name__ == '__main__':
    # get command-line arguments and setup the scripts to call depending on the pipeline type
    args = parsers['run_pipeline'].parse_args()
    folders = ['data', 'features', 'modeling', 'scoring', 'detection']
    scripts = ['make_datasets', 'build_features', 'train_model', 'train_scorer', 'train_detector']
    for folder, script in zip(folders, scripts):
        # if the output of a main pipeline step already exist, we don't rerun it
        if os.path.exists(get_output_path(args, script)):
            print(f'We don\'t run pipeline step "{script}" as it was run already.')
            continue
        # get the relevant arguments as a command-line string passed to the script call
        script_args = get_command_line_string(args, script)
        sts = call(f'python {os.path.join(folder, script + ".py")} {script_args}', shell=True)
