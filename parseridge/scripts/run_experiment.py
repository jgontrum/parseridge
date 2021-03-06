import argparse
import os
import subprocess
from copy import copy

import yaml

IGNORE_KEYS = ["repository", "code_path", "python_bin", "experiment_group"]

"""
Script that takes an experiment definition file and creates a new folder for it.
It then downloads the parser code at the given commit hash and passes the parameters
to the training scripts. Useful when running a lot of experiments on a compute cluster.
"""


def get_args():
    parser = argparse.ArgumentParser(
        description="Helper script to run experiments in a repeatable way.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("experiment", type=str, help="Path to the experiment definition.")
    parser.add_argument("--dry_run", action="store_true")

    return parser.parse_args()


def dict_generator(indict):
    """
    Yields the leaves of a dict (keys + values).
    """
    if isinstance(indict, dict):
        for key, value in indict.items():
            if isinstance(value, dict):
                for d in dict_generator(value):
                    yield d
            else:
                yield key, value
    else:
        yield indict


if __name__ == "__main__":
    args = get_args()

    experiment_definition = yaml.safe_load(open(args.experiment))

    experiment = experiment_definition["experiment"]

    # Fill in the templates in the path values
    experiment_copy = copy(experiment)
    for k, v in experiment.items():
        if not isinstance(v, str):
            continue

        experiment[k] = experiment[k].format(**experiment_copy)

        # Make sure the paths are empty and create the directories
        if v.endswith("_path"):
            if os.path.exists(v):
                raise Exception(f"Folder already exists: {v}.")
            if args.dry_run:
                print(f"Would create folder: {v}")
            else:
                os.makedirs(v, exist_ok=True)

        elif v.endswith("_file"):
            if os.path.exists(v):
                raise Exception(f"File already exists: {v}.")
            if args.dry_run:
                print(f"Would create folder: {os.path.dirname(v)}")
            else:
                os.makedirs(os.path.dirname(v), exist_ok=True)
    #
    # # Clone the code base and switch to the required commit
    cmd = (
        f"git clone {experiment['repository']} {experiment['code_path']} &&"
        f"cd {experiment['code_path']} &&"
        f"git checkout --quiet  {experiment['commit']}"
    )

    if args.dry_run:
        print(f"Would run: {cmd}")
    else:
        subprocess.run(cmd, shell=True)

    # Get all the arguments we want to pass to the trainer
    training_args = {}
    for k, v in dict_generator(experiment_definition):
        if k not in IGNORE_KEYS:
            training_args[k] = v

    arguments = []
    for option, value in training_args.items():
        if isinstance(value, list):
            value = " ".join([str(v) for v in value])

        if isinstance(value, bool):
            value = str(value).lower()

        if value != "":
            arguments.append(f"--{option} {value}")

    arguments = " ".join(arguments)

    # Run the experiment
    cmd = (
        f"PYTHONPATH=$PYTHONPATH:{experiment['code_path']} {experiment['python_bin']} "
        f"parseridge/train.py {arguments}"
    )
    if args.dry_run:
        print(f"Would run: {cmd}")
    else:
        subprocess.run(cmd, cwd=experiment["code_path"], shell=True)
