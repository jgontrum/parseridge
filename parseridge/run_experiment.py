import argparse
import os
import subprocess
from copy import copy

import yaml


def get_args():
    parser = argparse.ArgumentParser(
        description="Helper script to run experiments in a repeatable way.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("experiment", type=str, help="Path to the experiment definition.")

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
            os.makedirs(v, exist_ok=True)

        elif v.endswith("_file"):
            if os.path.exists(v):
                raise Exception(f"File already exists: {v}.")
            os.makedirs(os.path.dirname(v), exist_ok=True)
    #
    # # Clone the code base and switch to the required commit
    subprocess.run(
        f"git clone {experiment['repository']} {experiment['code_path']} &&"
        f"cd {experiment['code_path']} &&"
        f"git checkout --quiet  {experiment['commit']}",
        shell=True,
    )

    # Get all the arguments we want to pass to the trainer
    training_args = {}
    for k, v in dict_generator(experiment_definition):
        if k not in ["repository", "code_path", "commit", "python_bin"]:
            training_args[k] = v

    arguments = []
    for option, value in training_args.items():
        if isinstance(value, list):
            value = ",".join([str(v) for v in value])
        arguments.append(f"--{option}={value}")

    arguments = " ".join(arguments)

    # Run the experiment
    subprocess.run(
        f"cd {experiment['code_path']} &&"
        f"{experiment['python_bin']} parseridge/train.py {arguments}",
        shell=True,
    )
