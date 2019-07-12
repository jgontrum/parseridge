import argparse
import os
import subprocess

import yaml


def get_args():
    parser = argparse.ArgumentParser(
        description="Helper script to run experiments in a repeatable way.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("experiment", type=str, help="Path to the experiment definition.")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    experiment_definition = yaml.safe_load(open(args.experiment))

    experiment = experiment_definition["experiment"]

    # Fill in the templates in the path values
    experiment["code_path"] = experiment["code_path"].format(**experiment)
    experiment["model_save_path"] = experiment["model_save_path"].format(**experiment)
    experiment["csv_output_path"] = experiment["csv_output_path"].format(**experiment)

    if os.path.exists(experiment["code_path"]):
        raise Exception(f"Folder already exists: {experiment['code_path']}.")

    if os.path.exists(experiment["model_save_path"]):
        raise Exception(f"Folder already exists: {experiment['model_save_path']}.")

    if os.path.exists(experiment["csv_output_path"]):
        raise Exception(f"File already exists: {experiment['csv_output_path']}.")

    # Create the directories
    os.makedirs(experiment["code_path"], exist_ok=True)
    os.makedirs(experiment["model_save_path"], exist_ok=True)
    os.makedirs(os.path.dirname(experiment["csv_output_path"]), exist_ok=True)

    # Clone the code base and switch to the required commit
    subprocess.run(
        f"git clone {experiment['repository']} {experiment['code_path']} &&"
        f"cd {experiment['code_path']} &&"
        f"git checkout {experiment['commit']}",
        shell=True,
    )

    # TODO start parser
