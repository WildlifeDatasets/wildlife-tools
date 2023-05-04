import os
import shutil
import argparse
import yaml
from datetime import datetime
from parser.experiment import Experiment

def setup_folder(output_folder, name=''):
    timestamp = datetime.now().strftime("%b%d-%H-%M-%S-%f")[:-2]
    folder = os.path.join(output_folder, timestamp + '_' + name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config .yaml file")
    parser.add_argument("--output", type=str, default="runs", help="Path to output folder")
    args = parser.parse_args()

    name, _ = os.path.splitext(os.path.basename(args.config))
    folder = setup_folder(args.output, name)
    shutil.copyfile(args.config, os.path.join(folder, os.path.basename(args.config)))

    with open(args.config, 'r') as stream:
        experiment = Experiment(yaml.safe_load(stream))

    for i, datasets in enumerate(experiment.datasets(experiment)):
        trainer = experiment.trainer(experiment, **datasets)
        trainer.train(**datasets, evaluation=None)

        folder_split = os.path.join(folder, f"split-{i}")
        if not os.path.exists(folder_split):
           os.makedirs(folder_split)
        trainer.save(folder_split)
