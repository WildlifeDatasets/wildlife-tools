import os
import shutil
import argparse
import yaml
from parser.experiment import Experiment
from utils import setup_folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config .yaml file")
    parser.add_argument("--output", type=str, default=None, help="Path to output folder")
    args = parser.parse_args()
    if args.output is None:
        folder = setup_folder(args.config)
    else:
        folder = args.output


    print('Running: ', args.config)
    shutil.copyfile(args.config, os.path.join(folder, os.path.basename(args.config)))
    with open(args.config, 'r') as stream:
        config_dict = yaml.safe_load(stream)
        experiment = Experiment(config_dict)

    for i, datasets in enumerate(experiment.datasets(experiment)):
        trainer = experiment.trainer(experiment, **datasets)
        if experiment.evaluation:
            evaluation = experiment.evaluation(datasets=datasets)
        else:
            evaluation = None
        trainer.train(**datasets, evaluation=experiment.evaluation)

        folder_split = os.path.join(folder, f"split-{i}")
        if not os.path.exists(folder_split):
           os.makedirs(folder_split)
        trainer.save(datasets=datasets, folder=folder_split)
