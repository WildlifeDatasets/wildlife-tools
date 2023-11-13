import os
import shutil
import argparse
import yaml
from datetime import datetime
from tools import parse_yaml, realize, set_seed


def setup_folder(config_filename, root='runs', name=''):
    name, _ = os.path.splitext(os.path.basename(config_filename))
    timestamp = datetime.now().strftime("%b%d-%H-%M-%S-%f")[:-2]
    return os.path.join(root, name + '_' + timestamp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config .yaml file")
    args = parser.parse_args()


    print('Parsing config file')    
    with open(args.config, 'r') as stream:
        config_dict = parse_yaml(stream)
        #config_dict = yaml.safe_load(stream)
        pipeline = realize('pipeline', config_dict)

    shutil.copyfile(args.config, os.path.join(folder, os.path.basename(args.config)))


    print('Creating output folder')
    if pipeline.workdir is None:
        folder = setup_folder(args.config)
        pipeline.workdir = folder
    else:
        folder = pipeline.workdir
    os.makedirs(folder, exist_ok=True)


    print('Running experiment')
    set_seed(pipeline.seed)
    pipeline.run()
