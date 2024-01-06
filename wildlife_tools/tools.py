import importlib
import os
import random
from copy import deepcopy

import numpy as np
import torch
import torch.backends.cudnn
import yaml
from jinja2 import Environment, meta


class Store:
    """Dictionary with components. Components are imported as needed."""

    modules = {
        # Data
        "SplitChunk": ("wildlife_tools.data.split", "SplitChunk"),
        "SplitWildlife": ("wildlife_tools.data.split", "SplitWildlife"),
        "SplitMetadata": ("wildlife_tools.data.split", "SplitMetadata"),
        "SplitChain": ("wildlife_tools.data.split", "SplitChain"),
        "TransformTimm": ("wildlife_tools.data.transform", "TransformTimm"),
        "TransformTorchvision": (
            "wildlife_tools.data.transform",
            "TransformTorchvision",
        ),
        "WildlifeDataset": ("wildlife_tools.data.dataset", "WildlifeDataset"),
        "FeatureDataset": ("wildlife_tools.data.dataset", "FeatureDataset"),
        # Features & Similarity
        "SimilarityPipeline": (
            "wildlife_tools.pipelines.similarity",
            "SimilarityPipeline",
        ),
        # Train
        "TrainingPipeline": ("wildlife_tools.pipelines.train", "TrainingPipeline"),
        "EpochCallbacks": ("wildlife_tools.train.callbacks", "EpochCallbacks"),
        "EpochLog": ("wildlife_tools.train.callbacks", "EpochLog"),
        "EpochCheckpoint": ("wildlife_tools.train.callbacks", "EpochCheckpoint"),
        "EmbeddingTrainer": ("wildlife_tools.train.trainer", "EmbeddingTrainer"),
        "ClassifierTrainer": ("wildlife_tools.train.trainer", "ClassifierTrainer"),
        "OptimizerAdam": ("wildlife_tools.train.optim", "OptimizerAdam"),
        "OptimizerSGD": ("wildlife_tools.train.optim", "OptimizerSGD"),
        "SchedulerCosine": ("wildlife_tools.train.optim", "SchedulerCosine"),
        "TimmBackbone": ("wildlife_tools.train.backbone", "TimmBackbone"),
        "ArcFaceLoss": ("wildlife_tools.train.objective", "ArcFaceLoss"),
        "TripletLoss": ("wildlife_tools.train.objective", "TripletLoss"),
        "SoftmaxLoss": ("wildlife_tools.train.objective", "SoftmaxLoss"),
    }

    def __getitem__(self, key):
        module_path, class_name = self.modules[key]
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def __contains__(self, key):
        return key in self.modules


def realize(config, store=None, **kwargs):
    """realize object from config given its name and method store"""

    if config is None:
        return None

    if store is None:
        store = Store()

    config_local = deepcopy(config)
    name = config_local.pop("method", None)

    if name is None:
        raise ValueError("No 'method' attribute in config")
    if name not in store:
        raise ValueError(f"No method {name} in method store.")

    obj_class = store[name]
    if hasattr(obj_class, "from_config"):
        obj = obj_class.from_config(config=config_local, **kwargs)
    else:
        obj = obj_class(**config_local, **kwargs)

    obj.source_config = deepcopy(config)
    return obj


def parse_yaml(yaml_string):
    '''
    Impute variables with in "{{ }}" with values from top level of the yaml dictionary.
    Example:

    yaml_string = """
    a:
        name: test
    b:
        uses: "{{a}}"
    """
    parse_yaml(yaml_string)
    >>> {'a': {'name': 'test'}, 'b': {'uses': {'name': 'test'}}}
    '''

    env = Environment(
        variable_start_string="'{{",
        variable_end_string="}}'",
    )

    data = yaml.safe_load(yaml_string)
    data_str = str(data)
    variables = meta.find_undeclared_variables(env.parse(data_str))

    while len(variables) != 0:
        data_str = env.from_string(data_str).render(data)
        data = yaml.safe_load(data_str)
        variables_new = meta.find_undeclared_variables(env.parse(data_str))

        # Check if number of resolved variables changed
        if len(variables) == len(variables_new):
            raise ValueError(f"Unable to impute variables in all places: {variables}")

        variables = variables_new
    return data


def set_seed(seed=0, device="cuda"):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
