import torch
from pytorch_metric_learning import losses, distances
from models.losses import ArcFaceLoss, TripletLoss, SoftmaxLoss
import torch.nn as nn
import timm
import torchvision.transforms as T
import os, sys
#sys.path.append(os.path.join('/home/cermavo3/projects/datasets/'))
from wildlife_datasets import splits
import pandas as pd
import itertools
from models.trainers import BasicTrainer
from models.matchers import LOFTRMatcher, SuperPointMatcher, SIFTMatcher
from data.dataset import WildlifeDataset
from inference.evaluate import EmbeddingEvaluate

class BaseFactory():
    methods = {}

    def __init__(self, config):
        config = config.copy()
        method_name = config.pop('method')
        if method_name not in self.methods:
            raise ValueError(f"Invalid method: {method_name}. Need one of: {', '.join(self.methods.keys())}")

        self.method = self.methods[method_name]
        self.config = config

    def __call__(self, *args, **kwargs):
        return self.method(*args, **kwargs)


class TransformsFactory(BaseFactory):
    '''
    Example configs in YAML file:

    transform_train:
        method: 'timm'
        input_size: 224
        is_training: True
        auto_augment: 'rand-m10-n2-mstd1'


    transform_valid:
        method: 'torchvision'
        compose:
            - 'Resize(size=256)'
            - 'ToTensor()'
    '''
    @property
    def methods(self):
        return {
            'timm': self.create_timm_transforms,
            'torchvision': self.create_torchvision_transforms,
        }

    def create_timm_transforms(self, **kwargs):
        return timm.data.transforms_factory.create_transform(**self.config)

    def create_torchvision_transforms(self, **kwargs):
        compose = self.config['compose']
        return T.Compose([eval(s, globals(), T.__dict__) for s in compose])


class BackboneFactory(BaseFactory):
    '''
    Example config in YAML file:
    
    backbone:
        method: 'timm'
        model_name: 'efficientnet_b0'
        pretrained: True
    '''
    @property
    def methods(self):
        return {
            'timm': self.create_timm_backbone,
        }

    def create_timm_backbone(self, num_classes, **kwargs):
        return timm.create_model(num_classes=num_classes, **self.config)


class OptimizerFactory(BaseFactory):
    '''
    Example config in YAML file:

    optimizer:
        method: 'adam'
        lr: 1e-3    
    '''
    @property
    def methods(self):
        return {
            'adam': self.create_adam,
            'sgd': self.create_sgd,
        }

    def create_adam(self, params, **kwargs):
        return torch.optim.Adam(params=params, **self.config)

    def create_sgd(self, params, **kwargs):
        return torch.optim.SGD(params=params, **self.config)


class SchedulerFactory(BaseFactory):
    '''
    Example config in YAML file:

    scheduler:
        method: 'cosine'
        lr: 1e-3
    '''
    @property
    def methods(self):
        return {
            'cosine': self.create_cosine_scheduler,
        }

    def create_cosine_scheduler(self, optimizer, epochs, **kwargs):
        lr = optimizer.defaults.get("lr")
        lr_min = lr * 1e-3 if lr is not None else 1e-5
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_min)


class SplitterFactory(BaseFactory):
    @property
    def methods(self):
        return {
            'open_set': lambda: splits.OpenSetSplit(**self.config),
            'closed_set': lambda: splits.ClosedSetSplit(**self.config),
            'disjoint_set': lambda: splits.DisjointSetSplit(**self.config),
            'full': lambda: splits.FullSplit(**self.config),
        }


class DatasetsFactory(BaseFactory):
    @property
    def methods(self):
        return {
            'folder': self.create_folder_datasets,
        }

    def create_folder_datasets(self, experiment):
        '''
        Create Wildlife datasets given datasets config and splits.
        '''
        metadata = pd.read_csv(os.path.join(self.config['path'], 'annotations.csv'), index_col=False)
        splitter = experiment.splitter()

        datasets = []
        for idx_train, idx_test in splitter.split(metadata):
            idx = {'test': idx_test, 'train': idx_train}

            split_datasets = {}
            for name, cfg in self.config['datasets'].items():

                if cfg.get('transform'):
                    transform_factory = getattr(experiment, cfg['transform'])
                    if not transform_factory:
                        raise ValueError(f"No transform {cfg['transform']} in the experiment")
                else:
                    transform_factory = lambda: None
                if cfg['split'] not in idx:
                    raise ValueError(f"Invalid split name: {cfg['split']}")

                split_datasets[name] = WildlifeDataset(
                            df = metadata.loc[idx[cfg['split']]],
                            root = self.config['path'],
                            transform = transform_factory()
                        )
            datasets.append(split_datasets)
        return datasets





class ObjectiveFactory(BaseFactory):
    '''
    Example:

    objective:
        method: 'arcface'
        margin: 20
        scale: 64

    objective: 
        method: 'ce_loss'
    '''

    @property
    def methods(self):
        return {
            'softmax_loss': self.create_softmax_loss,
            'arcface_loss': self.create_arcface_loss,
            'triplet_loss': self.create_triplet_loss,
            'ce_loss': self.create_ce_loss,
        }

    def create_softmax_loss(self, num_classes, embedding_size, **kwargs):
        return SoftmaxLoss(num_classes, embedding_size)

    def create_arcface_loss(self, num_classes, embedding_size, **kwargs):
        return ArcFaceLoss(num_classes, embedding_size, **self.config)

    def create_triplet_loss(self, **kwargs):
        return TripletLoss(**self.config)

    def create_ce_loss(self, **kwargs):
        return torch.nn.CrossEntropyLoss()


class EvaluateFactory(BaseFactory):
    @property
    def methods(self):
        return {
        'embedding': self.create_embedding_evaluate,
        }

    def create_embedding_evaluate(self, dataset_train, dataset_test, **kwargs):
        return EmbeddingEvaluate(
            dataset_train=dataset_train,
            dataset_test=dataset_test,
            **self.config
        )


class TrainerFactory(BaseFactory):
    @property
    def methods(self):
        return {
        'embedding': self.create_embedding_trainer,
        'classifier': self.create_classifier_trainer,
        'loftr_matcher': self.create_loftr_matcher,
        'sift_matcher': self.create_sift_matcher,
        'superpoint_matcher': self.create_superpoint_matcher,
        }

    def create_classifier_trainer(self, experiment, dataset_train, **kwargs):
        model = experiment.backbone(embedding_size=dataset_train.num_classes)
        objective = experiment.objective()
        optimizer = experiment.optimizer(params=model.parameters())
        if experiment.scheduler:
            scheduler = experiment.scheduler(optimizer, epoch=self.config['epochs'])
        else:
            scheduler = None
        return BasicTrainer(
            model=model,
            objective=objective,
            optimize=optimizer,
            scheduler=scheduler,
            **self.config
        )

    def create_embedding_trainer(self, experiment, dataset_train, **kwargs):
        if 'embedding_size' in self.config:
            embedding_size = self.config['embedding_size']
            model = experiment.backbone(num_classes=embedding_size)
        else:
            model = experiment.backbone(num_classes=0)
            dummy_input = torch.randn(model.default_cfg['input_size']).unsqueeze(0)
            with torch.no_grad():
                embedding_size = model(dummy_input).shape[1]

        objective = experiment.objective(
            embedding_size=embedding_size,
            num_classes=dataset_train.num_classes,
        )
        optimizer = experiment.optimizer(params=itertools.chain(model.parameters(), objective.parameters()))
        if experiment.scheduler:
            scheduler = experiment.scheduler(optimizer, epochs=self.config['epochs'])
        else:
            scheduler = None

        return BasicTrainer(
            model=model,
            objective=objective,
            optimizer=optimizer,
            scheduler=scheduler,
            **self.config
        )

    def create_loftr_matcher(self, *args, **kwargs):
        return LOFTRMatcher(**self.config)

    def create_superpoint_matcher(self, *args, **kwargs):
        return SuperPointMatcher(**self.config)

    def create_sift_matcher(self, *args, **kwargs):
        return SIFTMatcher(**self.config)