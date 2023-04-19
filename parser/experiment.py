from parser.factories import *

class Experiment():
    factories = {
        'optimizer': OptimizerFactory,
        'objective': ObjectiveFactory,
        'miner': MinerFactory,
        'scheduler': lambda optimizer, **kwargs: None, # TODO:
        'transform': TransformsFactory,
        'backbone': BackboneFactory,
        'trainer': TrainerFactory,
        'splitter': SplitterFactory,
        'datasets': DatasetsFactory,
    }

    def __init__(self, config):
        self.config = config
        for name, values in config.items():
            try:
                if name.startswith('transform'):
                    setattr(self, name, self.factories['transform'](values))
                else:
                    setattr(self, name, self.factories[name](values))
            except KeyError:
                raise ValueError(f'Cannot process attribute: {name}')

    def __getattr__(self, name):
        if name not in self.__dict__:
            return None
