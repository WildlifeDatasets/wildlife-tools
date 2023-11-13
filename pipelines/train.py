from tools import realize
from copy import deepcopy


class TrainingPipeline():
    def __init__(
        self,
        trainer,
        workdir='.',
        seed=0,
    ):
        self.trainer = trainer
        self.workdir = workdir
        self.seed = seed

    def __call__(self):
        print('Training')
        self.trainer.train()
        
        print('Saving Final Checkpoint')
        self.trainer.save(folder=self.workdir, checkpoint_file=f"checkpoint-final.pth")

        print('Done - training pipeline')


    @classmethod
    def from_config(cls, config):
        config = deepcopy(config)
        config['trainer'] = realize(config['trainer'])
        return cls(**config)
                