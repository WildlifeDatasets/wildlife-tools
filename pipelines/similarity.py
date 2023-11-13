from tools import realize
from copy import deepcopy


class SimilarityPipeline():
    def __init__(
        self,
        query,
        database,
        feature_extractor,
        similarity,
        workdir='.',
        seed=0,
    ):
        self.query = query
        self.database = database
        self.feature_extractor = feature_extractor
        self.similarity = similarity
        self.workdir = workdir
        self.seed = seed
    
    def __call__(self):
        print('Extracting database features')
        features_database = self.feature_extractor(self.database)

        print('Extracting query features')
        features_query = self.feature_extractor(self.query)

        print('Running similarity method')
        self.similarity(query=features_query, database=features_database, path=self.workdir)

        print('Done - similarity pipeline')


    @classmethod
    def from_config(cls, config):
        config = deepcopy(config)
        config['query'] = realize(config['query'])
        config['database'] = realize(config['database'])
        config['feature_extractor'] = realize(config['feature_extractor'])
        config['similarity'] = realize(config['similarity'])
        return cls(**config)