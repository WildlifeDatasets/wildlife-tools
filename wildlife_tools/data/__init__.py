'''
from data.split import SplitChunk, SplitSplitter, SplitMetadata, SplitPipeline
from data.transform import TransformTimm, TransformTorchvision
from data.dataset import WildlifeDataset

split_store = {
    'SplitChunk': SplitChunk,
    'SplitSplitter': SplitSplitter,
    'SplitMetadata': SplitMetadata,
    'SplitPipeline': SplitPipeline,
}

transform_store = {
    'TransformTimm': TransformTimm,
    'TransformTorchvision': TransformTorchvision,
}


dataset_store = {
    'WildlifeDataset': WildlifeDataset,
    #'FeatureDataset': FeatureDataset,
}


'''