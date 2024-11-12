from tqdm import tqdm

from ..data import FeatureDataset, ImageDataset


class DataToMemory:
    """
    Loads the dataset into memory for quicker access

    Ideal for LOFTR, which operates directly on images, because loading images from storage can
    become a bottleneck when matching all query-database pairs, requiring n_query x n_database
    image loads.

    """

    def __call__(self, dataset: ImageDataset):
        """Loads data from input dataset into array and returns them as a new FeatureDataset."""

        features = []
        for x, _ in tqdm(dataset, mininterval=1, ncols=100):
            features.append(x)
        return FeatureDataset(
            metadata=dataset.metadata,
            features=features,
            col_label=dataset.col_label,
        )
