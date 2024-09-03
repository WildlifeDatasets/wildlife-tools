import os
import pickle
from wildlife_tools.data.datasets import WildlifeDataset, FeatureDataset


class FeatureExtractor:
    def __call__(self, dataset: WildlifeDataset) -> FeatureDataset:
        raise NotImplementedError()

    def run_and_save(self, dataset, save_path):
        feature_dataset = self(dataset)

        os.makedirs(save_path, exist_ok=True)
        name = self.__class__.__name__
        data = {
            "name": name,
            "features": feature_dataset.features,
            "metadata": feature_dataset.metadata,
            "col_label": feature_dataset.col_label,
        }

        file_name = os.path.join(save_path, name + ".pkl")
        with open(file_name, "wb") as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
        return file_name
