import os
import pickle


class FeatureExtractor:
    def __call__(self, dataset):
        raise NotImplementedError()

    def run_and_save(self, dataset, save_path):
        features = self(dataset)

        os.makedirs(save_path, exist_ok=True)
        name = self.__class__.__name__
        data = {
            "name": name,
            "features": features,
            "metadata": getattr(dataset, "metadata", None),
        }

        file_name = os.path.join(save_path, name + ".pkl")
        with open(file_name, "wb") as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
        return file_name
