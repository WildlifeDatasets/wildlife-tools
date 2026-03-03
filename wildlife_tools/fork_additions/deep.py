import torch
import numpy as np
from addict import Dict

from wildlife_tools.tools import check_dataset_output
from wildlife_tools.features import DeepFeatures


class ForkedDeepFeatures(DeepFeatures):
    def __call__(self, dataset):
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

        check_dataset_output(dataset, check_label=False)
        loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
        )
        outputs = []
        labels = []
        labels_string = []
        isolations = []
        for image, label, isolation in loader:
            with torch.no_grad():
                output = self.model(image.to(self.device))
                outputs.append(output.cpu())
                label = label.cpu()
                labels.append(label)
                isolations.append(isolation.cpu())
                labels_string.extend([dataset.labels_map[lbl] for lbl in label])

        self.model = self.model.to("cpu")
        features = torch.cat(outputs).numpy()
        labels = torch.cat(labels).numpy()
        isolations = torch.cat(isolations).numpy()
        labels_string = np.array(labels_string)

        return Dict(dict(features=features, labels=labels, isolations=isolations, labels_string=labels_string))
