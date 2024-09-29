import torch
from tqdm import tqdm
from gluefactory.models import get_model
from omegaconf import OmegaConf
from wildlife_tools.data import FeatureDataset, WildlifeDataset
from wildlife_tools.features.base import FeatureExtractor

# Fix https://github.com/cvg/glue-factory/pull/50
import types
from .gluefactory_fix import extract_single_image_fix 


class GlueFactoryExtractor(FeatureExtractor):
    '''
    Args:
        num_workers: Number of workers used for data loading.
        device: Select between cuda and cpu devices.
        max_num_keypoints: Maximum number of keypoints to return.
        detection_threshold: Threshold for keypoints detection (use 0.0 if force_num_keypoints = True).
        force_num_keypoints: Force to return exactly max_num_keypoints keypoints.

    Returns:
        An array with a shape of `n_input` x `dim_embedding`.

    Note:
        Gluefactory extractors requires with 3 channel RBG tensors scaled to [0, 1]. e.g:
        transform = T.Compose([
            T.Resize([224, 224]),
            T.ToTensor(),
        ])

    '''
    def __init__(
        self,
        config: dict,
        device: None | str = None,
        num_workers: int = 1,
    ):
        config = OmegaConf.create(config)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = get_model(config.name)(config)
        self.device = device
        self.num_workers = num_workers

    def __call__(self, dataset: WildlifeDataset):
        features = []
        loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=self.num_workers,
            batch_size=1,
            shuffle=False,
        )

        self.model.to(self.device)
        for image, _ in tqdm(loader, mininterval=1, ncols=100):
            image = image.to(self.device)
            with torch.inference_mode():
                output = self.model({'image': image})
                output = {k: v.squeeze(0).cpu() for k, v in output.items()}
                output['image_size'] = torch.tensor(image.shape[2:])
            features.append(output)

        self.model.to('cpu')
        return FeatureDataset(
            metadata=dataset.metadata,
            features=features,
            col_label=dataset.col_label
        )


class SuperPointExtractor(GlueFactoryExtractor):
    '''
    Superpoint keypoints and descriptor.
    
    "SuperPoint: Self-Supervised Interest Point Detection and Description"
    https://arxiv.org/abs/1712.07629
    '''

    def __init__(
        self,
        detection_threshold = 0.0,
        force_num_keypoints = True,
        max_num_keypoints = 256,
        device: None | str = None,
        **model_config
    ):
        config = {
            "name": "gluefactory_nonfree.superpoint",
            'nms_radius': 3,
            'detection_threshold': detection_threshold,
            'force_num_keypoints': force_num_keypoints,
            'max_num_keypoints': max_num_keypoints,
        } | model_config
        super().__init__(config, device=device)


class DiskExtractor(GlueFactoryExtractor):
    '''
    DISK keypoints and descriptor from 

    "DISK: learning local features with policy gradient" 
    https://arxiv.org/abs/2006.13566
    '''

    def __init__(
        self,
        detection_threshold = 0.0,
        force_num_keypoints = True,
        max_num_keypoints = 256,
        device: None | str = None,
        **model_config
    ):
        config = {
            "name": "extractors.disk_kornia",
            'detection_threshold': detection_threshold,
            'force_num_keypoints': force_num_keypoints,
            'max_num_keypoints': max_num_keypoints,
        } | model_config
        super().__init__(config, device=device)


class AlikedExtractor(GlueFactoryExtractor):
    '''
    ALIKED keypoints and descriptor. 
    
    "ALIKED: A Lighter Keypoint and Descriptor Extraction Network via Deformable Transformation"
    https://arxiv.org/abs/2304.03608
    '''

    def __init__(
        self,
        detection_threshold = 0.0,
        force_num_keypoints = True,
        max_num_keypoints = 256,
        device: None | str = None,
        **model_config
    ):

        config = {
            "name": "extractors.aliked",
            'detection_threshold': detection_threshold,
            'force_num_keypoints': force_num_keypoints,
            'max_num_keypoints': max_num_keypoints,
        } | model_config
        super().__init__(config, device=device)


class SiftExtractor(GlueFactoryExtractor):
    ''' SIFT keypoints and descriptor. '''

    def __init__(
        self,
        backend='opencv',
        detection_threshold = 0.0,
        force_num_keypoints = True,
        max_num_keypoints = 256,
        device: None | str = None,
        **model_config
    ):

        config = {
            "name": "extractors.sift",
            "backend": backend,
            'detection_threshold': detection_threshold,
            'force_num_keypoints': force_num_keypoints,
            'max_num_keypoints': max_num_keypoints,
        } | model_config
        super().__init__(config)

        # Fix extract_single_image method.
        self.model.extract_single_image = types.MethodType(extract_single_image_fix, self.model)
