import pycocotools.mask as mask_coco
import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image
import json
from wildlife_tools.tools import realize


class WildlifeDataset():
    '''
    PyTorch-style dataset for a wildlife image classification task.

    Args:
        metadata: A pandas dataframe containing image metadata.
        root: Root directory for images.
        transform: A function/transform that takes in an image and returns a transformed version.
        img_load: Method to load images. 
            Options: 'full', 'full_mask', 'full_hide', 'bbox', 'bbox_mask', 'bbox_hide', and 'crop_black'.
        col_path: Column name in the dataframe containing image file paths.
        col_identity: Column name in the dataframe containing class labels.

    Attributes:
        label: An array of integer labels for all images in the dataset.
        label_map: A mapping between integer label and string label.
            
    Methods:
        get_image(path): Load an image from file and convert it to a PIL Image object.
        
    Properties:
        num_classes: Return the number of unique classes in the dataset.
        __len__: Return the number of samples in the dataset.
        __getitem__(idx): Return (sample, label) tuple. Get a sample from the dataset at a given index.
    '''

    def __init__(
        self,
        metadata,
        root,
        split=None,
        transform=None,
        img_load='full',
        col_path='path',
        col_label='identity',
        load_label=True,
    ):
        self.split = split
        if self.split:
            metadata = self.split(metadata)

        self.metadata = metadata.reset_index(drop=True)
        self.root = root
        self.transform = transform
        self.img_load = img_load
        self.col_path = col_path
        self.col_label = col_label
        self.load_label = load_label
        self.labels, self.labels_map = pd.factorize(self.metadata[self.col_label].values)

    @property
    def labels_string(self):
        return self.metadata[self.col_label].astype(str).values

    @property
    def num_classes(self):
        return len(self.labels_map)


    def __len__(self):
        return len(self.metadata)


    def get_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        return img


    def __getitem__(self, idx):
        data = self.metadata.iloc[idx]
        img = self.get_image(os.path.join(self.root, data[self.col_path]))
        
        if self.img_load in ['full_mask', 'full_hide', 'bbox_mask', 'bbox_hide']:
            if not ('segmentation' in data):
                raise ValueError(f"{self.img_load} selected but no segmentation found.")
            if type(data['segmentation']) == str:
                segmentation = eval(data['segmentation'])
            else:
                segmentation = data['segmentation']

        if self.img_load in ['bbox', 'bbox_mask', 'bbox_mask']:
            if not ('bbox' in data):
                raise ValueError(f"{self.img_load} selected but no bbox found.")
            if type(data['bbox']) == str:
                bbox = json.loads(data['bbox'])
            else:
                bbox = data['bbox']

        # Load full image as it is.
        if self.img_load == 'full':
            img = img

        # Mask background using segmentation mask.
        elif self.img_load == 'full_mask':
            mask = mask_coco.decode(segmentation).astype('bool')
            img = Image.fromarray(img * mask[..., np.newaxis])
        
        # Hide object using segmentation mask
        elif self.img_load == 'full_hide':
            mask = mask_coco.decode(segmentation).astype('bool')
            img = Image.fromarray(img * ~mask[..., np.newaxis])

        # Crop to bounding box
        elif self.img_load == 'bbox':
            img = img.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))

        # Mask background using segmentation mask and crop to bounding box.
        elif self.img_load == 'bbox_mask':
            mask = mask_coco.decode(segmentation).astype('bool')
            img = Image.fromarray(img * mask[..., np.newaxis])
            img = img.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))

        # Hide object using segmentation mask and crop to bounding box.
        elif self.img_load == 'bbox_hide':
            mask = mask_coco.decode(segmentation).astype('bool')
            img = Image.fromarray(img * ~mask[..., np.newaxis])
            img = img.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))

        # Crop black background around images
        elif self.img_load == 'crop_black':
            y_nonzero, x_nonzero, _ = np.nonzero(img)
            img = img.crop((np.min(x_nonzero), np.min(y_nonzero), np.max(x_nonzero), np.max(y_nonzero)))

        else:
            raise ValueError(f'Invalid img_load argument: {self.img_load}')

        if self.transform:
            img = self.transform(img)


        if self.load_label:
            return img, self.labels[idx]
        else:
            return img


    @classmethod
    def from_config(cls, config):
        config['split'] = realize(config.get('split'))
        config['transform'] = realize(config.get('transform'))
        config['metadata'] = pd.read_csv(config['metadata'], index_col=False)
        return cls(**config)

    


class FeatureDataset():
    def __init__(
        self,
        features,
        metadata,
        col_label='identity',
        load_label=True,
    ):

        if len(features) != len(metadata):
            raise ValueError('Features and metadata (lables) have different length ! ') 

        self.load_label = load_label
        self.features = features
        self.metadata = metadata.reset_index(drop=True)
        self.col_label=col_label
        self.labels, self.labels_map = pd.factorize(self.metadata[self.col_label].values)


    def save(self, path):
        data = {
            'features': self.features,
            'metadata': self.metadata,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


    @classmethod
    def load(cls, path, **config):
        with open(path, 'rb') as file:
            data = pickle.load(file)
        return cls(**data, **config)


    def __getitem__(self, idx):
        if self.load_label:
            return self.features[idx], self.labels[idx]
        else:
            return self.features[idx]


    def __len__(self):
        return len(self.metadata)


    @property
    def num_classes(self):
        return len(self.labels_map)


    @classmethod
    def from_config(cls, config):
        path = config.pop('path')
        return cls.load(path, **config)
