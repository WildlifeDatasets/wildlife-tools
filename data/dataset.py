import pycocotools.mask as mask_coco
import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image


class WildlifeDataset():
    '''
    PyTorch-style dataset for a wildlife image classification task.

    Args:
        df: A pandas dataframe containing image metadata.
        root: Root directory for images.
        transform: A function/transform that takes in an image and returns a transformed version.
        img_load: Method to load images. Valid options are 'full', 'full_mask', 'full_hide', 'bbox', 'bbox_mask', 'bbox_hide', and 'crop_black'.
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

    def __init__(self, df, root='.', transform=None, img_load='full', col_path='path', col_identity='identity'):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.transform = transform
        self.img_load = img_load
        self.label, self.label_map = pd.factorize(df[col_identity].values)
        self.col_path = col_path

    def get_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        return img

    @property
    def num_classes(self):
        return len(self.label_map)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        img = self.get_image(os.path.join(self.root, data[self.col_path]))
        
        # Load full image as it is.
        if self.img_load == 'full':
            img = img

        # Mask background using segmentation mask.
        elif self.img_load == 'full_mask':
            mask = mask_coco.decode(data['segmentation']).astype('bool')
            img = Image.fromarray(img * mask[..., np.newaxis])
        
        # Hide object using segmentation mask
        elif self.img_load == 'full_hide':
            mask = mask_coco.decode(data['segmentation']).astype('bool')
            img = Image.fromarray(img * ~mask[..., np.newaxis])

        # Crop to bounding box
        elif self.img_load == 'bbox':
            bbox = data['bbox']
            img = img.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))

        # Mask background using segmentation mask and crop to bounding box.
        elif self.img_load == 'bbox_mask':
            mask = mask_coco.decode(data['segmentation']).astype('bool')
            img = Image.fromarray(img * mask[..., np.newaxis])
            bbox = data['bbox']
            img = img.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))

        # Hide object using segmentation mask and crop to bounding box.
        elif self.img_load == 'bbox_hide':
            mask = mask_coco.decode(data['segmentation']).astype('bool')
            img = Image.fromarray(img * ~mask[..., np.newaxis])
            bbox = data['bbox']
            img = img.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))

        # Crop black background around images
        elif self.img_load == 'crop_black':
            y_nonzero, x_nonzero, _ = np.nonzero(img)
            img = img.crop((np.min(x_nonzero), np.min(y_nonzero), np.max(x_nonzero), np.max(y_nonzero)))

        else:
            raise ValueError(f'Invalid img_load argument: {self.img_load}')

        if self.transform:
            img = self.transform(img)
        return img, self.label[idx]
