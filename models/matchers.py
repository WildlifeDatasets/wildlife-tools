import numpy as np
import pandas as pd
import itertools
import pandas as pd
import torch
import kornia.feature as KF
from data.dataset import WildlifeDataset
import math
import os
from tqdm import tqdm
import cv2
import faiss
from tqdm import tqdm
import torch
from models.superpoint import SuperPoint # Superpoint from https://github.com/magicleap/SuperGluePretrainedNetwork


def get_faiss_index(d, device='cpu'):
    if device == 'cuda':
        resource = faiss.StandardGpuResources()
        config = faiss.GpuIndexFlatConfig()
        config.device = 0
        return faiss.GpuIndexFlatL2(resource, d, config)
    elif device == 'cpu':
        return faiss.IndexFlatL2(d)
    else:
        raise ValueError(f'Invalid device: {device}')


def batched(iterable, n):
    '''
    Batch data into tuples of length n. The last batch may be shorter.
    Example: batched('ABCDEFG', 3) --> ABC DEF G
    '''
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch


def prepare_pair_batches(query, database=None, batch_size=128, chunk=1, chunk_total=1):
    '''
    Prepares batches of data pairs given query and optionally the database. 
    
    Returns an iterator that iterates through the batches. Optionally, it can
    be splited to chunks for better memory optimization and parallelization.
    '''
    if database:
        pair_total = len(query)*len(database)
        pair_iterator = itertools.product(enumerate(query), enumerate(database))
    else:
        pair_total = math.comb(len(query), 2)
        pair_iterator = itertools.combinations(enumerate(query), 2)

    batch_total = int(np.ceil(pair_total / batch_size))
    batch_iterator = batched(pair_iterator, batch_size)

    batches = np.array_split(np.arange(batch_total), chunk_total)[chunk-1]
    batch_min, batch_max = batches[0], batches[-1] + 1
    iterator = itertools.islice(batch_iterator, batch_min, batch_max)

    print(f'Total pairs     : {pair_total}')
    print(f'Total batches   : {batch_total}')
    print(f'Batches in chunk: {len(batches)}')
    return iterator, len(batched)


def prepare_pairs(query, database=None, batch_size=128, chunk=1, chunk_total=1):
    '''
    Prepares data pairs given query and optionally the database. 
    
    Returns an iterator that iterates through the batches. Optionally, it can
    be splited to chunks for better memory optimization and parallelization.
    '''
    if database:
        pair_total = len(query)*len(database)
        pair_iterator = itertools.product(enumerate(query), enumerate(database))
    else:
        pair_total = math.comb(len(query), 2)
        pair_iterator = itertools.combinations(enumerate(query), 2)

    pairs = np.array_split(np.arange(pair_total), chunk_total)[chunk-1]
    pair_min, pair_max = pairs[0], pairs[-1] + 1
    iterator = itertools.islice(pair_iterator, pair_min, pair_max)

    print(f'Total pairs     : {pair_total}')
    print(f'Pairs in chunk  : {len(pairs)}')
    return iterator, len(pairs)


def compose_similarity(folders):
    '''
    Create similarity matrix given folders with similarity matrix chunks.
    '''
    data = []
    for folder in folders:
        path = os.path.join(folder, 'similarity.npy')
        data_chunk = np.load(path, allow_pickle='TRUE').item()
        data.append(data_chunk)

    similarity = {}
    for key in data_chunk.keys():
        similarity[key] = np.sum([d[key] for d in data], axis=0)
    return similarity


class LOFTRMatcher():
    def __init__(
        self,
        device: str ='cuda',
        pretrained: str ='outdoor',
        thresholds: tuple[float] = (0.99, ),
        batch_size: int = 128,
        chunk: int = 1,
        chunk_total: int = 1,
    ):
        self.device = device
        self.matcher = KF.LoFTR(pretrained=pretrained).to(device)
        self.thresholds = thresholds
        self.batch_size = batch_size
        if chunk > chunk_total:
            raise ValueError('Current chunk is larger that chunk total.')
        self.chunk = chunk
        self.chunk_total = chunk_total
        self.similarity = None


    def train(
        self,
        dataset_query: WildlifeDataset,
        dataset_database: WildlifeDataset | None = None,
        **kwargs,
    ):
        if dataset_database:
            print('Matching query with database')
            query = [i[0] for i in dataset_query]
            database = [i[0] for i in dataset_database]
            self.similarity = {t: np.zeros((len(query), len(database))) for t in self.thresholds}
        else:
            print('Matching all pairs in query')
            query = [i[0] for i in dataset_query]
            database = None
            self.similarity = {t: np.zeros((len(query), len(query))) for t in self.thresholds}

        iterator, iterator_size = prepare_pair_batches(
            query = query,
            database = database,
            batch_size = self.batch_size,
            chunk = self.chunk,
            chunk_total = self.chunk_total
        )

        for pair_batch in tqdm(iterator, total=iterator_size):
            a, b = zip(*pair_batch)
            a_idx, a_data = list(zip(*a))
            b_idx, b_data = list(zip(*b))
            input_dict = {
                "image0": torch.stack(a_data).to(self.device),
                "image1": torch.stack(b_data).to(self.device),
            }
            with torch.inference_mode():
                correspondences = self.matcher(input_dict)

            batch_idx = correspondences['batch_indexes'].cpu().numpy()
            confidence = correspondences['confidence'].cpu().numpy()
            for t in self.thresholds:
                series = pd.Series(confidence > t)
                for j, group in series.groupby(batch_idx):
                    self.similarity[t][a_idx[j], b_idx[j]] = group.sum()


    def save(self, path, name='similarity.npy'):
        np.save(os.path.join(path, name), self.similarity)

    def load(self, path):
        self.similarity = np.load(path, allow_pickle='TRUE').item()


class DescriptorMatcher():
    def __init__(
        self,
        descriptor_function = None,
        descriptor_dim: int = 128,
        max_keypoints: int | None = None,
        thresholds: tuple[float] = (0.5, ),
        device: str = 'cpu',
        chunk: int = 1,
        chunk_total: int = 1,
    ):
    
        self.descriptor_function = descriptor_function
        self.descriptor_dim = descriptor_dim
        self.max_keypoints = max_keypoints
        self.thresholds = thresholds
        self.device = device
        if chunk > chunk_total:
            raise ValueError('Current chunk is larger that chunk total.')
        self.chunk = chunk
        self.chunk_total = chunk_total
        self.similarity = None

    def get_descriptors(self, dataset):
        if self.descriptor_function:
            return self.descriptor_function(dataset)
        else:
            raise ValueError('No descriptor function provided.')

    def train(
        self,
        dataset_query: WildlifeDataset,
        dataset_database: WildlifeDataset | None = None,
        **kwargs,
    ):
        if dataset_database:
            print('Mode: Matching query with database')
            query = self.get_descriptors(dataset_query)
            database = self.get_descriptors(dataset_database)
            self.similarity = {t: np.zeros((len(query), len(database))) for t in self.thresholds}
        else:
            print('Mode: Matching all pairs in query')
            query = self.get_descriptors(dataset_query)
            database = None
            self.similarity = {t: np.zeros((len(query), len(query))) for t in self.thresholds}

        iterator, iterator_size = prepare_pairs(
            query = query,
            database = database,
            chunk = self.chunk,
            chunk_total = self.chunk_total
        )

        index = get_faiss_index(d=self.descriptor_dim, device=self.device)
        for (a_idx, a_data), (b_idx, b_data) in tqdm(iterator, total=iterator_size):
            if (a_data is None) or (b_data is None):
                continue
            else:
                index.reset()
                index.add(a_data)
                score, idx = index.search(b_data, k=2)
                with np.errstate(divide='ignore'):
                    ratio = score[:, 0] / score[:, 1]
                for t in self.thresholds:
                    self.similarity[t][a_idx, b_idx] = np.sum(ratio < t)


class SIFTMatcher(DescriptorMatcher):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.descriptor_dim = 128

    def get_descriptors(self, dataset):
        if self.max_keypoints:
            sift = cv2.SIFT_create(nfeatures=self.max_keypoints)
        else:
            sift = cv2.SIFT_create()

        descriptors = []
        for img, y in tqdm(dataset):
            keypoint, d = sift.detectAndCompute(np.array(img), None)
            if len(keypoint) <= 1:
                descriptors.append(None)
            else:
                descriptors.append(d)
        return descriptors


class SuperPointMatcher(DescriptorMatcher):
    def __init__(
        self,
        *args, 
        nms_radius: int = 4,
        keypoint_threshold: float = 0.005,
        remove_borders: int = 4,
        **kwargs):

        super().__init__(*args, **kwargs)
        self.descriptor_dim = 256
        self.nms_radius = nms_radius
        self.keypoint_threshold = keypoint_threshold
        self.remove_borders = remove_borders

    def get_descriptors(self, dataset):
        if not self.max_keypoints:
            max_keypoints = -1
        else:
            max_keypoints = self.max_keypoints

        model = SuperPoint(config={
                'descriptor_dim': self.descriptor_dim,
                'nms_radius': self.nms_radius,
                'keypoint_threshold': self.keypoint_threshold,
                'max_keypoints': max_keypoints,
                'remove_borders': self.remove_borders,
        })
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
        )

        descriptors = []
        for image, label in tqdm(loader):
            with torch.no_grad():
                output = model({'image': image})
            descriptors.extend([d.permute(1, 0).cpu().numpy() for d in output['descriptors']])
        return descriptors
