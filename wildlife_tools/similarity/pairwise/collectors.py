import numpy as np
import cv2


class CollectAll:
    ''' Collect all matcher results as list of tuples with all output data. '''

    def __init__(self, **kwargs):
        self.data = None


    def init_store(self, **kwargs):
        self.data = []


    def add(self, results_list):
        self.data.extend(results_list)


    def process_results(self):
        return self.data


class CollectCounts:
    '''
    Collect count of significant matches given confidence thresholds.
    Output is stored in dictionary with [n_query x n_database] grid for each threshold.
    '''

    def __init__(
        self,
        grid_dtype: str = 'float16',
        thresholds: tuple = (0.5, ),
        **kwargs
    ):

        self.data = None
        self.grid_shape = None
        self.grid_dtype = grid_dtype
        self.thresholds = thresholds


    def init_store(self, grid_shape: tuple| None = None):
        if grid_shape is not None:
            self.grid_shape = grid_shape
            self.data = {t: np.full(grid_shape, np.nan, dtype=self.grid_dtype) for t in self.thresholds}
        else:
            self.data = {'idx0': [], 'idx1': []} + {t: [] for t in self.thresholds}


    def add(self, results_list):
        for item in results_list:
            i0, i1, scores = item['idx0'], item['idx1'], item['scores']

            if self.grid_shape is not None:
                for t in self.thresholds:
                    self.data[t][i0, i1] = np.sum(scores > t)

            else:
                self.data['idx0'].append(i0)
                self.data['idx1'].append(i1)
                for t in self.thresholds:
                    self.data[t].append(np.sum(scores > t))


    def process_results(self):
        ''' if dictionary have one key, return only value. '''
        if len(self.data) == 1:
            return list(self.data.values())[0]
        else:
            return self.data


class CollectCountsRansac(CollectCounts):
    ''' Collect count of RANSAC inliers of fundamental matrix estimate. '''

    def __init__(
        self,
        grid_dtype='float16',
        ransacReprojThreshold = 1.0,
        method = cv2.USAC_MAGSAC,
        confidence = 0.999,
        maxIters = 100,
        **kwargs
    ):
        self.data = None
        self.grid_shape = None
        self.grid_dtype = grid_dtype
        self.config = {
            'ransacReprojThreshold': ransacReprojThreshold,
            'method': method,
            'confidence': confidence,
            'maxIters': maxIters
        }

        #config_str = '_'.join([f'{k}={v}' for k, v in self.config.items()])
 
    def init_store(self, grid_shape: tuple| None = None):
        if grid_shape is not None:
            self.grid_shape = grid_shape
            self.data = {'score': np.full(grid_shape, np.nan, dtype=self.grid_dtype)}
        else:
            self.data = {'idx0': [], 'idx1': [], 'score': []}



    def add(self, results_list):
        for item in results_list:
            i0, i1, kpts0, kpts1 = item['idx0'], item['idx1'], item['kpts0'], item['kpts1']

            if (len(kpts0) < 8) or (len(kpts1) < 8): # findFundamentalMat needs 8 + samples
                score = 0
            else:
                F, mask = cv2.findFundamentalMat(kpts0, kpts1, **self.config)
                score = np.sum(mask == 1)

            if self.grid_shape is not None:
                self.data['score'][i0, i1] = score
            else:
                self.data['idx0'].append(i0)
                self.data['idx1'].append(i1)
                self.data['score'].append(score)
