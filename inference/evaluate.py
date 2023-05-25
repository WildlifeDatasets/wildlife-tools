from wildlife_datasets import metrics
import torch
import torch.nn.functional as F
import numpy as np

def nn_classifier(distance, train_labels):
    scores, idx = torch.tensor(distance).topk(k=1, dim=1, largest=False)
    return train_labels[idx].flatten(), scores.flatten()


def topk_rank(distance, train_labels, k=10, ignore_diag=False):
    if k >= distance.shape[1]:
        raise ValueError('k must be lesser or equal than number of database samples.')

    if ignore_diag:
        if k > distance.shape[1]:
            raise ValueError('k must be lesser than number of database samples.')
        if distance.shape[0] != distance.shape[1]:
            raise ValueError('Distance matrix must be square matrix')
        distance = distance.copy()
        np.fill_diagonal(distance, np.inf)

    _, idx = torch.tensor(distance).topk(k=k, dim=1, largest=False)
    return train_labels[idx]


def evaluate_closed(distance, train_labels, test_labels):
    prediction, score = nn_classifier(distance, train_labels)
    return {
        'acc': metrics.accuracy(test_labels, prediction),
    }

def evaluate_open(distance, train_labels, test_labels):
    new = np.array(list(set(test_labels) - set(train_labels)))
    test_labels = ['new' if x in new else x for x in test_labels]

    prediction, score = nn_classifier(distance, train_labels)
    return {
        'acc_known': metrics.accuracy_known_samples(test_labels, prediction, 'new'),
        'aucroc_new': metrics.auc_roc_new_class(test_labels, -score, 'new'),
    }

def evaluate_disjoint(distance, test_labels, k=10):
    topk = topk_rank(distance, test_labels, k=k, ignore_diag=True)
    return {
        f'map@{k}': metrics.mean_average_precision(test_labels, topk)
    }



def cosine_distance(self, a, b):
    return 1 - torch.matmul(F.normalize(a), F.normalize(b).T)

    
class EmbeddingEvaluate():
    '''Nearest neigbour evaluator'''
    def __init__(self, dataset_train, dataset_test, split, epoch_step=1, print_console=True, distance='cosine'):
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.split = split
        self.epoch_step = 1
        self.print_console = print_console

    def log(self, trainer, metrics=None):
        if metrics is None:
            metrics = {}

        if trainer.epoch % self.epoch_step == 0:
            metrics.update(self.evaluate(trainer) )

        if self.print_console:
            print(f'Epoch: {epoch} - {metrics}')


    def evaluate(self, trainer):
        labels_train = self.dataset_train.df.identity.values
        labels_test = self.dataset_test.df.identity.values
        
        if self.split == 'disjoint':
            pred = {'test': trainer.predict(self.dataset_test)}
            distance = cosine_distance(pred['test'], pred['test'])
            return evaluate_disjoint(distance, labels_test)

        elif self.split == 'closed':
            pred = {'train': trainer.predict(self.dataset_train), 'test': trainer.predict(dataset_test)}
            distance = cosine_distance(pred['test'], pred['train'])
            return evaluate_closed(distance, labels_train, labels_test)

        elif self.split == 'open':
            pred = {'train': trainer.predict(self.dataset_train), 'test': trainer.predict(self.dataset_test)}
            distance = cosine_distance(pred['test'], pred['train'])
            return evaluate_open(distance, labels_train, labels_test)
        else:
            raise ValueError(f'Invalid split {self.split}')

