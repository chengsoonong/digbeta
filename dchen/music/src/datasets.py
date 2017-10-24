import os
import torchfile
import numpy as np
import pickle as pkl
from scipy.io import arff


data_dir = 'data'

dataset_names = ['yeast', 'scene', 'emotions', 'bibtex', 'bookmarks', 'delicious', 'mediamill']

_yeast_nLabels = 14
_scene_nLabels = 6
_emotions_nLabels = 6
_bibtex_nLabels = 159
_bookmarks_nFeatures = 2150
_bookmarks_nLabels = 208
_delicious_nLabels = 983
_mediamill_nLabels = 101

nLabels_dict = {
    'yeast':     _yeast_nLabels,
    'scene':     _scene_nLabels,
    'emotions':  _emotions_nLabels,
    'bibtex':    _bibtex_nLabels,
    'bookmarks': _bookmarks_nLabels,
    'delicious': _delicious_nLabels,
    'mediamill': _mediamill_nLabels,
}


# def create_dataset_per_label(label_ix, dataset_name, train_data=True):
#     """
#         Create a dataset for each label
#     """
#     assert dataset_name in dataset_names
#     #assert type(label_ix) in [int, np.int]
#     assert isinstance(label_ix, numbers.Integral)
#     assert label_ix >= 0
#     assert label_ix < nLabels_dict[dataset_name]
#     assert type(train_data) == bool
#     dname = dataset_name
#     X, Y = create_dataset(dname, train_data=train_data)
#     return X, Y[:, label_ix]


def create_dataset(dataset_name, train_data=True):
    """
        Load the dataset, and filter out examples with all positive/negative labels
    """
    assert dataset_name in dataset_names
    assert type(train_data) == bool
    dname = dataset_name
    split = 'train' if train_data else 'test'

    if dname == 'bookmarks':
        return _load_bookmarks_data(train_data=train_data)
    else:
        if dname in ['bibtex', 'delicious']:
            fname = os.path.join(data_dir, '%s/%s-%s.pkl' % (dname, dname, split))
            data_dict = pkl.load(open(fname, 'rb'))
            data = data_dict['data']
        else:
            fname = os.path.join(data_dir, '%s/%s-%s.arff' % (dname, dname, split))
            data, meta = arff.loadarff(fname)
        X, Y = _create_dataset(data, nLabels_dict[dname])
        kpos = Y.sum(axis=1)
        return X[np.logical_and(kpos > 0, kpos < nLabels_dict[dname]), :], \
            Y[np.logical_and(kpos > 0, kpos < nLabels_dict[dname]), :]


def _load_bookmarks_data(train_data=True):
    """
        Load (pre-split) bookmarks dataset, no examples with all positive/negative labels (checked)
    """
    features = np.zeros((0, _bookmarks_nFeatures))
    labels = np.zeros((0, _bookmarks_nLabels))

    if train_data is True:
        # load train data
        for k in range(1, 6):
            data_dict = torchfile.load(os.path.join(data_dir, 'bookmarks/bookmarks-train-%d.torch' % k))
            labels = np.concatenate([labels, data_dict[b'labels']], axis=0)
            features = np.concatenate([features, data_dict[b'data'][:, 0:_bookmarks_nFeatures]], axis=0)

        # load dev data
        data_dict = torchfile.load(os.path.join(data_dir, 'bookmarks/bookmarks-dev.torch'))
        labels = np.concatenate([labels, data_dict[b'labels']], axis=0)
        features = np.concatenate([features, data_dict[b'data'][:, 0:_bookmarks_nFeatures]], axis=0)

    else:
        # load test data
        for k in range(1, 4):
            data_dict = torchfile.load(os.path.join(data_dir, 'bookmarks/bookmarks-test-%d.torch' % k))
            labels = np.concatenate([labels, data_dict[b'labels']], axis=0)
            features = np.concatenate([features, data_dict[b'data'][:, 0:_bookmarks_nFeatures]], axis=0)
    return features, labels


def _create_dataset(data, nLabels):
    """
        Create the labelled dataset for a given label index

        Input:
            - data: original data with features + labels

        Output:
            - (Feature, Label) pair (X, y)
              X comprises the features for each example
              Y comprises the labels of the corresponding example
    """

    assert nLabels > 0
    assert type(nLabels) == int

    N = len(data)
    D = len(data[0]) - nLabels
    L = nLabels
    magic = -nLabels

    X = np.zeros((N, D), dtype=np.float)
    Y = np.zeros((N, L), dtype=np.int)

    for i in range(N):
        X[i, :] = list(data[i])[:magic]
        Y[i, :] = list(data[i])[magic:]

    return X, Y
