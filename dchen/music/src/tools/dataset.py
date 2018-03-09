import os
import numpy as np
import torchfile  # pip install torchfile
import arff  # pip install liac-arff


data_dir = 'data'

dataset_names = ['yeast', 'scene', 'bibtex', 'bookmarks']

_yeast_nLabels = 14
_scene_nLabels = 6
_bibtex_nLabels = 159
_bookmarks_nFeatures = 2150
_bookmarks_nLabels = 208

nLabels_dict = {
    'yeast':     _yeast_nLabels,
    'scene':     _scene_nLabels,
    'bibtex':    _bibtex_nLabels,
    'bookmarks': _bookmarks_nLabels,
}


def create_dataset(dataset_name, train_data=True, shuffle=False, random_state=None):
    """
        Load the dataset, and filter out examples with all positive/negative labels,
        examples are shuffled is `shuffle=True`.
        NOTE that the `StratifiedKFold` or `KFold` used by `GridSearchCV` does not shuffle examples by default,
        see `scikit-learn/sklearn/model_selection/_split.py` for details.
    """
    assert dataset_name in dataset_names
    assert type(train_data) == bool
    assert type(shuffle) == bool
    dname = dataset_name
    nlabels = nLabels_dict[dname]
    split = 'train' if train_data else 'test'

    if dname == 'bookmarks':
        X, Y = _load_bookmarks_data(train_data=train_data)
    else:
        fname = os.path.join(data_dir, '%s/%s-%s.arff' % (dname, dname, split))
        data_dict = arff.load(open(fname, 'r'))
        data = np.asarray(data_dict['data'], dtype=np.float)
        X = data[:, :-nlabels]
        Y = data[:, -nlabels:].astype(np.bool)

    # filtering out examples with all positive/negative labels
    kpos = Y.sum(axis=1)
    X1 = X[np.logical_and(kpos > 0, kpos < nlabels), :]
    Y1 = Y[np.logical_and(kpos > 0, kpos < nlabels), :]

    # shuffling
    if shuffle is True:
        if random_state is not None:
            np.random.seed(random_state)
        ind = np.arange(X1.shape[0])
        np.random.shuffle(ind)
        return X1[ind], Y1[ind]
    else:
        return X1, Y1


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
            features = np.concatenate([features, data_dict[b'data'][:, 0:_bookmarks_nFeatures]], axis=0)
            labels = np.concatenate([labels, data_dict[b'labels']], axis=0)

        # load dev data
        data_dict = torchfile.load(os.path.join(data_dir, 'bookmarks/bookmarks-dev.torch'))
        features = np.concatenate([features, data_dict[b'data'][:, 0:_bookmarks_nFeatures]], axis=0)
        labels = np.concatenate([labels, data_dict[b'labels']], axis=0)

    else:
        # load test data
        for k in range(1, 4):
            data_dict = torchfile.load(os.path.join(data_dir, 'bookmarks/bookmarks-test-%d.torch' % k))
            features = np.concatenate([features, data_dict[b'data'][:, 0:_bookmarks_nFeatures]], axis=0)
            labels = np.concatenate([labels, data_dict[b'labels']], axis=0)
    return features, labels.astype(np.bool)
