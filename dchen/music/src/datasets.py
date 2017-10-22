import os
import torchfile
import numpy as np
import pickle as pkl
from scipy.io import arff


data_dir = 'data'

yeast_ftrain = os.path.join(data_dir, 'yeast/yeast-train.arff')
yeast_ftest = os.path.join(data_dir, 'yeast/yeast-test.arff')
yeast_nLabels = 14

scene_ftrain = os.path.join(data_dir, 'scene/scene-train.arff')
scene_ftest = os.path.join(data_dir, 'scene/scene-test.arff')
scene_nLabels = 6

emotions_ftrain = os.path.join(data_dir, 'emotions/emotions-train.arff')
emotions_ftest = os.path.join(data_dir, 'emotions/emotions-test.arff')
emotions_nLabels = 6

bibtex_ftrain = os.path.join(data_dir, 'bibtex/bibtex-train.pkl')
bibtex_ftest = os.path.join(data_dir, 'bibtex/bibtex-test.pkl')
bibtex_nLabels = 159

delicious_ftrain = os.path.join(data_dir, 'delicious/delicious-train.pkl')
delicious_ftest = os.path.join(data_dir, 'delicious/delicious-test.pkl')
delicious_nLabels = 983

mm_ftrain = os.path.join(data_dir, 'mediamill/mediamill-train.arff')
mm_ftest = os.path.join(data_dir, 'mediamill/mediamill-test.arff')
mm_nLabels = 101

SEED = 123456789
RATIO = 0.05

bookmarks_nFeatures = 2150
bookmarks_nLabels = 208


# The yeast dataset

def create_dataset_per_label_yeast_train(label_ix):
    data, meta = arff.loadarff(yeast_ftrain)
    return create_dataset_per_label(label_ix, data, yeast_nLabels)


def create_dataset_per_label_yeast_test(label_ix):
    data, meta = arff.loadarff(yeast_ftest)
    return create_dataset_per_label(label_ix, data, yeast_nLabels)


def create_dataset_yeast_train():
    data, meta = arff.loadarff(yeast_ftrain)
    return create_dataset(data, yeast_nLabels)


def create_dataset_yeast_test():
    data, meta = arff.loadarff(yeast_ftest)
    return create_dataset(data, yeast_nLabels)


# The scene dataset

def create_dataset_per_label_scene_train(label_ix):
    data, meta = arff.loadarff(scene_ftrain)
    return create_dataset_per_label(label_ix, data, scene_nLabels)


def create_dataset_per_label_scene_test(label_ix):
    data, meta = arff.loadarff(scene_ftest)
    return create_dataset_per_label(label_ix, data, scene_nLabels)


def create_dataset_scene_train():
    data, meta = arff.loadarff(scene_ftrain)
    return create_dataset(data, scene_nLabels)


def create_dataset_scene_test():
    data, meta = arff.loadarff(scene_ftest)
    return create_dataset(data, scene_nLabels)


# The emotions dataset

def create_dataset_per_label_emotions_train(label_ix):
    data, meta = arff.loadarff(emotions_ftrain)
    return create_dataset_per_label(label_ix, data, emotions_nLabels)


def create_dataset_per_label_emotions_test(label_ix):
    data, meta = arff.loadarff(emotions_ftest)
    return create_dataset_per_label(label_ix, data, emotions_nLabels)


def create_dataset_emotions_train():
    data, meta = arff.loadarff(emotions_ftrain)
    return create_dataset(data, emotions_nLabels)


def create_dataset_emotions_test():
    data, meta = arff.loadarff(emotions_ftest)
    return create_dataset(data, emotions_nLabels)


# The bibtex dataset

def create_dataset_per_label_bibtex_train(label_ix):
    data_dict = pkl.load(open(bibtex_ftrain, 'rb'))
    return create_dataset_per_label(label_ix, data_dict['data'], bibtex_nLabels)


def create_dataset_per_label_bibtex_test(label_ix):
    data_dict = pkl.load(open(bibtex_ftest, 'rb'))
    return create_dataset_per_label(label_ix, data_dict['data'], bibtex_nLabels)


def create_dataset_bibtex_train():
    data_dict = pkl.load(open(bibtex_ftrain, 'rb'))
    return create_dataset(data_dict['data'], bibtex_nLabels)


def create_dataset_bibtex_test():
    data_dict = pkl.load(open(bibtex_ftest, 'rb'))
    return create_dataset(data_dict['data'], bibtex_nLabels)


# The bookmarks dataset

def create_dataset_per_label_bookmarks_train(label_ix):
    assert label_ix >= 0
    assert label_ix < bookmarks_nLabels
    X_train, Y_train = load_bookmarks_data(train_data=True)
    return X_train, Y_train[:, label_ix]


def create_dataset_per_label_bookmarks_test(label_ix):
    assert label_ix >= 0
    assert label_ix < bookmarks_nLabels
    X_test, Y_test = load_bookmarks_data(train_data=False)
    return X_test, Y_test[:, label_ix]


def create_dataset_bookmarks_train():
    return load_bookmarks_data(train_data=True)


def create_dataset_bookmarks_test():
    return load_bookmarks_data(train_data=False)


# The delicious dataset
# filtering out examples with all negative labels
# no examples with all positive labels (checked)

def create_dataset_per_label_delicious_train(label_ix):
    data_dict = pkl.load(open(delicious_ftrain, 'rb'))
    X_train, y_train = create_dataset_per_label(label_ix, data_dict['data'], delicious_nLabels)
    _, Y_train = create_dataset(data_dict['data'], delicious_nLabels)
    kpos = Y_train.sum(axis = 1)
    return X_train[kpos > 0, :], y_train[kpos > 0, :]


def create_dataset_per_label_delicious_test(label_ix):
    data_dict = pkl.load(open(delicious_ftest, 'rb'))
    X_test, y_test = create_dataset_per_label(label_ix, data_dict['data'], delicious_nLabels)
    _, Y_test = create_dataset(data_dict['data'], delicious_nLabels)
    kpos = Y_test.sum(axis = 1)
    return X_test[kpos > 0, :], y_test[kpos > 0, :]


def create_dataset_delicious_train():
    data_dict = pkl.load(open(delicious_ftrain, 'rb'))
    X_train, Y_train = create_dataset(data_dict['data'], delicious_nLabels)
    kpos = Y_train.sum(axis = 1)
    return X_train[kpos > 0, :], Y_train[kpos > 0, :]


def create_dataset_delicious_test():
    data_dict = pkl.load(open(delicious_ftest, 'rb'))
    X_test, Y_test = create_dataset(data_dict['data'], delicious_nLabels)
    kpos = Y_test.sum(axis = 1)
    return X_test[kpos > 0, :], Y_test[kpos > 0, :]


# The mediamill dataset

def create_dataset_per_label_mediamill_train(label_ix):
    data, meta = arff.loadarff(mm_ftrain)
    return create_dataset_per_label(label_ix, data, mm_nLabels)


def create_dataset_per_label_mediamill_test(label_ix):
    data, meta = arff.loadarff(mm_ftest)
    return create_dataset_per_label(label_ix, data, mm_nLabels)


def create_dataset_mediamill_train():
    data, meta = arff.loadarff(mm_ftrain)
    return create_dataset(data, mm_nLabels)


def create_dataset_mediamill_test():
    data, meta = arff.loadarff(mm_ftest)
    return create_dataset(data, mm_nLabels)


def create_dataset_mediamill_subset_train():
    """ Sample a subset from the original training set """
    data, meta = arff.loadarff(mm_ftrain)
    np.random.seed(SEED)
    N = data.shape[0]
    sample_ix = np.random.permutation(N)[:int(N*RATIO)]
    data_subset = data[sample_ix]
    return create_dataset(data_subset, mm_nLabels)


def create_dataset_mediamill_subset_test():
    """ Sample a subset from the original test set """
    data, meta = arff.loadarff(mm_ftest)
    np.random.seed(SEED)
    N = data.shape[0]
    sample_ix = np.random.permutation(N)[:int(N*RATIO)]
    data_subset = data[sample_ix]
    return create_dataset(data_subset, mm_nLabels)


# Common utilities
def create_dataset_per_label(label_ix, data, nLabels):
    """
        Create the labelled dataset for a given label index

        Input:
            - label_ix: label index, number in { 0, ..., # labels }
            - data: original data with features + labels

        Output:
            - (Feature, Label) pair (X, y)
              X comprises the features for each example
              y comprises the labels of the corresponding example
    """

    assert label_ix >= 0
    assert nLabels > 0
    assert type(nLabels) == int
    assert label_ix < nLabels

    N = len(data)
    D = len(data[0]) - nLabels
    magic = -nLabels

    X = np.zeros((N, D), dtype=np.float)
    y = np.zeros(N, dtype=np.int)

    for i in range(N):
        X[i, :] = list(data[i])[:magic]
        y[i] = list(data[i])[magic:][label_ix]

    return X, y


def create_dataset(data, nLabels):
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


def load_bookmarks_data(train_data=True):
    """
        Load (pre-split) bookmarks dataset, no examples with all positive/negative labels (checked)
    """
    features = np.zeros((0, bookmarks_nFeatures))
    labels = np.zeros((0, bookmarks_nLabels))

    if train_data is True:
        # load train data
        for k in range(1, 6):
            data_dict = torchfile.load(os.path.join(data_dir, 'bookmarks/bookmarks-train-%d.torch' % k))
            labels = np.concatenate([labels, data_dict[b'labels']], axis=0)
            features = np.concatenate([features, data_dict[b'data'][:, 0:bookmarks_nFeatures]], axis=0)

        # load dev data
        data_dict = torchfile.load(os.path.join(data_dir, 'bookmarks/bookmarks-dev.torch'))
        labels = np.concatenate([labels, data_dict[b'labels']], axis=0)
        features = np.concatenate([features, data_dict[b'data'][:, 0:bookmarks_nFeatures]], axis=0)
        
    else:
        # load test data
        for k in range(1, 4):
            data_dict = torchfile.load(os.path.join(data_dir, 'bookmarks/bookmarks-test-%d.torch' % k))
            labels = np.concatenate([labels, data_dict[b'labels']], axis=0)
            features = np.concatenate([features, data_dict[b'data'][:, 0:bookmarks_nFeatures]], axis=0)
    return features, labels
