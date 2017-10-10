import os
import numpy as np
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

mm_ftrain = os.path.join(data_dir, 'mediamill/mediamill-train.arff')
mm_ftest = os.path.join(data_dir, 'mediamill/mediamill-test.arff')
mm_nLabels = 101

SEED = 123456789
RATIO = 0.05


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
