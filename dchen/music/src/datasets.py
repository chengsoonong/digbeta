import os
import numpy as np
from scipy.io import arff


data_dir = 'data'

yeast_ftrain = os.path.join(data_dir, 'yeast/yeast-train.arff')
yeast_ftest  = os.path.join(data_dir, 'yeast/yeast-test.arff')
yeast_nLabels = 14

scene_ftrain = os.path.join(data_dir, 'scene/scene-train.arff')
scene_ftest  = os.path.join(data_dir, 'scene/scene-test.arff')
scene_nLabels = 6

mm_ftrain = os.path.join(data_dir, 'mediamill/mediamill-train.arff')
mm_ftest  = os.path.join(data_dir, 'mediamill/mediamill-test.arff')
mm_nLabels = 101

SEED = 123456789
RATIO = 0.05

## The yeast dataset

def create_dataset_per_label_yeast_train(label_ix):
    yeast_train, yeast_meta_train = arff.loadarff(yeast_ftrain)
    return create_dataset_per_label(label_ix, yeast_train, yeast_nLabels)


def create_dataset_per_label_yeast_test(label_ix):
    yeast_test,  yeast_meta_test  = arff.loadarff(yeast_ftest)
    return create_dataset_per_label(label_ix, yeast_test, yeast_nLabels)


def create_dataset_yeast_train():
    yeast_train, yeast_meta_train = arff.loadarff(yeast_ftrain)
    return create_dataset(yeast_train, yeast_nLabels)


def create_dataset_yeast_test():
    yeast_test,  yeast_meta_test  = arff.loadarff(yeast_ftest)
    return create_dataset(yeast_test, yeast_nLabels)


## The scene dataset

def create_dataset_per_label_scene_train(label_ix):
    scene_train, scene_meta_train = arff.loadarff(scene_ftrain)
    return create_dataset_per_label(label_ix, scene_train, scene_nLabels)


def create_dataset_per_label_scene_test(label_ix):
    scene_test, scene_meta_test = arff.loadarff(scene_ftest)
    return create_dataset_per_label(label_ix, scene_test, scene_nLabels)


def create_dataset_scene_train():
    scene_train, scene_meta_train = arff.loadarff(scene_ftrain)
    return create_dataset(scene_train, scene_nLabels)


def create_dataset_scene_test():
    scene_test, scene_meta_test = arff.loadarff(scene_ftest)
    return create_dataset(scene_test, scene_nLabels)


# The mediamill dataset

def create_dataset_per_label_mediamill_train(label_ix):
    mm_train, mm_meta_train = arff.loadarff(mm_ftrain)
    return create_dataset_per_label(label_ix, mm_train, mm_nLabels)


def create_dataset_per_label_mediamill_test(label_ix):
    mm_test,  mm_meta_test  = arff.loadarff(mm_ftest)
    return create_dataset_per_label(label_ix, mm_test, mm_nLabels)


def create_dataset_mediamill_train():
    mm_train, mm_meta_train = arff.loadarff(mm_ftrain)
    return create_dataset(mm_train, mm_nLabels)


def create_dataset_mediamill_test():
    mm_test,  mm_meta_test  = arff.loadarff(mm_ftest)
    return create_dataset(mm_test, mm_nLabels)


def create_dataset_mediamill_subset_train():
    """ Sample a subset from the original training set """
    mm_train, mm_meta_train = arff.loadarff(mm_ftrain)
    np.random.seed(SEED)
    N = mm_train.shape[0]
    sample_ix = np.random.permutation(N)[:int(N*RATIO)]
    mm_train_subset = mm_train[sample_ix]
    return create_dataset(mm_train_subset, mm_nLabels)


def create_dataset_mediamill_subset_test():
    """ Sample a subset from the original test set """
    mm_test,  mm_meta_test  = arff.loadarff(mm_ftest)
    np.random.seed(SEED)
    N = mm_test.shape[0]
    sample_ix = np.random.permutation(N)[:int(N*RATIO)]
    mm_test_subset = mm_test[sample_ix]
    return create_dataset(mm_test_subset, mm_nLabels)


## Common utilities

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
    
    X = np.zeros((N, D), dtype = np.float)
    y = np.zeros(N, dtype = np.int)
       
    for i in range(N):
        X[i, :] = list(data[i])[:magic]
        y[i]    = list(data[i])[magic:][label_ix]

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

    X = np.zeros((N, D), dtype = np.float)
    Y = np.zeros((N, L), dtype = np.int)
       
    for i in range(N):
        X[i, :] = list(data[i])[:magic]
        Y[i, :] = list(data[i])[magic:]

    return X, Y
