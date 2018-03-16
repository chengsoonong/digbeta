import os
import sys
import gzip
import time
import numpy as np
import pickle as pkl
from models import MLC


if len(sys.argv) != 9:
    print('Usage: python', sys.argv[0],
          'WORK_DIR  DATASET  C1  C3  P  BATCH_SIZE  MT_REG(Y/N)  TRAIN_DEV(Y/N)')
    sys.exit(0)
else:
    work_dir = sys.argv[1]
    dataset = sys.argv[2]
    C1 = float(sys.argv[3])
    C3 = float(sys.argv[4])
    p = float(sys.argv[5])
    bs = int(sys.argv[6])
    multitask = sys.argv[7]
    trndev = sys.argv[8]

assert multitask in ['Y', 'N']
assert trndev in ['Y', 'N']

data_dir = os.path.join(work_dir, 'data/%s/setting1' % dataset)

if trndev == 'N':
    fxtrain = os.path.join(data_dir, 'X_train.pkl.gz')
    fytrain = os.path.join(data_dir, 'Y_train.pkl.gz')
    fxdev = os.path.join(data_dir, 'X_dev.pkl.gz')
    fydev = os.path.join(data_dir, 'Y_dev.pkl.gz')
    fprefix = 'mlr-%s-%g-%g-%g' % (multitask, C1, C3, p)
else:
    assert trndev == 'Y'
    fxtrain = os.path.join(data_dir, 'X_train_dev.pkl.gz')
    fytrain = os.path.join(data_dir, 'Y_train_dev.pkl.gz')
    fxdev = os.path.join(data_dir, 'X_test.pkl.gz')
    fydev = os.path.join(data_dir, 'Y_test.pkl.gz')
    fprefix = 'trndev-mlr-%s-%g-%g-%g' % (multitask, C1, C3, p)

fcliques = os.path.join(data_dir, 'cliques_all.pkl.gz')
fmodel = os.path.join(data_dir, '%s.pkl.gz' % fprefix)
fnpy = os.path.join(data_dir, '%s.npy' % fprefix)

X_train = pkl.load(gzip.open(fxtrain, 'rb'))
Y_train = pkl.load(gzip.open(fytrain, 'rb'))
X_dev = pkl.load(gzip.open(fxdev, 'rb'))
Y_dev = pkl.load(gzip.open(fydev, 'rb'))

if multitask == 'Y':
    cliques = pkl.load(gzip.open(fcliques, 'rb'))
else:
    cliques = None

print('C: %g, %g, p: %g' % (C1, C3, p))
print(X_train.shape, Y_train.shape)
print(time.strftime('%Y-%m-%d %H:%M:%S'))

if os.path.exists(fmodel):
    print('evaluating ...')
    clf = pkl.load(gzip.open(fmodel, 'rb'))   # for evaluation
else:
    print('training ...')
    clf = MLC(C1=C1, C3=C3, p=p)
    clf.fit(X_train, Y_train, user_playlist_indices=cliques, batch_size=bs, njobs=3, verbose=2, fnpy=fnpy)

if clf.trained is True:
    W = clf.W
    b = clf.b
    rps = []
    for j in range(Y_dev.shape[1]):
        y_true = Y_dev[:, j].toarray().reshape(-1)
        npos = y_true.sum()
        if npos < 1:
            continue
        wj = W[j, :].reshape(-1)
        y_pred = np.dot(X_dev, wj) + b[0, j]
        sortix = np.argsort(-y_pred)
        y_ = y_true[sortix]
        rps.append(np.mean(y_[:npos]))
    clf.metric_score = (np.mean(rps), len(rps), Y_dev.shape[1])
    pkl.dump(clf, gzip.open(fmodel, 'wb'))
    print('\n%.5f, %d / %d' % clf.metric_score)
