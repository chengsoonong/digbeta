import os
import sys
import gzip
import time
import numpy as np
import pickle as pkl
from sklearn.metrics import roc_auc_score
from models import MTC


if len(sys.argv) != 6:
    print('Usage: python', sys.argv[0],
          'WORK_DIR  DATASET  C  P  TRAIN_DEV(Y/N)')
    sys.exit(0)
else:
    work_dir = sys.argv[1]
    dataset = sys.argv[2]
    C = float(sys.argv[3])
    p = float(sys.argv[4])
    trndev = sys.argv[5]

assert trndev in ['Y', 'N']

data_dir = os.path.join(work_dir, 'data/%s/setting1' % dataset)

if trndev == 'N':
    fxtrain = os.path.join(data_dir, 'X_train.pkl.gz')
    fytrain = os.path.join(data_dir, 'Y_train.pkl.gz')
    fxdev = os.path.join(data_dir, 'X_dev.pkl.gz')
    fydev = os.path.join(data_dir, 'Y_dev.pkl.gz')
    fcliques = os.path.join(data_dir, 'cliques_train.pkl.gz')
    fprefix = 'nsr-%g-%g' % (C, p)
else:
    assert trndev == 'Y'
    fxtrain = os.path.join(data_dir, 'X_train_dev.pkl.gz')
    fytrain = os.path.join(data_dir, 'Y_train_dev.pkl.gz')
    fxdev = os.path.join(data_dir, 'X_test.pkl.gz')
    fydev = os.path.join(data_dir, 'Y_test.pkl.gz')
    fcliques = os.path.join(data_dir, 'cliques_trndev.pkl.gz')
    fprefix = 'trndev-nsr-%g-%g' % (C, p)

fmodel = os.path.join(data_dir, '%s.pkl.gz' % fprefix)
fnpy = os.path.join(data_dir, '%s.npy' % fprefix)

X_train = pkl.load(gzip.open(fxtrain, 'rb'))
Y_train = pkl.load(gzip.open(fytrain, 'rb'))
X_dev = pkl.load(gzip.open(fxdev, 'rb'))
Y_dev = pkl.load(gzip.open(fydev, 'rb'))
cliques = pkl.load(gzip.open(fcliques, 'rb'))

print('C: %g, p: %g' % (C, p))
print(X_train.shape, Y_train.shape)
print(time.strftime('%Y-%m-%d %H:%M:%S'))

if os.path.exists(fmodel):
    print('evaluating ...')
    clf = pkl.load(gzip.open(fmodel, 'rb'))   # for evaluation
else:
    print('training ...')
    clf = MTC(X_train, Y_train, C=C, p=p, user_playlist_indices=cliques)
    clf.fit(njobs=1, verbose=2, fnpy=fnpy)

if clf.trained is True:
    pkl.dump(clf, gzip.open(fmodel, 'wb'))
    rps = []
    aucs = []
    # Y_pred = clf.predict(X_dev)
    # assert Y_dev.shape == Y_pred.shape
    for j in range(Y_dev.shape[1]):
        y_true = Y_dev[:, j].A.reshape(-1)
        npos = y_true.sum()
        if npos < 1:
            continue
        u = clf.pl2u[j]
        wj = clf.V[u, :] + clf.W[j, :] + clf.mu
        y_pred = np.dot(X_dev, wj).reshape(-1)
        sortix = np.argsort(-y_pred)
        y_ = y_true[sortix]
        rps.append(np.mean(y_[:npos]))
        aucs.append(roc_auc_score(y_true, y_pred))
    clf.metric_score = (np.mean(rps), np.mean(aucs), len(rps), Y_dev.shape[1])
    pkl.dump(clf, gzip.open(fmodel, 'wb'))
    print('\n%g, %g, %d / %d' % clf.metric_score)
