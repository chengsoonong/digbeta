import os
import sys
import gzip
import time
import numpy as np
import pickle as pkl
from scipy.sparse import hstack
from sklearn.metrics import roc_auc_score
from models import MTC


if len(sys.argv) != 7:
    print('Usage: python', sys.argv[0],
          'WORK_DIR  DATASET  C  P  N_SEED  TRAIN_DEV(Y/N)')
    sys.exit(0)
else:
    work_dir = sys.argv[1]
    dataset = sys.argv[2]
    C = float(sys.argv[3])
    p = float(sys.argv[4])
    n_seed = int(sys.argv[5])
    trndev = sys.argv[6]

assert trndev in ['Y', 'N']

data_dir = os.path.join(work_dir, 'data/%s/setting2' % dataset)

if trndev == 'N':
    fxtrain = os.path.join(data_dir, 'X_train_pop_%d.pkl.gz' % n_seed)
    fytrain = os.path.join(data_dir, 'Y_train.pkl.gz')
    fytrndev = os.path.join(data_dir, 'Y_trndev.pkl.gz')
    fydev = os.path.join(data_dir, 'PU_dev_%d.pkl.gz' % n_seed)
    fcliques = os.path.join(data_dir, 'cliques_trndev.pkl.gz')
    fprefix = 'pop-%g-%g-%g' % (n_seed, C, p)
else:
    fxtrain = os.path.join(data_dir, 'X_trndev_pop_%d.pkl.gz' % n_seed)
    fytrain = os.path.join(data_dir, 'Y_trndev.pkl.gz')
    fytrndev = os.path.join(data_dir, 'Y.pkl.gz')
    fydev = os.path.join(data_dir, 'PU_test_%d.pkl.gz' % n_seed)
    fcliques = os.path.join(data_dir, 'cliques_all.pkl.gz')
    fprefix = 'trndev-pop-%g-%g-%g' % (n_seed, C, p)

fmodel = os.path.join(data_dir, '%s.pkl.gz' % fprefix)
fnpy = os.path.join(data_dir, '%s.npy' % fprefix)

X_train = pkl.load(gzip.open(fxtrain, 'rb'))
Y_train = pkl.load(gzip.open(fytrain, 'rb'))
Y_train_dev = pkl.load(gzip.open(fytrndev, 'rb'))
PU_dev = pkl.load(gzip.open(fydev, 'rb'))
cliques = pkl.load(gzip.open(fcliques, 'rb'))

print('N_SEED: %g, C: %g, p: %g' % (n_seed, C, p))
print(X_train.shape, Y_train.shape)
print(time.strftime('%Y-%m-%d %H:%M:%S'))

if os.path.exists(fmodel):
    print('evaluating ...')
    clf = pkl.load(gzip.open(fmodel, 'rb'))  # for evaluation
else:
    print('training ...')
    Y = hstack([Y_train, PU_dev]).tocsc().astype(np.bool)
    clf = MTC(X_train, Y, C=C, p=p, user_playlist_indices=cliques, label_feature=False)
    clf.fit(njobs=1, verbose=2, fnpy=fnpy)

if clf.trained is True:
    # pkl.dump(clf, gzip.open(fmodel, 'wb'))
    Y_dev = Y_train_dev[:, -PU_dev.shape[1]:]
    offset = Y_train_dev.shape[1] - PU_dev.shape[1]
    rps = []
    aucs = []
    for j in range(Y_dev.shape[1]):
        y1 = Y_dev[:, j].toarray().reshape(-1)
        y2 = PU_dev[:, j].toarray().reshape(-1)
        indices = np.where(0 == y2)[0]
        y_true = y1[indices]
        npos = y_true.sum()
        assert npos > 0
        assert npos + y2.sum() == y1.sum()
        k = offset + j
        u = clf.pl2u[k]
        wk = clf.V[u, :] + clf.W[k, :] + clf.mu
        X = X_train
        y_pred = np.dot(X, wk)[indices]
        sortix = np.argsort(-y_pred)
        y_ = y_true[sortix]
        rps.append(np.mean(y_[:npos]))
        aucs.append(roc_auc_score(y_true, y_pred))
    clf.metric_score = (np.mean(aucs), np.mean(rps), len(rps), Y_dev.shape[1])
    pkl.dump(clf, gzip.open(fmodel, 'wb'))
    print('\n%.5f, %.5f, %d / %d' % clf.metric_score)
