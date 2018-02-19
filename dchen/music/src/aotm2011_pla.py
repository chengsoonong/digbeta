import os
import sys
import gzip
import time
import numpy as np
import pickle as pkl
from models import PCMLC

if len(sys.argv) != 9:
    print('Usage: python', sys.argv[0], 'WORK_DIR  C1  C2  C3  P  N_EPOCH  LOSS_TYPE(example/label/both)  MT_REG(Y/N)')
    sys.exit(0)
else:
    work_dir = sys.argv[1]
    C1 = float(sys.argv[2])
    C2 = float(sys.argv[3])
    C3 = float(sys.argv[4])
    p = float(sys.argv[5])
    n_epochs = int(sys.argv[6])
    loss = sys.argv[7]
    multitask = sys.argv[8]

assert loss in ['example', 'label', 'both']
assert multitask in ['Y', 'N']

data_dir = os.path.join(work_dir, 'data')
# src_dir = os.path.join(work_dir, 'src')
# sys.path.append(src_dir)

pkl_data_dir = os.path.join(data_dir, 'aotm-2011/setting2')
fxtrain = os.path.join(pkl_data_dir, 'X_train.pkl.gz')
fytrain = os.path.join(pkl_data_dir, 'Y_train.pkl.gz')
fytrndev = os.path.join(pkl_data_dir, 'Y_train_dev.pkl.gz')
fydev = os.path.join(pkl_data_dir, 'PU_dev.pkl.gz')
fcliques = os.path.join(pkl_data_dir, 'cliques_train_dev.pkl.gz')

X_train = pkl.load(gzip.open(fxtrain, 'rb'))
Y_train = pkl.load(gzip.open(fytrain, 'rb'))
Y_train_dev = pkl.load(gzip.open(fytrndev, 'rb'))
PU_dev = pkl.load(gzip.open(fydev, 'rb'))
if multitask == 'Y':
    cliques = pkl.load(gzip.open(fcliques, 'rb'))
else:
    cliques = None

print('C: %g, %g, %g, p: %g' % (C1, C2, C3, p))

print(time.strftime('%Y-%m-%d %H:%M:%S'))
clf = PCMLC(C1=C1, C2=C2, C3=C3, p=p,  loss_type=loss)
clf.fit_minibatch_pla(X_train, Y_train, PUMat=PU_dev, user_playlist_indices=cliques, batch_size=512, n_epochs=n_epochs, verbose=1)

if clf.trained is True:
    fmodel = os.path.join(pkl_data_dir, 'pla-%s-%g-%g-%g-%g.pkl.gz' % (loss, C1, C2, C3, p))
    pkl.dump(clf, gzip.open(fmodel, 'wb'))
    Y_dev = Y_train_dev[:, -PU_dev.shape[1]:]
    offset = Y_train_dev.shape[1] - PU_dev.shape[1]
    W = clf.W
    b = clf.b
    aucs = []
    for j in range(Y_dev.shape[1]):
        y1 = Y_dev[:, j].toarray().reshape(-1)
        y2 = PU_dev[:, j].toarray().reshape(-1)
        indices = np.where(0 == y2)[0]
        y_true = y1[indices]
        assert y_true.sum() + PU_dev[:, j].sum() == Y_dev[:, j].sum()
        wj = W[j + offset, :].reshape(-1)
        y_pred = (np.dot(X_train, wj) + b)[indices]
        aucs.append(roc_auc_score(y_true, y_pred))
    print('\n%.5f, %d / %d' % (np.mean(aucs), len(aucs), Y_dev.shape[1]))

