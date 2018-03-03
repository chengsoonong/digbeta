import os
import sys
import gzip
import time
import numpy as np
import pickle as pkl
from sklearn.metrics import roc_auc_score
from models import PCMLC

if len(sys.argv) != 10:
    print('Usage: python', sys.argv[0], 'WORK_DIR  DATASET  C1  C2  C3  P  N_EPOCH  LOSS_TYPE(example/label/both)  MT_REG(Y/N)')
    sys.exit(0)
else:
    work_dir = sys.argv[1]
    dataset = sys.argv[2]
    C1 = float(sys.argv[3])
    C2 = float(sys.argv[4])
    C3 = float(sys.argv[5])
    p = float(sys.argv[6])
    n_epochs = int(sys.argv[7])
    loss = sys.argv[8]
    multitask = sys.argv[9]

assert loss in ['example', 'label', 'both']
assert multitask in ['Y', 'N']

data_dir = os.path.join(work_dir, 'data/%s/setting1' % dataset)
# src_dir = os.path.join(work_dir, 'src')
# sys.path.append(src_dir)

fxtrain = os.path.join(data_dir, 'X_train_dev.pkl.gz')
fytrain = os.path.join(data_dir, 'Y_train_dev.pkl.gz')
fxdev = os.path.join(data_dir, 'X_test.pkl.gz')
fydev = os.path.join(data_dir, 'Y_test.pkl.gz')
fcliques = os.path.join(data_dir, 'cliques_all.pkl.gz')

X_train = pkl.load(gzip.open(fxtrain, 'rb'))
Y_train = pkl.load(gzip.open(fytrain, 'rb'))
X_dev = pkl.load(gzip.open(fxdev, 'rb'))
Y_dev = pkl.load(gzip.open(fydev, 'rb'))

if multitask == 'Y':
    cliques = pkl.load(gzip.open(fcliques, 'rb'))
else:
    cliques = None

print('C: %g, %g, %g, p: %g' % (C1, C2, C3, p))

print(time.strftime('%Y-%m-%d %H:%M:%S'))
clf = PCMLC(C1=C1, C2=C2, C3=C3, p=p, loss_type=loss)
clf.fit_minibatch_mlr(X_train, Y_train, user_playlist_indices=cliques, batch_size=512, n_epochs=n_epochs, verbose=1)

if clf.trained is True:
    W = clf.W
    b = clf.b
    aucs = []
    for j in range(Y_dev.shape[1]):
        # if (j+1) % 100 == 0:
        #    sys.stdout.write('\r%d / %d' % (j+1, Y_dev.shape[1]))
        #    sys.stdout.flush()
        y_true = Y_dev[:, j].toarray().reshape(-1)
        if y_true.sum() < 1: continue
        wj = W[j, :].reshape(-1)
        y_pred = np.dot(X_dev, wj) + b
        aucs.append(roc_auc_score(y_true, y_pred))
    clf.dev_auc = (np.mean(aucs), len(aucs), Y_dev.shape[1])
    fmodel = os.path.join(data_dir, 'trndev-mlr-%s-%s-%g-%g-%g-%g.pkl.gz' % (loss, multitask, C1, C2, C3, p))
    pkl.dump(clf, gzip.open(fmodel, 'wb'))
    print('\n%.5f, %d / %d' % clf.dev_auc)

