import os, sys, gzip
import pickle as pkl
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.sparse import lil_matrix, issparse
from models import PCMLC, obj_pclassification

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage:', sys.argv[0], 'WORK_DIR  PKL_FILE')
        sys.exit(0)
work_dir = sys.argv[1]
data_dir = os.path.join(work_dir, 'data/aotm-2011/setting2')
fpkl = os.path.join(data_dir, sys.argv[2])
X_train = pkl.load(gzip.open(os.path.join(data_dir, 'X_train.pkl.gz'), 'rb'))
Y_train = pkl.load(gzip.open(os.path.join(data_dir, 'Y_train.pkl.gz'), 'rb'))
Y_train_dev = pkl.load(gzip.open(os.path.join(data_dir, 'Y_train_dev.pkl.gz'), 'rb'))
PU_dev = pkl.load(gzip.open(os.path.join(data_dir, 'PU_dev.pkl.gz'), 'rb'))
#PU_test = pkl.load(gzip.open(os.path.join(data_dir, 'PU_test.pkl.gz'), 'rb'))
#cliques = pkl.load(gzip.open(os.path.join(data_dir, 'cliques_train_dev.pkl.gz'), 'rb'))

clf = pkl.load(gzip.open(fpkl, 'rb'))

Y_dev = Y_train_dev[:, -PU_dev.shape[1]:]
offset = Y_train_dev.shape[1] - PU_dev.shape[1]
W = clf.W
b = clf.b
print(clf)
aucs = []
for j in range(Y_dev.shape[1]):
    if (j+1) % 100 == 0:
        sys.stdout.write('\r%d / %d' % (j+1, Y_dev.shape[1]))
        sys.stdout.flush()
    y1 = Y_dev[:, j].toarray().reshape(-1)
    y2 = PU_dev[:, j].toarray().reshape(-1)
    indices = np.where(0 == y2)[0]
    #print(indices); break
    y_true = y1[indices]
    assert y_true.sum() + PU_dev[:, j].sum() == Y_dev[:, j].sum()
    wj = W[j + offset, :].reshape(-1)
    y_pred = (np.dot(X_train, wj) + b)[indices]
    aucs.append(roc_auc_score(y_true, y_pred))
print('\n%.5f, %d / %d' % (np.mean(aucs), len(aucs), Y_dev.shape[1]))

