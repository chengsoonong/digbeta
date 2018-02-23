import os, sys, gzip
import pickle as pkl
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.sparse import issparse
from models import BinaryRelevance

TOPs = [5, 10, 20, 30, 50, 100, 200, 300, 500, 1000]

def calc_hitrate(y_true, y_pred, tops=[100]):
    """
        Compute hitrate at top-N.
    """
    assert y_true.ndim == y_pred.ndim == 1
    assert len(y_true) == len(y_pred)
    assert type(tops) == list
    hitrates = dict()
    sortix = np.argsort(-y_pred)
    npos = y_true.sum()
    assert npos > 0
    for top in tops:
        assert 0 < top <= len(y_true)
        hitrates[top] = np.sum(y_true[sortix[:top]]) / npos
    return hitrates

if len(sys.argv) != 3:
    print('Usage:', sys.argv[0], 'WORK_DIR  DATASET')
    sys.exit(0)

work_dir = sys.argv[1]
dataset = sys.argv[2]
data_dir = os.path.join(work_dir, 'data/%s/setting1' % dataset)
fsplit = os.path.join(data_dir, 'br1/br1.split')
fperf = os.path.join(data_dir, 'perf-br1.pkl')
X_test  = pkl.load(gzip.open(os.path.join(data_dir, 'X_test.pkl.gz'), 'rb'))
Y_test  = pkl.load(gzip.open(os.path.join(data_dir, 'Y_test.pkl.gz'), 'rb'))

aucs = []
hitrates = {top: [] for top in TOPs}
with open(fsplit, 'r') as fd:
    for line in fd:
        start, end = line.strip().split(' ')
        print(start, end)
        fname = os.path.join(data_dir, 'br1/br1-%s-%s-%s.pkl.gz' % (dataset, start, end))
        br = pkl.load(gzip.open(fname, 'rb'))
        preds = br.predict(X_test)
        for j in range(int(start), int(end)):
            y_true = Y_test[:, j]
            if issparse(y_true):
                y_true = y_true.toarray().reshape(-1)
            else:
                y_true = y_true.reshape(-1)
            npos = y_true.sum()
            if npos < 1: continue
            y_pred = preds[:, j-int(start)].reshape(-1)
            auc = roc_auc_score(y_true, y_pred)
            aucs.append(auc)
            hr_dict = calc_hitrate(y_true, y_pred, tops=TOPs)
            for top in TOPs:
                hitrates[top].append(hr_dict[top])

br1_perf = {dataset: {'Test': {'AUC': np.mean(aucs), 'HitRate': {top: np.mean(hitrates[top]) for top in TOPs}}}}
pkl.dump(br1_perf, open(fperf, 'wb'))
print(br1_perf)
