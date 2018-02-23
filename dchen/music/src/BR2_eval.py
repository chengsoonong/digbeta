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
data_dir = os.path.join(work_dir, 'data/%s/setting2' % dataset)
fsplit = os.path.join(data_dir, 'br2/br2.split')
fperf = os.path.join(data_dir, 'perf-br2.pkl')
X = pkl.load(gzip.open(os.path.join(data_dir, 'X_train.pkl.gz'), 'rb'))
Y = pkl.load(gzip.open(os.path.join(data_dir, 'Y.pkl.gz'), 'rb'))
PU_test = pkl.load(gzip.open(os.path.join(data_dir, 'PU_test.pkl.gz'), 'rb'))
Y_test = Y[:, -PU_test.shape[1]:]

aucs = []
hitrates = {top: [] for top in TOPs}
with open(fsplit, 'r') as fd:
    for line in fd:
        start, end = line.strip().split(' ')
        print(start, end)
        fname = os.path.join(data_dir, 'br2/br2-%s-%s-%s.pkl.gz' % (dataset, start, end))
        br = pkl.load(gzip.open(fname, 'rb'))
        preds = br.predict(X)
        for j in range(int(start), int(end)):
            yj = PU_test[:, j]
            if issparse(yj):
                yj = yj.toarray().reshape(-1)
            else:
                yj = yj.reshape(-1)
            indices = np.where(0 == yj)[0]
            y_true = Y_test[:, j]
            if issparse(y_true):
                y_true = y_true.toarray().reshape(-1)
            else:
                y_true = y_true.reshape(-1)
            y_true = y_true[indices]
            assert y_true.sum() > 0
            y_pred = preds[indices, j-int(start)].reshape(-1)
            auc = roc_auc_score(y_true, y_pred)
            aucs.append(auc)
            hr_dict = calc_hitrate(y_true, y_pred, tops=TOPs)
            for top in TOPs:
                hitrates[top].append(hr_dict[top])

br2_perf = {dataset: {'Test': {'AUC': np.mean(aucs), 'HitRate': {top: np.mean(hitrates[top]) for top in TOPs}}}}
pkl.dump(br2_perf, open(fperf, 'wb'))
print(br2_perf)

