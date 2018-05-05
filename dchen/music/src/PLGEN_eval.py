import os
import sys
import gzip
import pickle as pkl
import numpy as np
from scipy.sparse import issparse
# from tools import calc_RPrecision_HitRate
from tools import calc_metrics

TOPs = [5, 10, 20, 30, 50, 100, 200, 300, 500, 1000]

if len(sys.argv) != 6:
    print('Usage:', sys.argv[0], 'WORK_DIR  DATASET  TASK(3/4)  MODEL_FILE.pkl.gz  OUTPUT_PKL')
    sys.exit(0)

work_dir = sys.argv[1]
dataset = sys.argv[2]
task = int(sys.argv[3])
fmodel = sys.argv[4]
fout = sys.argv[5]

assert fmodel.endswith('pkl.gz')
assert os.path.exists(fmodel)
assert task in [3, 4]

data_dir = os.path.join(work_dir, 'data/%s/setting%d' % (dataset, task))
fperf = os.path.join(data_dir, fout)
X = pkl.load(gzip.open(os.path.join(data_dir, 'X.pkl.gz'), 'rb'))
X = np.hstack([np.ones((X.shape[0], 1)), X])
Y_train = pkl.load(gzip.open(os.path.join(data_dir, 'Y_train.pkl.gz'), 'rb'))
Y_test = pkl.load(gzip.open(os.path.join(data_dir, 'Y_test.pkl.gz'), 'rb'))
cliques_train = pkl.load(gzip.open(os.path.join(data_dir, 'cliques_train.pkl.gz'), 'rb'))

if task == 3:
    cliques_all = pkl.load(gzip.open(os.path.join(data_dir, 'cliques_all.pkl.gz'), 'rb'))

assert issparse(Y_test)
clf = pkl.load(gzip.open(fmodel, 'rb'))
assert clf.trained is True

if task == 3:
    pl2u = np.zeros(Y_train.shape[1] + Y_test.shape[1], dtype=np.int)
    U = len(cliques_train)
    assert len(cliques_all) == U
    for u in range(U):
        clq = cliques_all[u]
        pl2u[clq] = u
    assert np.all(clf.pl2u == pl2u[:Y_train.shape[1]])

rps = []
hitrates = {top: [] for top in TOPs}
aucs = []
offset = Y_train.shape[1]
for j in range(Y_test.shape[1]):
    if (j+1) % 100 == 0:
        sys.stdout.write('\r%d / %d' % (j+1, Y_test.shape[1]))
        sys.stdout.flush()
    y_true = Y_test[:, j].A.reshape(-1)
    assert y_true.sum() > 0
    if task == 3:
        u = pl2u[j + offset]
        wj = clf.V[u, :] + clf.mu
        y_pred = np.dot(X, wj).reshape(-1)
    else:
        y_pred = np.dot(X, clf.mu).reshape(-1)

    # rp, hr_dict = calc_RPrecision_HitRate(y_true, y_pred, tops=TOPs)
    rp, hr_dict, auc = calc_metrics(y_true, y_pred, tops=TOPs)
    rps.append(rp)
    for top in TOPs:
        hitrates[top].append(hr_dict[top])
    aucs.append(auc)

print('\n%d, %d' % (len(rps), Y_test.shape[1]))
perf = {dataset: {'Test': {'R-Precision': np.mean(rps), 'Hit-Rate': {top: np.mean(hitrates[top]) for top in TOPs},
                           'AUC': np.mean(aucs)}}}
pkl.dump(perf, open(fperf, 'wb'))
print(perf)
