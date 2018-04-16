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
    print('Usage:', sys.argv[0], 'WORK_DIR  DATASET  N_SEED  MODEL_FILE.pkl.gz  OUTPUT_PKL')
    sys.exit(0)

work_dir = sys.argv[1]
dataset = sys.argv[2]
n_seed = int(sys.argv[3])
fmodel = sys.argv[4]
fout = sys.argv[5]

assert fmodel.endswith('pkl.gz')
assert os.path.exists(fmodel)

data_dir = os.path.join(work_dir, 'data/%s/setting2' % dataset)
# fperf = os.path.join(data_dir, 'perf-pla-%d.pkl' % n_seed)
fperf = os.path.join(data_dir, fout)
X = pkl.load(gzip.open(os.path.join(data_dir, 'X_trndev_%d.pkl.gz' % n_seed), 'rb'))
Y = pkl.load(gzip.open(os.path.join(data_dir, 'Y.pkl.gz'), 'rb'))
PU_test = pkl.load(gzip.open(os.path.join(data_dir, 'PU_test_%d.pkl.gz' % n_seed), 'rb'))
Y_test = Y[:, -PU_test.shape[1]:]
offset = Y.shape[1] - PU_test.shape[1]

assert issparse(PU_test)
assert issparse(Y_test)
clf = pkl.load(gzip.open(fmodel, 'rb'))
assert clf.trained is True

rps = []
hitrates = {top: [] for top in TOPs}
aucs = []
for j in range(Y_test.shape[1]):
    if (j+1) % 100 == 0:
        sys.stdout.write('\r%d / %d' % (j+1, Y_test.shape[1]))
        sys.stdout.flush()
    y1 = Y_test[:, j].toarray().reshape(-1)
    y2 = PU_test[:, j].toarray().reshape(-1)
    indices = np.where(0 == y2)[0]
    y_true = y1[indices]
    npos = y_true.sum()
    assert npos > 0
    assert npos + y2.sum() == y1.sum()
    k = offset + j
    u = clf.pl2u[k]
    wk = clf.V[u, :] + clf.W[k, :] + clf.mu
    # X = X if clf.UF is None else np.concatenate([X, np.delete(clf.UF, u, axis=1)], axis=1)
    y_pred = np.dot(X, wk)[indices]
    # rp, hr_dict = calc_RPrecision_HitRate(y_true, y_pred, tops=TOPs)
    rp, hr_dict, auc = calc_metrics(y_true, y_pred, tops=TOPs)
    rps.append(rp)
    for top in TOPs:
        hitrates[top].append(hr_dict[top])
    aucs.append(auc)

print('\n%d, %d' % (len(rps), Y_test.shape[1]))
pla_perf = {dataset: {'Test': {'R-Precision': np.mean(rps), 'Hit-Rate': {top: np.mean(hitrates[top]) for top in TOPs},
                               'AUC': np.mean(aucs)}}}
pkl.dump(pla_perf, open(fperf, 'wb'))
print(pla_perf)
