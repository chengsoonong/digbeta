import os
import sys
import gzip
import pickle as pkl
import numpy as np
from scipy.sparse import issparse
# from tools import calc_RPrecision_HitRate
from tools import calc_metrics

TOPs = [5, 10, 20, 30, 50, 100, 200, 300, 500, 1000]

if len(sys.argv) != 4:
    print('Usage:', sys.argv[0], 'WORK_DIR  DATASET  N_SEED')
    sys.exit(0)

work_dir = sys.argv[1]
dataset = sys.argv[2]
n_seed = int(sys.argv[3])
data_dir = os.path.join(work_dir, 'data/%s/setting2' % dataset)
fsplit = os.path.join(data_dir, 'br2/br2.%s.split' % dataset)
fperf = os.path.join(data_dir, 'perf-br2-%d.pkl' % n_seed)
X = pkl.load(gzip.open(os.path.join(data_dir, 'X_trndev_%d.pkl.gz' % n_seed), 'rb'))
Y = pkl.load(gzip.open(os.path.join(data_dir, 'Y.pkl.gz'), 'rb'))
PU_test = pkl.load(gzip.open(os.path.join(data_dir, 'PU_test_%d.pkl.gz' % n_seed), 'rb'))
Y_test = Y[:, -PU_test.shape[1]:]

rps = []
hitrates = {top: [] for top in TOPs}
aucs = []
with open(fsplit, 'r') as fd:
    for line in fd:
        start, end = line.strip().split(' ')
        print(start, end)
        fname = os.path.join(data_dir, 'br2/br2-%s-%s-%s-%s.pkl.gz' % (dataset, n_seed, start, end))
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
            # rp, hr_dict = calc_RPrecision_HitRate(y_true, y_pred, tops=TOPs)
            rp, hr_dict, auc = calc_metrics(y_true, y_pred, tops=TOPs)
            rps.append(rp)
            for top in TOPs:
                hitrates[top].append(hr_dict[top])
            aucs.append(auc)

print(len(rps), Y_test.shape[1])
br2_perf = {dataset: {'Test': {'R-Precision': np.mean(rps), 'Hit-Rate': {top: np.mean(hitrates[top]) for top in TOPs},
                               'AUC': np.mean(aucs)}}}
pkl.dump(br2_perf, open(fperf, 'wb'))
print(br2_perf)
