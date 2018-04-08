import os
import sys
import gzip
import time
import pickle as pkl
from scipy.sparse import issparse
from models import BinaryRelevance


if len(sys.argv) != 6:
    print('Usage: python', sys.argv[0], 'WORK_DIR  DATASET  NSEED  START  END')
    sys.exit(0)
else:
    work_dir = sys.argv[1]
    dataset = sys.argv[2]
    n_seed = int(sys.argv[3])
    start = int(sys.argv[4])
    end = int(sys.argv[5])

assert n_seed in [1, 2, 3, 4]

data_dir = os.path.join(work_dir, 'data/%s' % dataset)
# src_dir = os.path.join(work_dir, 'src')
# sys.path.append(src_dir)

pkl_data_dir = os.path.join(data_dir, 'setting2')
fxtrain = os.path.join(pkl_data_dir, 'X_trndev_%d.pkl.gz' % n_seed)
fytrain = os.path.join(pkl_data_dir, 'PU_test_%d.pkl.gz' % n_seed)

X_train = pkl.load(gzip.open(fxtrain, 'rb'))
Y_train = pkl.load(gzip.open(fytrain, 'rb'))

assert issparse(Y_train)
assert X_train.shape[0] == Y_train.shape[0]

Y = Y_train[:, start:end]

print(time.strftime('%Y-%m-%d %H:%M:%S'))

clf = BinaryRelevance(C=1, n_jobs=2)
clf.fit(X_train, Y)

print(time.strftime('%Y-%m-%d %H:%M:%S'))

fmodel = os.path.join(pkl_data_dir, 'br2-%s-%d-%d-%d.pkl.gz' % (dataset, n_seed, start, end))
pkl.dump(clf, gzip.open(fmodel, 'wb'))
