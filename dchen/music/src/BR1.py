import os
import sys
import gzip
import time
import pickle as pkl
from scipy.sparse import issparse
from models import BinaryRelevance


if len(sys.argv) != 5:
    print('Usage: python', sys.argv[0], 'WORK_DIR  DATASET  START  END')
    sys.exit(0)
else:
    work_dir = sys.argv[1]
    dataset = sys.argv[2]
    start = int(sys.argv[3])
    end = int(sys.argv[4])

data_dir = os.path.join(work_dir, 'data/%s' % dataset)
# src_dir = os.path.join(work_dir, 'src')
# sys.path.append(src_dir)

pkl_data_dir = os.path.join(data_dir, 'setting1')
fxtrain = os.path.join(pkl_data_dir, 'X_train_dev.pkl.gz')
fytrain = os.path.join(pkl_data_dir, 'Y_train_dev.pkl.gz')

X_train = pkl.load(gzip.open(fxtrain, 'rb'))
Y_train = pkl.load(gzip.open(fytrain, 'rb'))

assert issparse(Y_train)
assert X_train.shape[0] == Y_train.shape[0]

Y = Y_train[:, start:end]

print(time.strftime('%Y-%m-%d %H:%M:%S'))

clf = BinaryRelevance(C=1, n_jobs=10)
clf.fit(X_train, Y)

print(time.strftime('%Y-%m-%d %H:%M:%S'))

fmodel = os.path.join(pkl_data_dir, 'br1-%s-%d-%d.pkl.gz' % (dataset, start, end))
pkl.dump(clf, gzip.open(fmodel, 'wb'))
