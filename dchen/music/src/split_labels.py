import os
import sys
import gzip
import time
import numpy as np
import pickle as pkl
from scipy.sparse import issparse, vstack


if len(sys.argv) != 2:
    print('Usage: python', sys.argv[0], 'WORK_DIR')
    sys.exit(0)
else:
    work_dir = sys.argv[1]

data_dir = os.path.join(work_dir, 'data')
src_dir = os.path.join(work_dir, 'src')
sys.path.append(src_dir)

pkl_data_dir = os.path.join(data_dir, 'aotm-2011/setting2')
fytrain = os.path.join(pkl_data_dir, 'PU_test.pkl.gz')

Y_train = pkl.load(gzip.open(fytrain, 'rb'))

nLabels = Y_train.shape[1]

ranges = np.arange(0, nLabels, 500).tolist()
ranges.append(nLabels)

for i in range(1, len(ranges)):
    start = ranges[i-1]
    end = ranges[i]
    print('%d %d' % (start, end))

