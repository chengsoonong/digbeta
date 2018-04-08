import os
import sys
import gzip
import numpy as np
import pickle as pkl
# from scipy.sparse import issparse, vstack


if len(sys.argv) != 3:
    print('Usage: python', sys.argv[0], 'WORK_DIR  DATASET')
    sys.exit(0)
else:
    work_dir = sys.argv[1]
    dataset = sys.argv[2]

data_dir = os.path.join(work_dir, 'data/%s' % dataset)
# src_dir = os.path.join(work_dir, 'src')
# sys.path.append(src_dir)

# setting 1
# pkl_data_dir = os.path.join(data_dir, 'setting1')
# fytrain = os.path.join(pkl_data_dir, 'Y_train_dev.pkl.gz')

# setting 2
pkl_data_dir = os.path.join(data_dir, 'setting2')
fytrain = os.path.join(pkl_data_dir, 'PU_test_1.pkl.gz')

Y_train = pkl.load(gzip.open(fytrain, 'rb'))

nLabels = Y_train.shape[1]

ranges = np.arange(0, nLabels, 1000).tolist()
ranges.append(nLabels)

for i in range(1, len(ranges)):
    start = ranges[i-1]
    end = ranges[i]
    print('%d %d' % (start, end))
