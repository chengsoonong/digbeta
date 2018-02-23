import gzip
import os, sys
import pickle as pkl
import numpy as np
from scipy.sparse import csr_matrix


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage:', sys.argv[0], 'DIR1 DIR2')
        sys.exit(0)

dir1 = sys.argv[1]
dir2 = sys.argv[2]

print('checking songs ...')

songs1 = pkl.load(gzip.open(os.path.join(dir1, 'songs_train_dev_test_s1.pkl.gz'), 'rb'))
songs2 = pkl.load(gzip.open(os.path.join(dir2, 'songs_train_dev_test_s1.pkl.gz'), 'rb'))
for key in songs1:
    assert np.all(np.array(songs1[key]) == np.array(songs2[key]))

print('checking playlists ...')

pl1 = pkl.load(gzip.open(os.path.join(dir1, 'playlists_s1.pkl.gz'), 'rb'))
pl2 = pkl.load(gzip.open(os.path.join(dir2, 'playlists_s1.pkl.gz'), 'rb'))
assert np.all(np.array(pl1) == np.array(pl2))


print('checking features ...')

for fname in ['X_train', 'X_dev', 'X_test', 'X_train_dev']:
    x1 = pkl.load(gzip.open(os.path.join(dir1, fname + '.pkl.gz'), 'rb'))
    x2 = pkl.load(gzip.open(os.path.join(dir2, fname + '.pkl.gz'), 'rb'))
    assert np.all(np.isclose(x1, x2))

print('checking labels (sparse boolean matrices) ...')

for fname in ['Y_train', 'Y_dev', 'Y_test', 'Y_train_dev']:
    y1 = pkl.load(gzip.open(os.path.join(dir1, fname + '.pkl.gz'), 'rb'))
    y2 = pkl.load(gzip.open(os.path.join(dir2, fname + '.pkl.gz'), 'rb'))
    assert np.all(np.equal(y1.indices, y2.indices))

