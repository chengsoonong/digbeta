import gzip
import os
import sys
import pickle as pkl
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix


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
    print('    checking %s ...' % fname)
    y1 = pkl.load(gzip.open(os.path.join(dir1, fname + '.pkl.gz'), 'rb'))
    y2 = pkl.load(gzip.open(os.path.join(dir2, fname + '.pkl.gz'), 'rb'))
    assert type(y1) == type(y2)
    if type(y1) == csr_matrix:
        y1 = y1.tocsc()
        y2 = y2.tocsc()
    elif type(y1) == csc_matrix:
        y1 = y1.tocsr()
        y2 = y2.tocsr()
    else:
        assert False, 'NOT CSR or CSC format'
    assert np.all(np.equal(y1.indices, y2.indices))
    # NOTE: the csr sparse representation of the same dense matrix can have different indices,
    # so transform them to another representation may result in the same indices.


print('checking user playlists indices ...')

for fname in ['cliques_train', 'cliques_trndev']:
    clq1 = pkl.load(gzip.open(os.path.join(dir1, fname + '.pkl.gz'), 'rb'))
    clq2 = pkl.load(gzip.open(os.path.join(dir2, fname + '.pkl.gz'), 'rb'))
    assert len(clq1) == len(clq2)
    for i in range(len(clq1)):
        assert np.all(clq1[i] == clq2[i])

print('done.')
