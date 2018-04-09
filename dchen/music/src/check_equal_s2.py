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

songs1 = pkl.load(gzip.open(os.path.join(dir1, 'all_songs.pkl.gz'), 'rb'))
songs2 = pkl.load(gzip.open(os.path.join(dir2, 'all_songs.pkl.gz'), 'rb'))
assert np.all(np.array(songs1) == np.array(songs2))


print('checking playlists ...')
fname = 'playlists_train_dev_test_s2'
for k in [1, 2, 3, 4]:
    pl1 = pkl.load(gzip.open(os.path.join(dir1, '%s_%d.pkl.gz' % (fname, k)), 'rb'))
    pl2 = pkl.load(gzip.open(os.path.join(dir2, '%s_%d.pkl.gz' % (fname, k)), 'rb'))
    assert np.all(np.array(pl1) == np.array(pl2))


print('checking features ...')

for fname in ['X_train', 'X_trndev']:
    for k in [1, 2, 3, 4]:
        x1 = pkl.load(gzip.open(os.path.join(dir1, '%s_%d.pkl.gz' % (fname, k)), 'rb'))
        x2 = pkl.load(gzip.open(os.path.join(dir2, '%s_%d.pkl.gz' % (fname, k)), 'rb'))
        assert np.all(np.isclose(x1, x2))

print('checking labels (sparse boolean matrices) ...')

for fname in ['Y', 'Y_train', 'Y_trndev', 'PU_dev', 'PU_test']:
    print('    checking %s ...' % fname)
    for k in [1, 2, 3, 4]:
        y1 = pkl.load(gzip.open(os.path.join(dir1, '%s_%d.pkl.gz' % (fname, k)), 'rb'))
        y2 = pkl.load(gzip.open(os.path.join(dir2, '%s_%d.pkl.gz' % (fname, k)), 'rb'))
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


print('checking song popularities ...')

pops1 = pkl.load(gzip.open(os.path.join(dir1, 'song2pop.pkl.gz'), 'rb'))
pops2 = pkl.load(gzip.open(os.path.join(dir2, 'song2pop.pkl.gz'), 'rb'))
for sid in pops1:
    assert pops1[sid] == pops2[sid]
