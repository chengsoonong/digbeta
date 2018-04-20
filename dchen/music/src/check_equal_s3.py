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
fname = 'playlists_train_test_s3'
pl1 = pkl.load(gzip.open(os.path.join(dir1, '%s.pkl.gz' % fname), 'rb'))
pl2 = pkl.load(gzip.open(os.path.join(dir2, '%s.pkl.gz' % fname), 'rb'))
assert np.all(np.array(pl1) == np.array(pl2))


print('checking features ...')
x1 = pkl.load(gzip.open(os.path.join(dir1, 'X.pkl.gz'), 'rb'))
x2 = pkl.load(gzip.open(os.path.join(dir2, 'X.pkl.gz'), 'rb'))
assert np.all(np.isclose(x1, x2))

print('checking labels (sparse boolean matrices) ...')

for fname in ['Y_train', 'Y_test']:
    print('    checking %s ...' % fname)
    y1 = pkl.load(gzip.open(os.path.join(dir1, '%s.pkl.gz' % fname), 'rb'))
    y2 = pkl.load(gzip.open(os.path.join(dir2, '%s.pkl.gz' % fname), 'rb'))
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

for fname in ['song2pop', 'song2pop_train']:
    pops1 = pkl.load(gzip.open(os.path.join(dir1, '%s.pkl.gz' % fname), 'rb'))
    pops2 = pkl.load(gzip.open(os.path.join(dir2, '%s.pkl.gz' % fname), 'rb'))
    for sid in pops1:
        assert pops1[sid] == pops2[sid]


print('checking user playlists indices ...')

for fname in ['cliques_train', 'cliques_all']:
    clq1 = pkl.load(gzip.open(os.path.join(dir1, fname + '.pkl.gz'), 'rb'))
    clq2 = pkl.load(gzip.open(os.path.join(dir2, fname + '.pkl.gz'), 'rb'))
    assert len(clq1) == len(clq2)
    nclqs = len(clq1)
    for i in range(nclqs):
        assert np.all(clq1[i] == clq2[i])
