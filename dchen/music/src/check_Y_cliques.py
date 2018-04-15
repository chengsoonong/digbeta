import gzip
import os
import sys
import pickle as pkl
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage:', sys.argv[0], 'DIR')
        sys.exit(0)

dir1 = sys.argv[1]

print('checking labels (sparse boolean matrices) ...')

for fname in ['Y', 'Y_train', 'Y_trndev']:
    y1 = pkl.load(gzip.open(os.path.join(dir1, '%s_1.pkl.gz' % fname), 'rb'))
    y2 = pkl.load(gzip.open(os.path.join(dir1, '%s_2.pkl.gz' % fname), 'rb'))
    y3 = pkl.load(gzip.open(os.path.join(dir1, '%s_3.pkl.gz' % fname), 'rb'))
    y4 = pkl.load(gzip.open(os.path.join(dir1, '%s_4.pkl.gz' % fname), 'rb'))
    assert type(y1) == type(y2) == type(y3) == type(y4)
    ytype = type(y1)
    if ytype == csr_matrix:
        y1 = y1.tocsc()
        y2 = y2.tocsc()
        y3 = y3.tocsc()
        y4 = y4.tocsc()
    elif ytype == csc_matrix:
        y1 = y1.tocsr()
        y2 = y2.tocsr()
        y3 = y3.tocsr()
        y4 = y4.tocsr()
    else:
        assert False, 'NOT CSR or CSC format'
    assert np.all(np.equal(y1.indices, y2.indices))
    assert np.all(np.equal(y2.indices, y3.indices))
    assert np.all(np.equal(y3.indices, y4.indices))
    # NOTE: the csr sparse representation of the same dense matrix can have different indices,
    # so transform them to another representation may result in the same indices.


print('checking user playlists indices ...')

for fname in ['cliques_trndev', 'cliques_all']:
    clq1 = pkl.load(gzip.open(os.path.join(dir1, fname + '_1.pkl.gz'), 'rb'))
    clq2 = pkl.load(gzip.open(os.path.join(dir1, fname + '_2.pkl.gz'), 'rb'))
    clq3 = pkl.load(gzip.open(os.path.join(dir1, fname + '_3.pkl.gz'), 'rb'))
    clq4 = pkl.load(gzip.open(os.path.join(dir1, fname + '_4.pkl.gz'), 'rb'))
    assert len(clq1) == len(clq2) == len(clq3) == len(clq4)
    nclqs = len(clq1)
    for i in range(nclqs):
        assert np.all(clq1[i] == clq2[i])
        assert np.all(clq2[i] == clq3[i])
        assert np.all(clq3[i] == clq4[i])

print('done.')
