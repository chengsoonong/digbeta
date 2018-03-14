import os
import sys
import gzip
import numpy as np
import pickle as pkl


if len(sys.argv) != 3:
    print('Usage: python', sys.argv[0], 'WORK_DIR  DATASET')
    sys.exit(0)
else:
    work_dir = sys.argv[1]
    dataset = sys.argv[2]
    
data_dir = os.path.join(work_dir, 'data/%s/setting2' % dataset)
fytrain = os.path.join(data_dir, 'Y_train_dev.pkl.gz')

Y = pkl.load(gzip.open(fytrain, 'rb')).tocsc()
N, K = Y.shape
index_set = set(np.arange(N))
np.random.seed(97531)

ustrs = ['U%d' % i for i in range(N)]
istrs = ['P%d' % j for j in range(K)]
r = 3

lines = []
for j in range(K):
    if (j+1) % 100 == 0:
        sys.stdout.write('\r%d / %d' % (j+1, K))
        sys.stdout.flush()
    pix = Y[:, j].nonzero()[0]
    nix_all = sorted(index_set - set(pix))
    np.random.permutation(nix_all)
    nix = nix_all[:r * len(pix)]
    
    lines += [','.join([ustrs[i], istrs[j], '5\n']) for i in pix]
    lines += [','.join([ustrs[i], istrs[j], '1\n']) for i in nix]

fname = os.path.join(data_dir, 'mftrain_%s.csv' % dataset)
with open(fname, 'w') as fd:
    fd.writelines(lines)
