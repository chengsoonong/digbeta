import os
import sys
import numpy as np
import pickle as pkl

if len(sys.argv) != 4:
    print('Usage: python', sys.argv[0], 'WORK_DIR  C  N_JOB')
    sys.exit(0)
else:
    work_dir = sys.argv[1]
    C = float(sys.argv[2])
    n_jobs = int(sys.argv[3])

data_dir = os.path.join(work_dir, 'data')
src_dir = os.path.join(work_dir, 'src')
sys.path.append(src_dir)

from BinaryRelevance import BinaryRelevance
from evaluate import evaluate_minibatch, calc_F1, calc_precisionK

pkl_data_dir = os.path.join(data_dir, 'aotm-2011/setting2')
fxtrain = os.path.join(pkl_data_dir, 'X_train_audio.pkl')
fytrain = os.path.join(pkl_data_dir, 'Y_train_audio.pkl')
fxdev = os.path.join(pkl_data_dir, 'X_dev_audio.pkl')
fydev = os.path.join(pkl_data_dir, 'Y_dev_audio.pkl')

X_train = pkl.load(open(fxtrain, 'rb'))
Y_train = pkl.load(open(fytrain, 'rb'))
X_dev = pkl.load(open(fxdev,   'rb'))
Y_dev = pkl.load(open(fydev,   'rb'))

print('C: %g' % C)

clf = BinaryRelevance(C=C, n_jobs=n_jobs)
clf.fit(X_train, Y_train)

# evaluate F1
F1 = evaluate_minibatch(clf, calc_F1, X_dev, Y_dev, threshold=0, batch_size=2000)
print('F1: %g' % np.mean(F1))

# evaluate Precision@K
pak = evaluate_minibatch(clf, calc_precisionK, X_dev, Y_dev, threshold=None, batch_size=2000)
print('Precision@K: %g' % np.mean(pak))

fmodel = os.path.join(pkl_data_dir, 'br-aotm2011-C-%g.pkl' % C)
pkl.dump(clf, open(fmodel, 'wb'))
