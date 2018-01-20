import os, sys
import numpy as np
import pickle as pkl

if len(sys.argv) != 4:
    print('Usage: python', sys.argv[0], 'WORK_DIR  C  P')
    sys.exit(0)
else:
    work_dir = sys.argv[1]
    C = float(sys.argv[2])
    p = float(sys.argv[3])

data_dir = os.path.join(work_dir, 'data')
src_dir = os.path.join(work_dir, 'src')
sys.path.append(src_dir)

from PClassificationMLC import PClassificationMLC
from evaluate import evaluate_SGD, calc_F1, calc_precisionK

pkl_data_dir = os.path.join(data_dir, 'aotm-2011')
fxtrain = os.path.join(pkl_data_dir, 'X_train_audio.pkl')
fytrain = os.path.join(pkl_data_dir, 'Y_train_audio.pkl')
fxdev   = os.path.join(pkl_data_dir, 'X_dev_audio.pkl')
fydev   = os.path.join(pkl_data_dir, 'Y_dev_audio.pkl')

X_train = pkl.load(open(fxtrain, 'rb'))
Y_train = pkl.load(open(fytrain, 'rb'))
X_dev   = pkl.load(open(fxdev,   'rb'))
Y_dev   = pkl.load(open(fydev,   'rb'))

thresholds = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]

print('C: %g, p: %g' % (C, p))

clf = PClassificationMLC(C=C, p=p, weighting=True)
clf.fit_SGD(X_train, Y_train, batch_size=1800, n_epochs=30, learning_rate=0.05, verbose=1)

if clf.trained is True:
    fmodel = os.path.join(pkl_data_dir, 'br-aotm2011-C-%g-p-%g.pkl' % (C, p))
    #pkl.dump(clf, open(fmodel, 'wb'))
    clf.dump_params(fmodel)

    # evaluate F1
    for th in thresholds:
        F1 = evaluate_SGD(clf, calc_F1, X_dev, Y_dev, threshold=th, batch_size=1800)
        print('Threshold: %.2f, F1: %g' % (th, np.mean(F1)))

    # evaluate Precision@K
    pak = evaluate_SGD(clf, calc_precisionK, X_dev, Y_dev, threshold=None, batch_size=1800)
    print('Precision@K: %g' % np.mean(pak))

