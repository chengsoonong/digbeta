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

pkl_data_dir = os.path.join(data_dir, 'aotm-2011')
fxtrain = os.path.join(pkl_data_dir, 'X_train_audio.pkl')
fytrain = os.path.join(pkl_data_dir, 'Y_train_audio.pkl')
fxdev   = os.path.join(pkl_data_dir, 'X_dev_audio.pkl')
fydev   = os.path.join(pkl_data_dir, 'Y_dev_audio.pkl')

X_train = pkl.load(open(fxtrain, 'rb'))
Y_train = pkl.load(open(fytrain, 'rb'))
#X_dev   = pkl.load(open(fxdev,   'rb'))
#Y_dev   = pkl.load(open(fydev,   'rb'))

print('C: %g, p: %g' % (C, p))

clf = PClassificationMLC(C=C, p=p, weighting=True)
clf.fit_SGD(X_train, Y_train, batch_size=1000, n_epochs=20, learning_rate=0.05)

fmodel = os.path.join(pkl_data_dir, 'br-aotm2011-C-%g-p-%g.pkl' % (C, p))
pkl.dump(clf, open(fmodel, 'wb'))
#print('F1:', avgF1(Y_test, clf.decision_function(X_test)))
