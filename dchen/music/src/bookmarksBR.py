import os
import sys
import pickle as pkl
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

if len(sys.argv) != 2:
    print('Usage: python', sys.argv[0], 'WORK_DIR')
    sys.exit(0)
else:
    work_dir = sys.argv[1]

data_dir = os.path.join(work_dir, 'data')
src_dir = os.path.join(work_dir, 'src')
sys.path.append(src_dir)

from BinaryRelevance import BinaryRelevance
from evaluate import f1_score_nowarn


def avgF1(Y_true, Y_pred):
    F1 = f1_score_nowarn(Y_true, Y_pred >= 0, average='samples')
    print('\nF1: %g, #examples: %g' % (F1, Y_true.shape[0]))
    return F1


X_train = pkl.load(open(os.path.join(data_dir, 'bookmarks/bk_X_train.pkl'), 'rb'))
Y_train = pkl.load(open(os.path.join(data_dir, 'bookmarks/bk_Y_train.pkl'), 'rb'))
X_test = pkl.load(open(os.path.join(data_dir, 'bookmarks/bk_X_test.pkl'), 'rb'))
Y_test = pkl.load(open(os.path.join(data_dir, 'bookmarks/bk_Y_test.pkl'), 'rb'))

C_set = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3,
         1, 3, 10, 30, 100, 300, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6]
parameters = [{'C': C_set}]
scorer = {'F1': make_scorer(avgF1)}

fmodel = os.path.join(data_dir, 'br-bookmarks-f1.pkl')
clf = GridSearchCV(BinaryRelevance(), parameters, scoring=scorer, verbose=2, cv=5, n_jobs=10, refit='F1')
clf.fit(X_train, Y_train)
pkl.dump(clf, open(fmodel, 'wb'))
print('F1:', avgF1(Y_test, clf.decision_function(X_test)))
