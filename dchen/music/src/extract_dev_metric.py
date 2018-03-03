import os, sys
import gzip
import pickle as pkl
from models import PCMLC

if len(sys.argv) != 2:
    print('Usage:', sys.argv[0], 'DIR')
    sys.exit(0)

dir1 = sys.argv[1]
for f in os.listdir(dir1):
    clf = pkl.load(gzip.open(os.path.join(dir1, f), 'rb'))
    if hasattr(clf, 'dev_metric'):
        print('%s | %s' % (str(clf.dev_metric), f))
    if hasattr(clf, 'test_metric'):
        print('%s | %s' % (str(clf.test_metric), f))

