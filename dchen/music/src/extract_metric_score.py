import os
import sys
import gzip
import pickle as pkl


if len(sys.argv) != 2:
    print('Usage:', sys.argv[0], 'DIR')
    sys.exit(0)

dir1 = sys.argv[1]
for f in sorted(os.listdir(dir1)):
    clf = pkl.load(gzip.open(os.path.join(dir1, f), 'rb'))
    if hasattr(clf, 'metric_score'):
        print('%s | %s' % (str(clf.metric_score), f))
