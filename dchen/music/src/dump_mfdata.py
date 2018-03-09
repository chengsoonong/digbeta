import os
import sys
import gzip
import pickle as pkl


if len(sys.argv) != 4:
    print('Usage: python', sys.argv[0], 'WORK_DIR  START  END')
    sys.exit(0)
else:
    work_dir = sys.argv[1]
    start = int(sys.argv[2])
    end = int(sys.argv[3])

data_dir = os.path.join(work_dir, 'data')
src_dir = os.path.join(work_dir, 'src')
sys.path.append(src_dir)

pkl_data_dir = os.path.join(data_dir, 'aotm-2011/setting2')
fytrain = os.path.join(pkl_data_dir, 'Y_train_dev.pkl.gz')

Y = pkl.load(gzip.open(fytrain, 'rb'))
Y_part = Y[:, start:end]

ustrs = ['U%d' % i for i in range(Y_part.shape[0])]
istrs = ['P%d' % (j+start) for j in range(Y_part.shape[1])]

lines = []
for i in range(Y_part.shape[0]):
    if (i+1) % 100 == 0:
        sys.stdout.write('\r%d / %d' % (i+1, Y_part.shape[0]))
        sys.stdout.flush()
    lines += [','.join([ustrs[i], istrs[j], '5\n']) if Y_part[i, j] is True else
              ','.join([ustrs[i], istrs[j], '1\n']) for j in range(Y_part.shape[1])]

fname = os.path.join(pkl_data_dir, 'ftrain_%d_%d.csv' % (start, end))
with open(fname, 'w') as fd:
    fd.writelines(lines)
