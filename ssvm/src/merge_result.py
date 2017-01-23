import os, sys
import pickle as pkl

if len(sys.argv) != 3:
    print('Usage: %s  DATA_DIR  OUTPUT_NAME' % sys.argv[0])
    sys.exit(0)

data_dir = sys.argv[1]
fout = sys.argv[2]

files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
recdicts = [pkl.load(open(os.path.join(data_dir, f), 'rb')) for f in files]

recdict_all = dict()
for dict_ in recdicts:
    if len(dict_) > 0:
        assert(len(dict_) == 1)
        key = list(dict_.keys())[0]
        assert(key not in recdict_all)
        recdict_all[key] = dict_[key]

pkl.dump(recdict_all, open(fout, 'wb'))
print('%d records' % len(recdict_all))
