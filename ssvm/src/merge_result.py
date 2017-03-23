import os
import sys
import numpy as np
import pickle as pkl

if len(sys.argv) != 3:
    print('Usage: %s  DATA_DIR  OUTPUT_NAME' % sys.argv[0])
    sys.exit(0)

data_dir = sys.argv[1]
fout = sys.argv[2]

files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
recdicts = [pkl.load(open(os.path.join(data_dir, f), 'rb')) for f in files]

assert(len(recdicts) > 0)
recdict_all = dict()
key0 = list(recdicts[0].keys())[0]

if 'mean_F1' in recdicts[0][key0]:
    metric_dict = dict()
    for i in range(len(recdicts)):
        dict_ = recdicts[i]
        if len(dict_) > 0:
            assert(len(dict_) == 1)
            key = list(dict_.keys())[0]
            if key in metric_dict:
                metric_dict[key]['F1_list'].append(dict_[key]['mean_F1'])
                metric_dict[key]['pF1_list'].append(dict_[key]['mean_pF1'])
                metric_dict[key]['Tau_list'].append(dict_[key]['mean_Tau'])
                metric_dict[key]['Index'].append(i)
            else:
                metric_dict[key] = {'F1_list': [dict_[key]['mean_F1']], 'pF1_list': [dict_[key]['mean_pF1']],
                                    'Tau_list': [dict_[key]['mean_Tau']], 'Index': [i]}
    print('#queries: %d' % len(metric_dict))
    for key in sorted(metric_dict.keys()):
        assert(key not in recdict_all)
        maxix = np.argmax(metric_dict[key]['Tau_list'])
        dix = metric_dict[key]['Index'][maxix]
        assert(key in recdicts[dix])
        recdict_all[key] = recdicts[dix][key]['C']

else:
    for dict_ in recdicts:
        if len(dict_) > 0:
            assert(len(dict_) == 1)
            key = list(dict_.keys())[0]
            assert(key not in recdict_all)
            recdict_all[key] = dict_[key]

pkl.dump(recdict_all, open(fout, 'wb'))
print('%d records' % len(recdict_all))
