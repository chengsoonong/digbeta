import sys
import os
import pickle
import numpy as np
import random
import cvxopt

if len(sys.argv) != 5:
    print('Usage: python', sys.argv[0], 'WORK_DIR DATA_INDEX QUERY_INDEX C')
    sys.exit(0)
else:
    work_dir = sys.argv[1]
    dat_ix = int(sys.argv[2])
    qix = int(sys.argv[3])
    ssvm_C = float(sys.argv[4])

data_dir = os.path.join(work_dir, 'data')
src_dir = os.path.join(work_dir, 'src')
sys.path.append(src_dir)

from shared import TrajData, evaluate  # noqa: E402
from ssvm import SSVM  # noqa: E402
# import pyximport  # noqa: E402
# pyximport.install(setup_args={'include_dirs': np.get_include()}, reload_support=True)
from inference_lv import do_inference_list_viterbi  # noqa: E402

random.seed(1234554321)
np.random.seed(123456789)
cvxopt.base.setseed(123456789)

dat_obj = TrajData(dat_ix, data_dir=data_dir)

N_JOBS = 4         # number of parallel jobs
# C_SET = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000]  # regularisation parameter
MC_PORTION = 0.1   # the portion of data that sampled by Monte-Carlo cross-validation
MC_NITER = 5       # number of iterations for Monte-Carlo cross-validation
SSVM_SHARE_PARAMS = True  # share params among POIs/transitions in SSVM
SSVM_MULTI_LABEL = True  # use multi-label SSVM

inference_method = do_inference_list_viterbi

recdict_ssvm = dict()
keys = sorted(dat_obj.TRAJID_GROUP_DICT.keys())

i = qix
ps, L = keys[i]
keys_cv = keys[:i] + keys[i + 1:]

# use all training+validation set to compute POI features,
# make sure features do NOT change for training and validation
trajid_set_i = set(dat_obj.trajid_set_all) - dat_obj.TRAJID_GROUP_DICT[keys[i]]
poi_info_i = dat_obj.calc_poi_info(list(trajid_set_i))

poi_set_i = {p for tid in trajid_set_i for p in dat_obj.traj_dict[tid] if len(dat_obj.traj_dict[tid]) >= 2}
if ps not in poi_set_i:
    sys.stderr.write('start POI of query %s does not exist in training set.\n' % str(keys[i]))
    sys.exit(0)

# tune regularisation constant C
print('\n--------------- try_C: %.2f ---------------\n' % ssvm_C)
sys.stdout.flush()
F1_ssvm = []
pF1_ssvm = []
Tau_ssvm = []

# inner loop to evaluate the performance of a model with a specified C by Monte-Carlo cross validation
for j in range(MC_NITER):
    poi_list = []
    while True:  # make sure the start POI in test set are also in training set
        rand_ix = np.arange(len(keys_cv))
        np.random.shuffle(rand_ix)
        test_ix = rand_ix[:int(MC_PORTION * len(rand_ix))]
        assert(len(test_ix) > 0)
        trajid_set_train = set(dat_obj.trajid_set_all) - dat_obj.TRAJID_GROUP_DICT[keys[i]]
        for j in test_ix:
            trajid_set_train = trajid_set_train - dat_obj.TRAJID_GROUP_DICT[keys_cv[j]]
        poi_set = {p for tid in sorted(trajid_set_train) for p in dat_obj.traj_dict[tid]
                   if len(dat_obj.traj_dict[tid]) >= 2}
        good_partition = True
        for j in test_ix:
            if keys_cv[j][0] not in poi_set:
                good_partition = False
                break
        if good_partition is True:
            poi_list = sorted(poi_set)
            break

    # train
    ssvm = SSVM(inference_train=inference_method, inference_pred=inference_method,
                dat_obj=dat_obj, share_params=SSVM_SHARE_PARAMS, multi_label=SSVM_MULTI_LABEL,
                C=ssvm_C, poi_info=poi_info_i.loc[poi_list].copy())
    if ssvm.train(sorted(trajid_set_train), n_jobs=N_JOBS) is True:
        for j in test_ix:  # test
            ps_cv, L_cv = keys_cv[j]
            y_hat_list = ssvm.predict(ps_cv, L_cv)
            if y_hat_list is not None:
                F1, pF1, tau = evaluate(dat_obj, keys_cv[j], y_hat_list)
                F1_ssvm.append(F1)
                pF1_ssvm.append(pF1)
                Tau_ssvm.append(tau)
    else:
        for j in test_ix:
            F1_ssvm.append(0)
            pF1_ssvm.append(0)
            Tau_ssvm.append(0)

mean_F1 = np.mean(F1_ssvm)
mean_pF1 = np.mean(pF1_ssvm)
mean_Tau = np.mean(Tau_ssvm)

recdict_ssvm[(ps, L)] = {'mean_F1': mean_F1, 'mean_pF1': mean_pF1, 'mean_Tau': mean_Tau, 'C': ssvm_C}
fssvm = os.path.join(data_dir, 'ssvm-' + dat_obj.dat_suffix[dat_ix] + '-%g-%d.pkl' % (ssvm_C, qix))
pickle.dump(recdict_ssvm, open(fssvm, 'bw'))
