import os
import sys
import gzip
import pickle as pkl
import numpy as np
from scipy.sparse import issparse
# from sklearn.metrics.pairwise import cosine_similarity
# from tools import calc_RPrecision_HitRate
from tools import calc_metrics, softmax  # diversity

TOPs = [5, 10, 20, 30, 50, 100, 200, 300, 500, 700, 1000]

if len(sys.argv) != 6:
    print('Usage:', sys.argv[0], 'WORK_DIR  DATASET  TASK(3/4)  MODEL_FILE.pkl.gz  OUTPUT_PKL')
    sys.exit(0)

work_dir = sys.argv[1]
dataset = sys.argv[2]
task = int(sys.argv[3])
fmodel = sys.argv[4]
fout = sys.argv[5]

assert fmodel.endswith('pkl.gz')
assert os.path.exists(fmodel)
assert task in [3, 4]

data_dir = os.path.join(work_dir, 'data/%s/coldstart/setting%d' % (dataset, task))
fperf = os.path.join(data_dir, fout)
X = pkl.load(gzip.open(os.path.join(data_dir, 'X.pkl.gz'), 'rb'))
X = np.hstack([np.ones((X.shape[0], 1)), X])
Y_train = pkl.load(gzip.open(os.path.join(data_dir, 'Y_train.pkl.gz'), 'rb'))
Y_test = pkl.load(gzip.open(os.path.join(data_dir, 'Y_test.pkl.gz'), 'rb'))
cliques_train = pkl.load(gzip.open(os.path.join(data_dir, 'cliques_train.pkl.gz'), 'rb'))
song2pop = pkl.load(gzip.open(os.path.join(data_dir, 'song2pop.pkl.gz'), 'rb'))
all_songs = pkl.load(gzip.open(os.path.join(data_dir, 'all_songs.pkl.gz'), 'rb'))
index2song = {ix: sid for ix, (sid, _) in enumerate(all_songs)}
cliques_all = pkl.load(gzip.open(os.path.join(data_dir, 'cliques_all.pkl.gz'), 'rb'))
song2artist = pkl.load(gzip.open('data/msd/song2artist.pkl.gz', 'rb'))
song2genre = pkl.load(gzip.open('data/msd/song2genre.pkl.gz', 'rb'))

assert issparse(Y_test)
clf = pkl.load(gzip.open(fmodel, 'rb'))
assert clf.trained is True

pl2u = np.zeros(Y_train.shape[1] + Y_test.shape[1], dtype=np.int)
if task == 3:
    U = len(cliques_train)
    assert len(cliques_all) == U
else:
    U = len(cliques_all)
for u in range(U):
    clq = cliques_all[u]
    pl2u[clq] = u
assert np.all(clf.pl2u == pl2u[:Y_train.shape[1]])

if task == 4 and dataset == '30music':
    # a different way to index user here
    pldata = pkl.load(gzip.open('%s/playlists_train_test_s4.pkl.gz' % data_dir, 'rb'))
    test_playlists = pldata['test_playlists']
    assert len(test_playlists) == Y_test.shape[1]
    _, test_user2index, cosine_similarities = pkl.load(gzip.open('%s/user_sim.pkl.gz' % data_dir, 'rb'))
    k = 10

rps = []
hitrates = {top: [] for top in TOPs}
aucs = []
spreads = []
novelties = {top: dict() for top in TOPs}
ptops = []
# diversities = []
# artist_diversities = {top: [] for top in TOPs}
# genre_diversities = {top: [] for top in TOPs}
np.random.seed(0)
offset = Y_train.shape[1]
for j in range(Y_test.shape[1]):
    if (j+1) % 100 == 0:
        sys.stdout.write('\r%d / %d' % (j+1, Y_test.shape[1]))
        sys.stdout.flush()
    y_true = Y_test[:, j].A.reshape(-1)
    assert y_true.sum() > 0
    if task == 3:
        u = pl2u[j + offset]
        wj = clf.V[u, :] + clf.mu
        y_pred = np.dot(X, wj).reshape(-1)
    else:
        if dataset == '30music':
            u = test_playlists[j][1]
            if u not in test_user2index:
                continue
            uix = test_user2index[u]
            neighbour_ix = np.argpartition(cosine_similarities[uix, :].reshape(-1), -k)[-k:]  # indices of kNN
            wj = clf.V[neighbour_ix, :].mean(axis=0).reshape(-1) + clf.mu
            # neighbour_weights = cosine_similarities[uix, neighbour_ix]
            # wj = np.dot(neighbour_weights / neighbour_weights.sum(), clf.V[neighbour_ix, :]) + clf.mu
            y_pred = np.dot(X, wj).reshape(-1)
        else:
            u = pl2u[j + offset]
            y_pred = np.dot(X, clf.mu).reshape(-1)

    # rp, hr_dict = calc_RPrecision_HitRate(y_true, y_pred, tops=TOPs)
    rp, hr_dict, auc = calc_metrics(y_true, y_pred, tops=TOPs)
    rps.append(rp)
    for top in TOPs:
        hitrates[top].append(hr_dict[top])
    aucs.append(auc)

    # spread
    y_pred_prob = softmax(y_pred)
    spreads.append(-np.dot(y_pred_prob, np.log(y_pred_prob)))

    # novelty
    sortix = np.argsort(-y_pred)
    for top in TOPs:
        nov = np.mean([-np.log2(song2pop[index2song[ix]]) for ix in sortix[:top]])
        try:
            novelties[top][u].append(nov)
        except KeyError:
            novelties[top][u] = [nov]

    # PTop: (#pos ranked above the top-ranked negative) / #pos
    assert y_true.dtype == np.bool
    npos = y_true.sum()
    assert npos > 0
    negIx = (1 - y_true).astype(np.bool)
    negMax = y_pred[negIx].max()
    pt = (y_pred[y_true] > negMax).sum() / npos
    ptops.append(pt)

    # compute diversity@100
    # csd = 1. / cosine_similarity(X[sortix[:100], :])
    # diversities.append((csd.sum() - np.trace(csd)) / (100 * 99))

    # artist/genre diversity
    # for top in TOPs:
    #     artist_vec = np.array([song2artist[index2song[ix]] if index2song[ix] in song2artist
    #                            else str(np.random.rand()) for ix in sortix[:top]])
    #     genre_vec = np.array([song2genre[index2song[ix]] if index2song[ix] in song2genre
    #                           else str(np.random.rand()) for ix in sortix[:top]])
    #     artist_diversities[top].append(diversity(artist_vec))
    #     genre_diversities[top].append(diversity(genre_vec))

print('\n%d, %d' % (len(rps), Y_test.shape[1]))
perf = {dataset: {'Test': {'R-Precision': np.mean(rps),
                           'Hit-Rate': {top: np.mean(hitrates[top]) for top in TOPs},
                           'AUC': np.mean(aucs),
                           'Spread': np.mean(spreads),
                           'Novelty': {t: np.mean([np.mean(novelties[t][u]) for u in novelties[t]]) for t in TOPs},
                           'PTop': np.mean(ptops),
                           # 'Artist-Diversity': {top: np.mean(artist_diversities[top]) for top in TOPs},
                           # 'Genre-Diversity': {top: np.mean(genre_diversities[top]) for top in TOPs}},
                           },
                  'Test_All': {'R-Precision': rps,
                               'Hit-Rate': {top: hitrates[top] for top in TOPs},
                               'AUC': aucs,
                               'Spread': spreads,
                               'Novelty': novelties,
                               'PTop': ptops,
                               # 'Artist-Diversity': artist_diversities,
                               # 'Genre-Diversity': genre_diversities}}}
                               }}}
pkl.dump(perf, open(fperf, 'wb'))
print(perf[dataset]['Test'])
