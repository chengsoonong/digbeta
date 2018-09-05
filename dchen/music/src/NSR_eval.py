import os
import sys
import gzip
import numpy as np
import pickle as pkl
from scipy.sparse import issparse
# from sklearn.metrics.pairwise import cosine_similarity
from tools import calc_metrics, softmax  # diversity

TOPs = [5, 10, 20, 30, 50, 100, 200, 300, 500, 700, 1000]

if len(sys.argv) != 4:
    print('Usage:', sys.argv[0], 'WORK_DIR  DATASET  MODEL_FILE.pkl.gz')
    sys.exit(0)

work_dir = sys.argv[1]
dataset = sys.argv[2]
fmodel = sys.argv[3]

assert fmodel.endswith('pkl.gz')
assert os.path.exists(fmodel)

data_dir = os.path.join(work_dir, 'data/%s/coldstart/setting1' % dataset)
fperf = os.path.join(data_dir, 'perf-mtc.pkl')
X_test = pkl.load(gzip.open(os.path.join(data_dir, 'X_test.pkl.gz'), 'rb'))
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
Y_test = pkl.load(gzip.open(os.path.join(data_dir, 'Y_test.pkl.gz'), 'rb'))
song2pop = pkl.load(gzip.open(os.path.join(data_dir, 'song2pop.pkl.gz'), 'rb'))
songs = pkl.load(gzip.open(os.path.join(data_dir, 'songs_train_dev_test_s1.pkl.gz'), 'rb'))
test_songs = [sid for sid, _ in songs['test_song_set']]
index2song = {ix: sid for ix, sid in enumerate(test_songs)}
song2artist = pkl.load(gzip.open('data/msd/song2artist.pkl.gz', 'rb'))
song2genre = pkl.load(gzip.open('data/msd/song2genre.pkl.gz', 'rb'))

assert issparse(Y_test)
clf = pkl.load(gzip.open(fmodel, 'rb'))
assert clf.trained is True

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
for j in range(Y_test.shape[1]):
    if (j+1) % 100 == 0:
        sys.stdout.write('\r%d / %d' % (j+1, Y_test.shape[1]))
        sys.stdout.flush()
    y_true = Y_test[:, j].toarray().reshape(-1)
    npos = y_true.sum()
    if npos < 1:
        continue
    u = clf.pl2u[j]
    wk = clf.V[u, :] + clf.W[j, :] + clf.mu
    y_pred = np.dot(X_test, wk)
    rp, hr_dict, auc = calc_metrics(y_true, y_pred, tops=TOPs)
    rps.append(rp)
    for top in TOPs:
        hitrates[top].append(hr_dict[top])
    aucs.append(auc)

    # spread
    y_pred_prob = softmax(y_pred)
    # spreads.append(-np.multiply(y_pred_prob, np.log(y_pred_prob)).sum())
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
    negIx = (1 - y_true).astype(np.bool)
    negMax = y_pred[negIx].max()
    pt = (y_pred[y_true] > negMax).sum() / npos
    ptops.append(pt)

    # compute diversity@100
    # csd = 1. / cosine_similarity(X_test[sortix[:100], :])
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
nsr_perf = {dataset: {'Test': {'R-Precision': np.mean(rps),
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
pkl.dump(nsr_perf, open(fperf, 'wb'))
print(nsr_perf[dataset]['Test'])
