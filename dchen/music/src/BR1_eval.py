import os
import sys
import gzip
import pickle as pkl
import numpy as np
# from sklearn.metrics import roc_auc_score
from scipy.sparse import issparse
# from tools import calc_RPrecision_HitRate
from tools import calc_metrics, diversity, softmax
# from BinaryRelevance import BinaryRelevance
# from sklearn.metrics.pairwise import cosine_similarity

TOPs = [5, 10, 20, 30, 50, 100, 200, 300, 500, 700, 1000]

if len(sys.argv) != 3:
    print('Usage:', sys.argv[0], 'WORK_DIR  DATASET')
    sys.exit(0)

work_dir = sys.argv[1]
dataset = sys.argv[2]
data_dir = os.path.join(work_dir, 'data/%s/coldstart/setting1' % dataset)
fsplit = os.path.join(data_dir, 'br1/br1.%s.split' % dataset)
fperf = os.path.join(data_dir, 'perf-br1.pkl')
X = pkl.load(gzip.open(os.path.join(data_dir, 'X_test.pkl.gz'), 'rb'))
X_test = np.hstack([np.ones((X.shape[0], 1)), X])
Y_test = pkl.load(gzip.open(os.path.join(data_dir, 'Y_test.pkl.gz'), 'rb'))
song2pop = pkl.load(gzip.open(os.path.join(data_dir, 'song2pop.pkl.gz'), 'rb'))
songs = pkl.load(gzip.open(os.path.join(data_dir, 'songs_train_dev_test_s1.pkl.gz'), 'rb'))
test_songs = [sid for sid, _ in songs['test_song_set']]
index2song = {ix: sid for ix, sid in enumerate(test_songs)}
song2artist = pkl.load(gzip.open('data/msd/song2artist.pkl.gz', 'rb'))
song2genre = pkl.load(gzip.open('data/msd/song2genre.pkl.gz', 'rb'))

cliques_all = pkl.load(gzip.open(os.path.join(data_dir, 'cliques_trndev.pkl.gz'), 'rb'))
U = len(cliques_all)
pl2u = np.zeros(Y_test.shape[1], dtype=np.int32)
for u in range(U):
    clq = cliques_all[u]
    pl2u[clq] = u

rps = []
hitrates = {top: [] for top in TOPs}
aucs = []
spreads = []
novelties = {top: dict() for top in TOPs}
# diversities = []
artist_diversities = {top: [] for top in TOPs}
genre_diversities = {top: [] for top in TOPs}
np.random.seed(0)
with open(fsplit, 'r') as fd:
    for line in fd:
        start, end = line.strip().split(' ')
        print(start, end)
        fname = os.path.join(data_dir, 'br1/br1-%s-%s-%s.pkl.gz' % (dataset, start, end))
        br = pkl.load(gzip.open(fname, 'rb'))
        preds = br.predict(X_test)
        for j in range(int(start), int(end)):
            y_true = Y_test[:, j]
            if issparse(y_true):
                y_true = y_true.toarray().reshape(-1)
            else:
                y_true = y_true.reshape(-1)
            npos = y_true.sum()
            if npos < 1:
                continue
            y_pred = preds[:, j-int(start)].reshape(-1)
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
            u = pl2u[j]
            for top in TOPs:
                nov = np.mean([-np.log2(song2pop[index2song[ix]]) for ix in sortix[:top]])
                try:
                    novelties[top][u].append(nov)
                except KeyError:
                    novelties[top][u] = [nov]

            # compute diversity@100
            # csd = 1. / cosine_similarity(X_test[sortix[:100], :])
            # diversities.append((csd.sum() - np.trace(csd)) / (100 * 99))

            # artist/genre diversity
            for top in TOPs:
                artist_vec = np.array([song2artist[index2song[ix]] if index2song[ix] in song2artist
                                       else str(np.random.rand()) for ix in sortix[:top]])
                genre_vec = np.array([song2genre[index2song[ix]] if index2song[ix] in song2genre
                                      else str(np.random.rand()) for ix in sortix[:top]])
                artist_diversities[top].append(diversity(artist_vec))
                genre_diversities[top].append(diversity(genre_vec))

print('\n%d, %d' % (len(rps), Y_test.shape[1]))
br1_perf = {dataset: {'Test': {'R-Precision': np.mean(rps),
                               'Hit-Rate': {top: np.mean(hitrates[top]) for top in TOPs},
                               'AUC': np.mean(aucs),
                               'Spread': np.mean(spreads),
                               'Novelty': {t: np.mean([np.mean(novelties[t][u]) for u in novelties[t]]) for t in TOPs},
                               'Artist-Diversity': {top: np.mean(artist_diversities[top]) for top in TOPs},
                               'Genre-Diversity': {top: np.mean(genre_diversities[top]) for top in TOPs}},
                      'Test_All': {'R-Precision': rps,
                                   'Hit-Rate': {top: hitrates[top] for top in TOPs},
                                   'AUC': aucs,
                                   'Spread': spreads,
                                   'Novelty': novelties,
                                   'Artist-Diversity': artist_diversities,
                                   'Genre-Diversity': genre_diversities}}}
pkl.dump(br1_perf, open(fperf, 'wb'))
print(len(rps), Y_test.shape[1])
print(br1_perf[dataset]['Test'])
