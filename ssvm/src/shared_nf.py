import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import kendalltau


LOG_SMALL = -10   # log(x) when x is a very small positive real number
LOG_ZERO = -1000  # log(0)
BIN_CLUSTER = 5   # discritization parameter
LOG_TRANSITION = False


"""
POI Features given query (startPOI, nPOI):
- category: one-hot encoding of POI category, encode True as 1 and False as -1
- neighbourhood: one-hot encoding of POI cluster, encode True as 1 and False as -1
- popularity: log of POI popularity, i.e., the number of distinct users that visited the POI
- nVisit: log of the total number of visit by all users
- nPhotoTotal: log of the total number of photos taken at this POI
- nPhotoMean: log of the average number of photos taken at this POI
- nPhotoP10: log of the 10th percentile of the number of photos taken at this POI
- nPhotoP50: log of the 50th percentile of the number of photos taken at this POI
- nPhotoP90: log of the 90th percentile of the number of photos taken at this POI
- durationTotal: log of the accumulated POI visit duration
- durationMean: log of the average POI visit duration
- durationP10: log of the 10th percentile of POI visit duration
- durationP50: log of the 50th percentile of POI visit duration
- durationP90: log of the 90th percentile of POI visit duration
- trajLen: trajectory length, i.e., the number of POIs nPOI in trajectory, copy from query
- sameCategory: 1 if POI category is the same as that of startPOI, -1 otherwise
- sameNeighbourhood: 1 if POI resides in the same cluster as that of startPOI, -1 otherwise
- diffPopularity: difference in POI popularity from that of startPOI (NO LOG as it could be negative)
- diffNVisit: difference in the total number of visit from that of startPOI
- diffNPhotoTotal: difference in the total number of photos from that of startPOI
- diffNPhotoMean: difference in the average number of photos from that of startPOI
- diffNPhotoP10: difference in the 10th percentile of the number of photos from that of startPOI
- diffNPhotoP50: difference in the 50th percentile of the number of photos from that of startPOI
- diffNPhotoP90: difference in the 90th percentile of the number of photos from that of startPOI
- diffDurationTotal: difference in the accumulated POI visit duration from that of startPOI
- diffDurationMean: difference in the average POI visit duration from that of startPOI
- diffDurationP10: difference in the 10th percentile of POI visit duration from that of startPOI
- diffDurationP50: difference in the 50th percentile of POI visit duration from that of startPOI
- diffDurationP90: difference in the 90th percentile of POI visit duration from that of startPOI
- distance: distance (haversine formula) from startPOI
"""
DF_COLUMNS = ['poiID', 'label', 'queryID', 'category', 'neighbourhood', 'popularity',
              'nVisit', 'avgDuration',
              'trajLen',
              'sameCatStart', 'distStart',
              'diffPopStart', 'diffNVisitStart', 'diffDurationStart', 'sameNeighbourhoodStart']

FEATURES = ['category', 'neighbourhood', 'popularity', 'nVisit',
            'nPhotoTotal', 'nPhotoMean', 'nPhotoP10', 'nPhotoP50', 'nPhotoP90',
            'durationTotal', 'durationMean', 'durationP10', 'durationP50', 'durationP90',
            'trajLen', 'sameCategory', 'sameNeighbourhood', 'diffPopularity', 'diffNVisit',
            'diffNPhotoTotal', 'diffNPhotoMean', 'diffNPhotoP10', 'diffNPhotoP50', 'diffNPhotoP90',
            'diffDurationTotal', 'diffDurationMean', 'diffDurationP10', 'diffDurationP50', 'diffDurationP90',
            'distance']


class TrajData:
    """POI and Trajectory Data"""
    def __init__(self, dat_ix, data_dir='data/data-new', debug=False):
        self.dat_suffix = ['Osak', 'Glas', 'Edin', 'Toro', 'Melb']
        assert(type(dat_ix) == int)
        assert(0 <= dat_ix < len(self.dat_suffix))

        self.debug = debug
        self.data_dir = data_dir
        self.load_data(dat_ix)
        self.trajid_set_all = sorted(self.traj_all['trajID'].unique().tolist())
        self.poi_info_all = self.calc_poi_info(self.trajid_set_all)
        self.calc_auxiliary_data()
        self.traj_user = self.traj_all[['userID', 'trajID']].groupby('trajID').first()

        if self.debug is True:
            num_user = self.traj_all['userID'].unique().shape[0]
            num_poi = self.traj_all['poiID'].unique().shape[0]
            num_traj = self.traj_all['trajID'].unique().shape[0]
            print('dataset:', self.dat_suffix[dat_ix])
            print('#user:', num_user)
            print('#poi:', num_poi)
            print('#traj:', num_traj)
            print('#traj/user:', num_traj / num_user)
            print('#traj (length >= 2):', self.traj_all[self.traj_all['trajLen'] >= 2]['trajID'].unique().shape[0])
            print('#traj length max:', self.traj_all['trajLen'].max())
            print('#query tuple:', len(self.QUERY_ID_DICT))

    def load_data(self, dat_ix):
        fpoi = os.path.join(self.data_dir, 'poi-' + self.dat_suffix[dat_ix] + '.csv')
        ftraj = os.path.join(self.data_dir, 'traj-' + self.dat_suffix[dat_ix] + '.csv')
        # fpoi = os.path.realpath(fpoi); ftraj = os.path.realpath(ftraj)
        # print(fpoi); print(ftraj)
        assert(os.path.exists(fpoi))
        assert(os.path.exists(ftraj))

        self.poi_all = pd.read_csv(fpoi)
        self.poi_all.set_index('poiID', inplace=True)
        self.traj_all = pd.read_csv(ftraj)

    def extract_traj(self, tid):
        traj = self.traj_all[self.traj_all['trajID'] == tid].copy()
        traj.sort_values(by=['startTime'], ascending=True, inplace=True)
        return traj['poiID'].tolist()

    def calc_poi_info(self, trajid_list):
        assert(len(trajid_list) > 0)
        poi_df = self.traj_all[self.traj_all['trajID'].isin(trajid_list)][['poiID', '#photo', 'poiDuration']].copy()
        photo_gb = poi_df[['poiID', '#photo']].groupby('poiID')
        duration_gb = poi_df[['poiID', 'poiDuration']].groupby('poiID')
        poi_info = photo_gb.agg([np.size, np.sum, np.mean]).copy()
        poi_info.columns = poi_info.columns.droplevel()
        poi_info.rename(columns={'size': 'nVisit', 'sum': 'nPhotoTotal', 'mean': 'nPhotoMean'}, inplace=True)
        poi_info['nPhotoP10'] = photo_gb.quantile(0.1).loc[poi_info.index, '#photo']
        poi_info['nPhotoP50'] = photo_gb.quantile(0.5).loc[poi_info.index, '#photo']
        poi_info['nPhotoP90'] = photo_gb.quantile(0.9).loc[poi_info.index, '#photo']
        poi_info['durationTotal'] = duration_gb.sum().loc[poi_info.index, 'poiDuration']
        poi_info['durationMean'] = duration_gb.mean().loc[poi_info.index, 'poiDuration']
        poi_info['durationP10'] = duration_gb.quantile(0.1).loc[poi_info.index, 'poiDuration']
        poi_info['durationP50'] = duration_gb.quantile(0.5).loc[poi_info.index, 'poiDuration']
        poi_info['durationP90'] = duration_gb.quantile(0.9).loc[poi_info.index, 'poiDuration']
        poi_info['poiCat'] = self.poi_all.loc[poi_info.index, 'poiCat']
        poi_info['poiLon'] = self.poi_all.loc[poi_info.index, 'poiLon']
        poi_info['poiLat'] = self.poi_all.loc[poi_info.index, 'poiLat']

        # POI popularity: the number of distinct users that visited the POI
        pop_df = self.traj_all[self.traj_all['trajID'].isin(trajid_list)][['poiID', 'userID']].copy()
        pop_df = pop_df.groupby('poiID').agg(pd.Series.nunique)
        pop_df.rename(columns={'userID': 'nunique'}, inplace=True)
        poi_info['popularity'] = pop_df.loc[poi_info.index, 'nunique']

        poi_info.fillna(value=0.0, inplace=True)  # replace NaN with 0.0

        return poi_info.copy()

    def calc_auxiliary_data(self):
        self.POI_DISTMAT = pd.DataFrame(data=np.zeros((self.poi_all.shape[0], self.poi_all.shape[0]), dtype=np.float),
                                        index=self.poi_all.index, columns=self.poi_all.index)
        for ix in self.poi_all.index:
            self.POI_DISTMAT.loc[ix] = calc_dist_vec(self.poi_all.loc[ix, 'poiLon'], self.poi_all.loc[ix, 'poiLat'],
                                                     self.poi_all['poiLon'], self.poi_all['poiLat'])

        # dictionary maps every trajectory ID to the actual trajectory
        self.traj_dict = {tid: self.extract_traj(tid) for tid in self.trajid_set_all}

        # define a query (in IR terminology) using tuple (start POI, #POI)
        self.TRAJID_GROUP_DICT = dict()
        for tid in sorted(self.traj_dict.keys()):
            if len(self.traj_dict[tid]) >= 2:
                key = (self.traj_dict[tid][0], len(self.traj_dict[tid]))
                if key in self.TRAJID_GROUP_DICT:
                    self.TRAJID_GROUP_DICT[key].add(tid)
                else:
                    self.TRAJID_GROUP_DICT[key] = set({tid})

        # (start, length) --> qid
        self.QUERY_ID_DICT = {query: ix for ix, query in enumerate(sorted(self.TRAJID_GROUP_DICT.keys()))}

        # POI ID --> index
        self.POI_ID_DICT = {poi: ix for ix, poi in enumerate(sorted(self.poi_all.index.tolist()))}

        # the list of POI categories
        poi_cats = self.poi_all['poiCat'].unique().tolist()
        self.POI_CAT_LIST = sorted(poi_cats)

        # discretize POI popularity with uniform log-scale bins
        poi_pops = self.poi_info_all['popularity'].values
        expo_pop1 = np.log10(max(1, min(poi_pops)))
        expo_pop2 = np.log10(max(poi_pops))
        self.LOGBINS_POP = np.logspace(np.floor(expo_pop1), np.ceil(expo_pop2), BIN_CLUSTER + 1)
        self.LOGBINS_POP[0] = 0  # deal with underflow
        if self.LOGBINS_POP[-1] < self.poi_info_all['popularity'].max():
            self.LOGBINS_POP[-1] = self.poi_info_all['popularity'].max() + 1

        # discretize the number of POI visit with uniform log-scale bins.
        poi_visits = self.poi_info_all['nVisit'].values
        expo_visit1 = np.log10(max(1, min(poi_visits)))
        expo_visit2 = np.log10(max(poi_visits))
        self.LOGBINS_VISIT = np.logspace(np.floor(expo_visit1), np.ceil(expo_visit2), BIN_CLUSTER + 1)
        self.LOGBINS_VISIT[0] = 0  # deal with underflow
        if self.LOGBINS_VISIT[-1] < self.poi_info_all['nVisit'].max():
            self.LOGBINS_VISIT[-1] = self.poi_info_all['nVisit'].max() + 1

        # discretize the average visit duration with uniform log-scale bins
        poi_durations = self.poi_info_all['durationMean'].values
        expo_duration1 = np.log10(max(1, min(poi_durations)))
        expo_duration2 = np.log10(max(poi_durations))
        self.LOGBINS_DURATION = np.logspace(np.floor(expo_duration1), np.ceil(expo_duration2), BIN_CLUSTER + 1)
        self.LOGBINS_DURATION[0] = 0  # deal with underflow
        self.LOGBINS_DURATION[-1] = np.power(10, expo_duration2 + 2)

        # KMeans in scikit-learn seems unable to use customized distance metric and
        # there's no implementation of Haversine formula, use Euclidean distance to approximate it.
        X = self.poi_all[['poiLon', 'poiLat']]
        kmeans = KMeans(n_clusters=BIN_CLUSTER, random_state=987654321)
        kmeans.fit(X)
        clusters = kmeans.predict(X)
        self.POI_CLUSTER_LIST = sorted(np.unique(clusters))
        self.POI_CLUSTERS = pd.DataFrame(data=clusters, index=self.poi_all.index)
        self.POI_CLUSTERS.index.name = 'poiID'
        self.POI_CLUSTERS.rename(columns={0: 'clusterID'}, inplace=True)
        self.POI_CLUSTERS['clusterID'] = self.POI_CLUSTERS['clusterID'].astype(np.int)

    """
    Factorised Transition Probabilities between POIs
    Estimate a transition matrix for each feature of POI,
    transition probabilities between different POIs can be computed by taking the Kronecker product of
    the individual transition matrix corresponding to each feature (with normalisation and a few constraints).
    POI features used to factorise transition matrix of Markov Chain with POI features (vector) as states:
    - Category of POI
    - Popularity of POI (discritize with uniform log-scale bins, #bins=5)
    - The number of POI visits (discritize with uniform log-scale bins, #bins=5)
    - The average visit duration of POI (discritise with uniform log-scale bins, #bins=5)
    - The neighborhood relationship between POIs (clustering POI(lat, lon) using k-means, #clusters=5)
    """

    def gen_transmat_cat(self, trajid_list, poi_info):
        """Compute Transition Matrix between POI Cateogries"""
        n_cats = len(self.POI_CAT_LIST)
        transmat_cat_cnt = pd.DataFrame(data=np.zeros((n_cats, n_cats), dtype=np.float),
                                        columns=self.POI_CAT_LIST, index=self.POI_CAT_LIST)
        for tid in trajid_list:
            t = self.traj_dict[tid]
            if len(t) >= 2:
                for pi in range(len(t) - 1):
                    p1 = t[pi]
                    p2 = t[pi + 1]
                    assert(p1 in poi_info.index and p2 in poi_info.index)
                    cat1 = poi_info.loc[p1, 'poiCat']
                    cat2 = poi_info.loc[p2, 'poiCat']
                    transmat_cat_cnt.loc[cat1, cat2] += 1
        return normalise_transmat(transmat_cat_cnt)

    def gen_transmat_pop(self, trajid_list, poi_info):
        """Compute Transition Matrix between POI Popularity Classes"""
        nbins = len(self.LOGBINS_POP) - 1
        transmat_pop_cnt = pd.DataFrame(data=np.zeros((nbins, nbins), dtype=np.float),
                                        columns=np.arange(1, nbins + 1), index=np.arange(1, nbins + 1))
        for tid in trajid_list:
            t = self.traj_dict[tid]
            if len(t) > 1:
                for pi in range(len(t) - 1):
                    p1 = t[pi]
                    p2 = t[pi + 1]
                    assert(p1 in poi_info.index and p2 in poi_info.index)
                    pop1 = poi_info.loc[p1, 'popularity']
                    pop2 = poi_info.loc[p2, 'popularity']
                    pc1, pc2 = np.digitize([pop1, pop2], self.LOGBINS_POP)
                    transmat_pop_cnt.loc[pc1, pc2] += 1
        return normalise_transmat(transmat_pop_cnt)

    def gen_transmat_visit(self, trajid_list, poi_info):
        """Compute Transition Matrix between the Number of POI Visit Classes"""
        nbins = len(self.LOGBINS_VISIT) - 1
        transmat_visit_cnt = pd.DataFrame(data=np.zeros((nbins, nbins), dtype=np.float),
                                          columns=np.arange(1, nbins + 1), index=np.arange(1, nbins + 1))
        for tid in trajid_list:
            t = self.traj_dict[tid]
            if len(t) > 1:
                for pi in range(len(t) - 1):
                    p1 = t[pi]
                    p2 = t[pi + 1]
                    assert(p1 in poi_info.index and p2 in poi_info.index)
                    visit1 = poi_info.loc[p1, 'nVisit']
                    visit2 = poi_info.loc[p2, 'nVisit']
                    vc1, vc2 = np.digitize([visit1, visit2], self.LOGBINS_VISIT)
                    transmat_visit_cnt.loc[vc1, vc2] += 1
        return normalise_transmat(transmat_visit_cnt)

    def gen_transmat_duration(self, trajid_list, poi_info):
        """Compute Transition Matrix between POI Average Visit Duration Classes"""
        nbins = len(self.LOGBINS_DURATION) - 1
        transmat_duration_cnt = pd.DataFrame(data=np.zeros((nbins, nbins), dtype=np.float),
                                             columns=np.arange(1, nbins + 1), index=np.arange(1, nbins + 1))
        for tid in trajid_list:
            t = self.traj_dict[tid]
            if len(t) > 1:
                for pi in range(len(t) - 1):
                    p1 = t[pi]
                    p2 = t[pi + 1]
                    assert(p1 in poi_info.index and p2 in poi_info.index)
                    d1 = poi_info.loc[p1, 'durationMean']
                    d2 = poi_info.loc[p2, 'durationMean']
                    dc1, dc2 = np.digitize([d1, d2], self.LOGBINS_DURATION)
                    transmat_duration_cnt.loc[dc1, dc2] += 1
        return normalise_transmat(transmat_duration_cnt)

    def gen_transmat_neighbor(self, trajid_list, poi_info):
        """Compute Transition Matrix between POI Neighborhood Classes"""
        nclusters = len(self.POI_CLUSTERS['clusterID'].unique())
        transmat_neighbor_cnt = pd.DataFrame(data=np.zeros((nclusters, nclusters), dtype=np.float),
                                             columns=np.arange(nclusters), index=np.arange(nclusters))
        for tid in trajid_list:
            t = self.traj_dict[tid]
            if len(t) > 1:
                for pi in range(len(t) - 1):
                    p1 = t[pi]
                    p2 = t[pi + 1]
                    assert(p1 in poi_info.index and p2 in poi_info.index)
                    c1 = self.POI_CLUSTERS.loc[p1, 'clusterID']
                    c2 = self.POI_CLUSTERS.loc[p2, 'clusterID']
                    transmat_neighbor_cnt.loc[c1, c2] += 1
        return normalise_transmat(transmat_neighbor_cnt)


def evaluate(dat_obj, query, y_hat_list, use_max=True):
    assert(type(dat_obj) == TrajData)
    assert(query in dat_obj.TRAJID_GROUP_DICT)
    assert(type(y_hat_list) == list)
    y_true_list = [dat_obj.traj_dict[tid] for tid in dat_obj.TRAJID_GROUP_DICT[query]]
    F1_list = []
    pF1_list = []
    Tau_list = []
    for y_hat in y_hat_list:
        F1, pF1, tau = calc_metrics(y_hat, y_true_list, dat_obj.POI_ID_DICT, use_max)
        F1_list.append(F1)
        pF1_list.append(pF1)
        Tau_list.append(tau)
    maxix = np.argmax(F1_list)  # max of pF1_list or Tau_list?
    return (F1_list[maxix], pF1_list[maxix], Tau_list[maxix])


def normalise_transmat(transmat_cnt):
    """
    We count the number of transition first,
    then normalise each row while taking care of zero by adding each cell a number eps=1
    """
    transmat = transmat_cnt.copy()
    assert(isinstance(transmat, pd.DataFrame))
    for row in range(transmat.index.shape[0]):
        rowsum = np.sum(transmat.iloc[row] + 1)
        assert(rowsum > 0)
        transmat.iloc[row] = (transmat.iloc[row] + 1) / rowsum
    return transmat


def calc_dist_vec(longitudes1, latitudes1, longitudes2, latitudes2):
        """Calculate the distance (unit: km) between two places on earth using Haversine formula (vectorised)"""
        # convert degrees to radians
        lng1 = np.radians(longitudes1)
        lat1 = np.radians(latitudes1)
        lng2 = np.radians(longitudes2)
        lat2 = np.radians(latitudes2)
        radius = 6371.0088  # mean earth radius, en.wikipedia.org/wiki/Earth_radius#Mean_radius

        # The haversine formula, en.wikipedia.org/wiki/Great-circle_distance
        dlng = np.fabs(lng1 - lng2)
        dlat = np.fabs(lat1 - lat2)
        dist = 2 * radius * np.arcsin(
            np.sqrt((np.sin(0.5 * dlat))**2 + np.cos(lat1) * np.cos(lat2) * (np.sin(0.5 * dlng))**2))
        return dist


def calc_metrics(y_hat, y_true_list, poi_id_dict, use_max=True):
    """
    compute all evaluation metrics:
    - F1 score on points,
    - F1 score on pairs,
    - Kendall's tau.

    y_hat - the prediction for query x
    y_true_list - ground truth trajecories for query x
    poi_id_dict - a mapping: POI ID --> index
    use_max - True: use the maximum of all scores for each metric, False: use the mean
    """
    assert(len(y_hat) > 0)
    assert(len(y_true_list) > 0)

    F1 = np.zeros(len(y_true_list), dtype=np.float)
    pF1 = np.zeros(len(y_true_list), dtype=np.float)
    Tau = np.zeros(len(y_true_list), dtype=np.float)
    for j in range(len(y_true_list)):
        assert(len(y_hat) == len(y_true_list[j]))
        F1[j] = calc_F1(y_true_list[j], y_hat)
        pF1[j] = calc_pairsF1(y_true_list[j], y_hat)
        Tau[j] = calc_kendalltau(y_true_list[j], y_hat, poi_id_dict)

    if use_max is True:  # use maximum similarity score
        return np.max(F1), np.max(pF1), np.max(Tau)
    else:                # use mean similarity score
        return np.mean(F1), np.mean(pF1), np.mean(Tau)


def calc_F1(traj_act, traj_rec, noloop=True):
    """Compute recall, precision and F1 for recommended trajectories"""
    assert(len(traj_act) > 0)
    assert(len(traj_rec) > 0)

    if noloop is True:
        intersize = len(set(traj_act) & set(traj_rec))
    else:  # if there are sub-tours in both ground truth and prediction
        match_tags = np.zeros(len(traj_act), dtype=np.bool)
        for poi in traj_rec:
            for j in range(len(traj_act)):
                if match_tags[j] is False and poi == traj_act[j]:
                    match_tags[j] = True
                    break
        intersize = np.nonzero(match_tags)[0].shape[0]

    recall = intersize / len(traj_act)
    precision = intersize / len(traj_rec)
    F1 = 2 * precision * recall / (precision + recall)
    return F1


def calc_pairsF1(y, y_hat):
    """Compute the pairs-F1 score"""
    assert(len(y) > 0)
    assert(len(y) == len(set(y)))  # no loops in y

    # y determines the correct visiting order
    order_dict = {y[i]: i for i in range(len(y))}

    nc = 0
    for i in range(len(y_hat)):
        poi1 = y_hat[i]
        for j in range(i + 1, len(y_hat)):
            poi2 = y_hat[j]
            if poi1 in order_dict and poi2 in order_dict and poi1 != poi2:
                if order_dict[poi1] < order_dict[poi2]:
                    nc += 1

    n0 = len(y) * (len(y) - 1) // 2
    n1 = len(y_hat) * (len(y_hat) - 1) // 2

    precision = nc / n1
    recall = nc / n0
    if nc > 0:
        F1 = 2 * precision * recall / (precision + recall)
    else:
        F1 = 0
    return F1


def gen_rank(y, M, default_rank=0):
    """
    compute the rank of all POIs given a trajectory
    The ranks of all POIs in trajectory `y` should be greater than all other POIs that do not appear in `y`,
    which we require that they have the same rank (use rank `0` by default).
    y - trajectory: a sequence of POIs without duplication
    M - total number of POIs
    default_rank - the rank for all POIs do not appear in y
    """
    assert(len(y) > 0)
    assert(len(y) <= M)
    assert(default_rank >= 0)
    assert(default_rank <= M)
    rank = np.ones(M) * default_rank
    for j in range(len(y)):
        poi = y[j]
        prank = M - j
        rank[poi - 1] = prank
    return rank


def calc_kendalltau(y, y_hat, poi_id_dict):
    """Compute the Kendall's tau (taking care of ties)"""
    # assert(len(y) == len(y_hat))
    M = len(poi_id_dict)
    assert(len(y) <= M)
    assert(len(y_hat) <= M)

    r1 = gen_rank([poi_id_dict[p] for p in y], M)
    r2 = gen_rank([poi_id_dict[p] for p in y_hat], M)

    return kendalltau(r1, r2)[0]


def vars_equal(d1, d2):
    """ Check equality of two variables"""

    def list_equal(d1, d2):
        assert type(d1) == type(d2) == list
        assert len(d1) == len(d2)
        for j in range(len(d1)):
            assert vars_equal(d1[j], d2[j])
        return True

    def set_equal(d1, d2):
        assert type(d1) == type(d2) == set
        assert list_equal(sorted(d1), sorted(d2))
        return True

    def dict_equal(d1, d2):
        assert type(d1) == type(d2) == dict
        assert len(d1.keys()) == len(d2.keys())
        assert pd.Series(sorted(d1.keys())).equals(pd.Series(sorted(d1.keys())))
        for key in d1.keys():
            assert vars_equal(d1[key], d2[key])
        return True

    assert type(d1) == type(d2)
    int_types = {int, np.int0, np.int8, np.int16, np.int32, np.int64}
    float_types = {float, np.float16, np.float32, np.float64, np.float128}
    if type(d1) == str:
        assert d1 == d2
    elif type(d1) in int_types:
        assert d1 == d2
    elif type(d1) in float_types:
        assert np.isclose(d1, d2)  # np.isclose(10, 10.0001) is True
    elif type(d1) == list:
        assert list_equal(d1, d2)
    elif type(d1) == set:
        assert set_equal(d1, d2)
    elif type(d1) == dict:
        assert dict_equal(d1, d2)
    elif type(d1) == np.ndarray:
        assert np.allclose(d1, d2)
    elif type(d1) in {pd.DataFrame, pd.Series}:
        assert d1.equals(d2)
    else:
        assert False, 'UNrecognised type: %s\n' % type(d1)
    return True


def do_evaluation(dat_obj, recdict):
    assert(type(dat_obj) == TrajData)

    F1_list = []
    pF1_list = []
    Tau_list = []
    for key in sorted(recdict.keys()):
        F1, pF1, tau = evaluate(dat_obj, key, recdict[key]['PRED'])
        F1_list.append(F1)
        pF1_list.append(pF1)
        Tau_list.append(tau)
    nF1 = np.sum([True if np.abs(x - 1.0) < 1e-6 else False for x in F1_list])
    npF1 = np.sum([True if np.abs(x - 1.0) < 1e-6 else False for x in pF1_list])

    print('F1 (%.3f, %.3f), pairsF1 (%.3f, %.3f), Tau (%.3f, %.3f), perfectF1: %d/%d, perfectPairsF1: %d/%d' %
          (np.mean(F1_list), np.std(F1_list) / np.sqrt(len(F1_list)),
           np.mean(pF1_list), np.std(pF1_list) / np.sqrt(len(pF1_list)),
           np.mean(Tau_list), np.std(Tau_list) / np.sqrt(len(Tau_list)), nF1, len(F1_list), npF1, len(pF1_list)))
