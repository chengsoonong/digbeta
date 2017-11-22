import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler
from pystruct.models import StructuredModel
from pystruct.learners import OneSlackSSVM

sys.path.append('.')
from shared import LOG_SMALL, LOG_TRANSITION, FEATURES  # noqa: E402


class SSVM:
    """Structured SVM wrapper"""
    def __init__(self, inference_train, inference_pred, dat_obj, C=1.0, share_params=True,
                 multi_label=True, poi_info=None, debug=False):
        assert(C > 0)
        self.C = C
        self.inference_train = inference_train
        self.inference_pred = inference_pred
        self.share_params = share_params
        self.multi_label = multi_label
        self.dat_obj = dat_obj
        self.debug = debug
        self.trained = False

        if poi_info is None:
            self.poi_info = None
        else:
            self.poi_info = poi_info

        self.scaler_node = MinMaxScaler(feature_range=(-1, 1), copy=False)
        self.scaler_edge = MinMaxScaler(feature_range=(-1, 1), copy=False)

    def train(self, trajid_list, n_jobs=4):
        if self.poi_info is None:
            self.poi_info = self.dat_obj.calc_poi_info(trajid_list)

        # build POI_ID <--> POI__INDEX mapping for POIs used to train CRF
        # which means only POIs in traj such that len(traj) >= 2 are included
        poi_set = {p for tid in trajid_list for p in self.dat_obj.traj_dict[tid]
                   if len(self.dat_obj.traj_dict[tid]) >= 2}
        self.poi_list = sorted(poi_set)
        self.poi_id_dict, self.poi_id_rdict = dict(), dict()
        for idx, poi in enumerate(self.poi_list):
            self.poi_id_dict[poi] = idx
            self.poi_id_rdict[idx] = poi

        # generate training data
        train_traj_list = [self.dat_obj.traj_dict[k] for k in trajid_list if len(self.dat_obj.traj_dict[k]) >= 2]
        node_features_list = Parallel(n_jobs=n_jobs)(delayed(calc_node_features)(
            tr[0], len(tr), self.poi_list, self.poi_info, self.dat_obj) for tr in train_traj_list)
        edge_features = calc_edge_features(trajid_list, self.poi_list, self.poi_info, self.dat_obj)

        # feature scaling: node features
        # should each example be flattened to one vector before scaling?
        self.fdim_node = node_features_list[0].shape
        X_node_all = np.vstack(node_features_list)
        X_node_all = self.scaler_node.fit_transform(X_node_all)
        X_node_all = X_node_all.reshape(-1, self.fdim_node[0], self.fdim_node[1])

        # feature scaling: edge features
        fdim_edge = edge_features.shape
        edge_features = self.scaler_edge.fit_transform(edge_features.reshape(fdim_edge[0] * fdim_edge[1], -1))
        self.edge_features = edge_features.reshape(fdim_edge)

        assert(len(train_traj_list) == X_node_all.shape[0])
        X_train = [(X_node_all[k, :, :],
                    self.edge_features.copy(),
                    (self.poi_id_dict[train_traj_list[k][0]], len(train_traj_list[k])))
                   for k in range(len(train_traj_list))]
        y_train = [np.array([self.poi_id_dict[k] for k in tr]) for tr in train_traj_list]
        assert(len(X_train) == len(y_train))

        # train
        sm = MyModel(inference_train=self.inference_train, inference_pred=self.inference_pred,
                     share_params=self.share_params, multi_label=self.multi_label)
        if self.debug is True:
            print('C:', self.C)
        verbose = 1 if self.debug is True else 0
        self.osssvm = OneSlackSSVM(model=sm, C=self.C, n_jobs=n_jobs, verbose=verbose)
        try:
            self.osssvm.fit(X_train, y_train, initialize=True)
            self.trained = True
            print('SSVM training finished.')
        except ValueError:
            # except:
            self.trained = False
            sys.stderr.write('SSVM training FAILED.\n')
            # raise
        return self.trained

    def predict(self, startPOI, nPOI):
        assert(self.trained is True)
        if startPOI not in self.poi_list:
            return None
        X_node_test = calc_node_features(startPOI, nPOI, self.poi_list, self.poi_info, self.dat_obj)

        # feature scaling
        # should each example be flattened to one vector before scaling?
        # X_node_test = X_node_test.reshape(1, -1) # flatten test example to a vector
        X_node_test = self.scaler_node.transform(X_node_test)
        # X_node_test = X_node_test.reshape(self.fdim)

        X_test = [(X_node_test, self.edge_features, (self.poi_id_dict[startPOI], nPOI))]
        y_hat_list = self.osssvm.predict(X_test)[0]
        # print(y_hat_list)

        return [np.array([self.poi_id_rdict[x] for x in y_hat]) for y_hat in y_hat_list]


class MyModel(StructuredModel):
    """A Sequence model"""
    def __init__(self, inference_train, inference_pred, share_params, multi_label,
                 n_states=None, n_features=None, n_edge_features=None):
        assert(type(share_params) == bool)
        assert(type(multi_label) == bool)
        self.inference_method = 'customized'
        self.inference_train = inference_train
        self.inference_pred = inference_pred
        self.class_weight = None
        self.inference_calls = 0
        self.n_states = n_states
        self.n_features = n_features
        self.n_edge_features = n_edge_features
        self.share_params = share_params
        self.multi_label = multi_label
        self._set_size_joint_feature()
        self._set_class_weight()

    def _set_size_joint_feature(self):
        if None not in [self.n_states, self.n_features, self.n_edge_features]:
            if self.share_params is True:  # share params among POIs/transitions
                self.size_joint_feature = self.n_features + self.n_edge_features
            else:
                self.size_joint_feature = self.n_states * self.n_features + \
                    self.n_states * self.n_states * self.n_edge_features

    def loss(self, y, y_hat):
        # return np.mean(np.asarray(y) != np.asarray(y_hat))     # hamming loss (normalised)
        return np.sum(np.asarray(y) != np.asarray(y_hat))     # hamming loss

    def initialize(self, X, Y):
        assert(len(X) == len(Y))
        n_features = X[0][0].shape[1]
        if self.n_features is None:
            self.n_features = n_features
        else:
            assert(self.n_features == n_features)

        n_states = len(np.unique(np.hstack([y.ravel() for y in Y])))
        if self.n_states is None:
            self.n_states = n_states
        else:
            assert(self.n_states == n_states)

        n_edge_features = X[0][1].shape[2]
        if self.n_edge_features is None:
            self.n_edge_features = n_edge_features
        else:
            assert(self.n_edge_features == n_edge_features)

        self._set_size_joint_feature()
        self._set_class_weight()

        self.traj_group_dict = dict()
        for i in range(len(X)):
            query = X[i][2]
            if query in self.traj_group_dict:
                # NO duplication
                if not np.any([np.all(np.asarray(Y[i]) == np.asarray(yj)) for yj in self.traj_group_dict[query]]):
                    self.traj_group_dict[query].append(Y[i])
            else:
                self.traj_group_dict[query] = [Y[i]]

    def __repr__(self):
        return ("%s(n_states: %d, inference_method: %s, n_features: %d, n_edge_features: %d)"
                % (type(self).__name__, self.n_states, self.inference_method, self.n_features, self.n_edge_features))

    def joint_feature(self, x, y):
        assert(not isinstance(y, tuple))
        unary_features = x[0]  # unary features of all POIs: n_POIs x n_features
        pw_features = x[1]     # pairwise features of all transitions: n_POIs x n_POIs x n_edge_features
        query = x[2]           # query = (startPOI, length)
        n_nodes = query[1]

        # assert(unary_features.ndim == 2)
        # assert(pw_features.ndim == 3)
        # assert(len(query) == 3)
        assert(n_nodes == len(y))
        assert(unary_features.shape == (self.n_states, self.n_features))
        assert(pw_features.shape == (self.n_states, self.n_states, self.n_edge_features))

        if self.share_params is True:
            node_features = np.zeros((self.n_features), dtype=np.float)
            edge_features = np.zeros((self.n_edge_features), dtype=np.float)
            node_features = unary_features[y[0], :]
            for j in range(len(y) - 1):
                ss, tt = y[j], y[j + 1]
                node_features = node_features + unary_features[tt, :]
                edge_features = edge_features + pw_features[ss, tt, :]
        else:
            node_features = np.zeros((self.n_states, self.n_features), dtype=np.float)
            edge_features = np.zeros((self.n_states, self.n_states, self.n_edge_features), dtype=np.float)
            node_features[y[0], :] = unary_features[y[0], :]
            for j in range(len(y) - 1):
                ss, tt = y[j], y[j + 1]
                node_features[tt, :] = unary_features[tt, :]
                edge_features[ss, tt, :] = pw_features[ss, tt, :]

        joint_feature_vector = np.hstack([node_features.ravel(), edge_features.ravel()])
        return joint_feature_vector

    def loss_augmented_inference(self, x, y, w, relaxed=None):
        # inference procedure for training: (x, y) from training set (with features already scaled)
        #
        # argmax_y_hat np.dot(w, joint_feature(x, y_hat)) + loss(y, y_hat)
        #
        # the loss function should be decomposible in order to use Viterbi decoding, here we use Hamming loss
        #
        # x[0]: (unscaled) unary features of all POIs: n_POIs x n_features
        # x[1]: (unscaled) pairwise features of all transitions: n_POIs x n_POIs x n_edge_features
        # x[2]: query = (startPOI, length)
        unary_features = x[0]
        pw_features = x[1]
        query = x[2]

        assert(unary_features.ndim == 2)
        assert(pw_features.ndim == 3)
        assert(len(query) == 2)

        ps = query[0]
        L = query[1]
        M = unary_features.shape[0]  # total number of POIs

        self._check_size_w(w)
        if self.share_params is True:
            unary_params = w[:self.n_features]
            pw_params = w[self.n_features:].reshape(self.n_edge_features)
            # duplicate params so that inference procedures work the same way no matter params are shared or not
            unary_params = np.tile(unary_params, (self.n_states, 1))
            pw_params = np.tile(pw_params, (self.n_states, self.n_states, 1))
        else:
            unary_params = w[:self.n_states * self.n_features].reshape((self.n_states, self.n_features))
            pw_params = w[self.n_states * self.n_features:].reshape(
                (self.n_states, self.n_states, self.n_edge_features))

        if self.multi_label is True:
            y_true_list = self.traj_group_dict[query]
        else:
            y_true_list = [y]

        y_hat = self.inference_train(ps, L, M, unary_params, pw_params, unary_features, pw_features,
                                     y_true=y, y_true_list=y_true_list)
        return y_hat

    def inference(self, x, w, relaxed=False, return_energy=False):
        # inference procedure for testing: x from test set (features needs to be scaled)
        #
        # argmax_y np.dot(w, joint_feature(x, y))
        #
        # x[0]: (unscaled) unary features of all POIs: n_POIs x n_features
        # x[1]: (unscaled) pairwise features of all transitions: n_POIs x n_POIs x n_edge_features
        # x[2]: query = (startPOI, length)
        unary_features = x[0]
        pw_features = x[1]
        query = x[2]

        assert(unary_features.ndim == 2)
        assert(pw_features.ndim == 3)
        assert(len(query) == 2)

        ps = query[0]
        L = query[1]
        M = unary_features.shape[0]  # total number of POIs

        self._check_size_w(w)
        if self.share_params is True:
            unary_params = w[:self.n_features]
            pw_params = w[self.n_features:].reshape(self.n_edge_features)
            # duplicate params so that inference procedures work the same way no matter params shared or not
            unary_params = np.tile(unary_params, (self.n_states, 1))
            pw_params = np.tile(pw_params, (self.n_states, self.n_states, 1))
        else:
            unary_params = w[:self.n_states * self.n_features].reshape((self.n_states, self.n_features))
            pw_params = w[self.n_states * self.n_features:].reshape(
                (self.n_states, self.n_states, self.n_edge_features))

        y_pred = self.inference_pred(ps, L, M, unary_params, pw_params, unary_features, pw_features)

        return y_pred


def calc_node_features(startPOI, nPOI, poi_list, poi_info, dat_obj):
    """
    Compute node features (singleton) for all POIs given query (startPOI, nPOI)
    """
    columns = FEATURES.copy()
    p0, trajLen = startPOI, nPOI
    assert(p0 in poi_info.index)

    # DEBUG: use uniform node features
    # nrows = len(poi_list)
    # ncols = len(columns) + len(dat_obj.POI_CAT_LIST) + len(dat_obj.POI_CLUSTER_LIST) - 2
    # return np.ones((nrows, ncols), dtype=np.float)
    # return np.zeros((nrows, ncols), dtype=np.float)

    df_ = pd.DataFrame(index=poi_list, columns=columns)

    for poi in poi_list:
        # lon, lat = poi_info.loc[poi, 'poiLon'], poi_info.loc[poi, 'poiLat']
        pop, nvisit = poi_info.loc[poi, 'popularity'], poi_info.loc[poi, 'nVisit']
        cat, cluster = poi_info.loc[poi, 'poiCat'], dat_obj.POI_CLUSTERS.loc[poi, 'clusterID']
        nphotos = poi_info.loc[poi, ['nPhotoTotal', 'nPhotoMean', 'nPhotoP10', 'nPhotoP50', 'nPhotoP90']].tolist()
        durations = poi_info.loc[poi, ['durationTotal', 'durationMean', 'durationP10', 'durationP50', 'durationP90']]\
            .tolist()
        idx = poi
        df_.set_value(idx, 'category', tuple((cat == np.array(dat_obj.POI_CAT_LIST)).astype(np.int) * 2 - 1))
        df_.set_value(idx, 'neighbourhood',
                      tuple((cluster == np.array(dat_obj.POI_CLUSTER_LIST)).astype(np.int) * 2 - 1))
        df_.loc[idx, 'popularity'] = LOG_SMALL if pop < 1 else np.log10(pop)
        df_.loc[idx, 'nVisit'] = LOG_SMALL if nvisit < 1 else np.log10(nvisit)
        df_.loc[idx, 'nPhotoTotal'] = LOG_SMALL if nphotos[0] < 1 else np.log10(nphotos[0])
        df_.loc[idx, 'nPhotoMean'] = LOG_SMALL if nphotos[1] < 1 else np.log10(nphotos[1])
        df_.loc[idx, 'nPhotoP10'] = LOG_SMALL if nphotos[2] < 1 else np.log10(nphotos[2])
        df_.loc[idx, 'nPhotoP50'] = LOG_SMALL if nphotos[3] < 1 else np.log10(nphotos[3])
        df_.loc[idx, 'nPhotoP90'] = LOG_SMALL if nphotos[4] < 1 else np.log10(nphotos[4])
        df_.loc[idx, 'durationTotal'] = LOG_SMALL if durations[0] < 1 else np.log10(durations[0])
        df_.loc[idx, 'durationMean'] = LOG_SMALL if durations[1] < 1 else np.log10(durations[1])
        df_.loc[idx, 'durationP10'] = LOG_SMALL if durations[2] < 1 else np.log10(durations[2])
        df_.loc[idx, 'durationP50'] = LOG_SMALL if durations[3] < 1 else np.log10(durations[3])
        df_.loc[idx, 'durationP90'] = LOG_SMALL if durations[4] < 1 else np.log10(durations[4])
        df_.loc[idx, 'trajLen'] = trajLen
        df_.loc[idx, 'sameCategory'] = 1 if cat == poi_info.loc[p0, 'poiCat'] else -1
        df_.loc[idx, 'sameNeighbourhood'] = 1 if cluster == dat_obj.POI_CLUSTERS.loc[p0, 'clusterID'] else -1
        df_.loc[idx, 'diffPopularity'] = pop - poi_info.loc[p0, 'popularity']
        df_.loc[idx, 'diffNVisit'] = nvisit - poi_info.loc[p0, 'nVisit']
        df_.loc[idx, 'diffNPhotoTotal'] = nphotos[0] - poi_info.loc[p0, 'nPhotoTotal']
        df_.loc[idx, 'diffNPhotoMean'] = nphotos[1] - poi_info.loc[p0, 'nPhotoMean']
        df_.loc[idx, 'diffNPhotoP10'] = nphotos[2] - poi_info.loc[p0, 'nPhotoP10']
        df_.loc[idx, 'diffNPhotoP50'] = nphotos[3] - poi_info.loc[p0, 'nPhotoP50']
        df_.loc[idx, 'diffNPhotoP90'] = nphotos[4] - poi_info.loc[p0, 'nPhotoP90']
        df_.loc[idx, 'diffDurationTotal'] = durations[0] - poi_info.loc[p0, 'durationTotal']
        df_.loc[idx, 'diffDurationMean'] = durations[1] - poi_info.loc[p0, 'durationMean']
        df_.loc[idx, 'diffDurationP10'] = durations[2] - poi_info.loc[p0, 'durationP10']
        df_.loc[idx, 'diffDurationP50'] = durations[3] - poi_info.loc[p0, 'durationP50']
        df_.loc[idx, 'diffDurationP90'] = durations[4] - poi_info.loc[p0, 'durationP90']
        df_.loc[idx, 'distance'] = dat_obj.POI_DISTMAT.loc[poi, p0]

    # features other than category and neighbourhood
    # X = df_[sorted(set(df_.columns) - {'category', 'neighbourhood'})].values
    X = df_[FEATURES[2:]].values

    # boolean features: category (+1, -1)
    cat_features = np.vstack([list(df_.loc[x, 'category']) for x in df_.index])

    # boolean features: neighbourhood (+1, -1)
    neigh_features = np.vstack([list(df_.loc[x, 'neighbourhood']) for x in df_.index])

    return np.hstack([cat_features, neigh_features, X]).astype(np.float)


def calc_edge_features(trajid_list, poi_list, poi_info, dat_obj, log_transition=LOG_TRANSITION):
    """
    Compute edge features (transiton / pairwise)
    """
    feature_names = ['poiCat', 'popularity', 'nVisit', 'durationMean', 'clusterID']
    n_features = len(feature_names)

    # DEBUG: use uniform edge features
    # return np.ones((len(poi_list), len(poi_list), n_features), dtype=np.float)
    # return np.zeros((len(poi_list), len(poi_list), n_features), dtype=np.float)

    transmat_cat = dat_obj.gen_transmat_cat(trajid_list, poi_info)
    transmat_pop = dat_obj.gen_transmat_pop(trajid_list, poi_info)
    transmat_visit = dat_obj.gen_transmat_visit(trajid_list, poi_info)
    transmat_duration = dat_obj.gen_transmat_duration(trajid_list, poi_info)
    transmat_neighbor = dat_obj.gen_transmat_neighbor(trajid_list, poi_info)

    poi_features = pd.DataFrame(data=np.zeros((len(poi_list), len(feature_names))),
                                columns=feature_names, index=poi_list)
    poi_features.index.name = 'poiID'
    poi_features['poiCat'] = poi_info.loc[poi_list, 'poiCat']
    poi_features['popularity'] = np.digitize(poi_info.loc[poi_list, 'popularity'], dat_obj.LOGBINS_POP)
    poi_features['nVisit'] = np.digitize(poi_info.loc[poi_list, 'nVisit'], dat_obj.LOGBINS_VISIT)
    poi_features['durationMean'] = np.digitize(poi_info.loc[poi_list, 'durationMean'], dat_obj.LOGBINS_DURATION)
    poi_features['clusterID'] = dat_obj.POI_CLUSTERS.loc[poi_list, 'clusterID']

    edge_features = np.zeros((len(poi_list), len(poi_list), n_features), dtype=np.float64)

    for j in range(len(poi_list)):  # NOTE: POI order
        pj = poi_list[j]
        cat, pop = poi_features.loc[pj, 'poiCat'], poi_features.loc[pj, 'popularity']
        visit, cluster = poi_features.loc[pj, 'nVisit'], poi_features.loc[pj, 'clusterID']
        duration = poi_features.loc[pj, 'durationMean']

        for k in range(len(poi_list)):  # NOTE: POI order
            pk = poi_list[k]
            edge_features[j, k, :] = np.array([
                                              transmat_cat.loc[cat, poi_features.loc[pk, 'poiCat']],
                                              transmat_pop.loc[pop, poi_features.loc[pk, 'popularity']],
                                              transmat_visit.loc[visit, poi_features.loc[pk, 'nVisit']],
                                              transmat_duration.loc[duration, poi_features.loc[pk, 'durationMean']],
                                              transmat_neighbor.loc[cluster, poi_features.loc[pk, 'clusterID']]])

    if log_transition is True:
        return np.log10(edge_features)
    else:
        return edge_features
