import numpy as np
# from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from joblib import Parallel, delayed


def calc_metrics(y_true, y_pred, tops=[]):
    assert y_true.ndim == y_pred.ndim == 1
    assert len(y_true) == len(y_pred)
    npos = y_true.sum()
    assert npos > 0
    assert npos < len(y_true)
    rp, hitrates = calc_RPrecision_HitRate(y_true, y_pred, tops=tops)
    auc = roc_auc_score(y_true, y_pred)
    return rp, hitrates, auc


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def diversity(vec):
    """
    diversity(L) = ( \sum_{i \in L} \sum_{j \in L \setminus i} dist(i, j) ) / (|L| * (|L| - 1))
    """
    assert vec.ndim == 1
    norm = len(vec) * (len(vec) - 1)
    sim_mat = vec[..., np.newaxis] == vec[np.newaxis, ...]  # pairwise comparison
    # dist_mat = 1 - sim_mat
    # return (dist_mat.sum() - dist_mat.trace()) / norm  # note that dist_mat.trace() = 0
    return (1 - sim_mat).sum() / norm


def pairwise_distance_hamming(X, normalise=True):
    """
    A vectorised approach to compute Hamming distance between all pairs of rows of matrix X.

    Note that `p XOR q = ( p AND NOT q )  OR  ( NOT p AND q )` from
    [here](https://math.stackexchange.com/questions/38473/is-xor-a-combination-of-and-and-not-operators),
    let p, q \in \{0, 1\}^{N}, then

    Hamming_distance(p, q)
    = (1 / N) \sum_{i=1}^N p_i XOR q_i
    = (1 / N) \sum_{i=1}^N ( p_i (1 - q_i) + (1 - p_i) q_i )
    = (1 / N) ( \sum_{i=1}^N p_i (1 - q_i) + \sum_{i=1}^N (1 - p_i) q_i \right)
    = (1 / N) ( p^T (1 - q) + (1 - p)^T q )
    = (1 / N) ( \sum_{i=1}^N p_i + \sum_{i=1}^N q_i - 2 p^T q )

    Sanity check:
    ```
    N, D = 1000, 200
    aa = np.zeros(N * D, dtype=np.int)
    idx = np.random.permutation(N * D)[:int(N * D * .3)]
    aa[idx] = 1
    aa = aa.reshape(N, D)
    d1 = sklearn.metrics.pairwise.pairwise_distances(aa, metric='hamming', n_jobs=2)
    d2 = (np.dot(aa, 1-aa.T) + np.dot(1-aa, aa.T)) / D
    sum_vec = aa.sum(axis=1, keepdims=True)
    d3 = (sum_vec + sum_vec.T - 2 * np.dot(aa, aa.T)) / D
    diff = (d1 - d2).ravel();  print(np.dot(diff, diff))
    diff2 = (d1 - d3).ravel(); print(np.dot(diff2, diff2))
    ```
    """
    M, D = X.shape
    # X = X.astype(np.int)
    assert X.dtype == np.int
    norm = D if normalise is True else 1
    # sum_vec = X.sum(axis=1, keepdims=True)
    # dist = (sum_vec + sum_vec.T - 2 * np.dot(X, X.T)) / D
    # support sparse matrix
    sum_vec = X.sum(axis=1).reshape(M, 1)
    dist = (sum_vec + sum_vec.T - 2 * X.dot(X.T)) / norm
    return dist


def calc_Precision_Recall(y_true, y_pred, K=[]):
    """
        Compute Precision (Hit-Rate) and Recall given top-K recommendation.
    """
    tops = K
    assert y_true.ndim == y_pred.ndim == 1
    assert len(y_true) == len(y_pred)
    assert y_true.dtype == np.bool
    assert type(tops) == list
    assert len(tops) > 0
    sortix = np.argsort(-y_pred)
    npos = y_true.sum()
    assert npos > 0
    # y_ = y_true[sortix]
    pak = dict()
    rak = dict()
    for top in tops:
        assert 0 < top <= len(y_true)
        true_pos = np.sum(y_true[sortix[:top]])
        pak[top] = true_pos / npos
        rak[top] = true_pos / top
    return (pak, rak)


def calc_RPrecision_HitRate(y_true, y_pred, tops=[]):
    """
        Compute R-Precision and Hit-Rate at top-N.
    """
    assert y_true.ndim == y_pred.ndim == 1
    assert len(y_true) == len(y_pred)
    assert y_true.dtype == np.bool
    assert type(tops) == list
    sortix = np.argsort(-y_pred)
    npos = y_true.sum()
    assert npos > 0
    y_ = y_true[sortix]
    rp = np.mean(y_[:npos])
    if len(tops) == 0:
        return (rp, None)
    hitrates = dict()
    for top in tops:
        assert 0 < top <= len(y_true)
        hitrates[top] = np.sum(y_true[sortix[:top]]) / npos
    return (rp, hitrates)


def calc_F1(Y_true, Y_pred):
    """
    Compute F1 scores for multilabel prediction, one score for each example.
    precision = true_positive / n_true
    recall = true_positive / n_positive
    f1 = (2 * precision * recall) / (precision + recall) = 2 * true_positive / (n_true + n_positive)
    """
    assert Y_true.shape == Y_pred.shape
    assert Y_true.dtype == Y_pred.dtype == np.bool
    N, K = Y_true.shape
    n_true = np.sum(Y_true, axis=1)
    n_positive = np.sum(Y_pred, axis=1)
    true_positive = np.sum(np.logical_and(Y_true, Y_pred), axis=1)

    numerator = 2 * true_positive
    denominator = n_true + n_positive
    nonzero_ix = np.nonzero(denominator)[0]
    f1 = np.zeros(N)
    f1[nonzero_ix] = np.divide(numerator[nonzero_ix], denominator[nonzero_ix])
    return f1


def calc_RPrecision(Y_true, Y_pred, axis=0):
    """
    Compute RPrecision, one score for each (example if axis=0 else label)
    - thresholding predictions using the K-th largest predicted score, K is #positives in ground truth for an example
    - RPrecision: #true_positives / K
      where by the definition, K = #positives_in_ground_truth
    """
    assert Y_true.shape == Y_pred.shape
    assert Y_true.dtype == np.bool
    assert axis in [0, 1]
    N, K = Y_true.shape
    ax = 1 - axis
    num = (N, K)[axis]
    numPos = np.sum(Y_true, axis=ax).astype(np.int)
    sort_ix = np.argsort(-Y_pred, axis=ax)

    if axis == 0:
        rows = np.arange(N)
        cols = sort_ix[rows, numPos-1]   # index of thresholds (the K-th largest scores, NOTE index starts at 0)
        thresholds = Y_pred[rows, cols].reshape(N, 1)  # the K-th largest scores
        Y_pred_bin = Y_pred >= thresholds  # convert scores to binary predictions
    else:
        cols = np.arange(K)
        rows = sort_ix[numPos-1, cols]
        thresholds = Y_pred[rows, cols].reshape(1, K)
        Y_pred_bin = Y_pred >= thresholds

    nonzero_ix = np.nonzero(numPos)[0]
    true_positives = np.logical_and(Y_true, Y_pred_bin)
    rps = np.zeros(num)
    rps[nonzero_ix] = true_positives.sum(axis=ax)[nonzero_ix] / numPos[nonzero_ix]
    return rps[nonzero_ix], nonzero_ix


def calc_rank(x, largestFirst=True):
    """
        Compute the rank of numbers in an array.

        Input
        - x: a 1D array of numbers
        - largestFirst: boolean
          if True, the largest number has rank 1, the second largest has rank 2, ...
          if False, the smallest number has rank 1, the second smallest has rank 2, ...
    """
    assert x.ndim == 1
    n = len(x)
    assert n > 0
    sortix = np.argsort(-x)
    rank = np.zeros(n, dtype=np.int)

    # use a loop
    # for i, six in enumerate(sortix):
    #    rank[six] = i+1

    # without using loop
    rank[sortix] = np.arange(n) + 1

    if largestFirst is True:
        return rank
    else:
        return n + 1 - rank


def f1_score_nowarn(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None):
    """
        Compute F1 score, use the same interface as sklearn.metrics.f1_score,
        but disable the warning when both precision and recall are zeros.
    """
    _, _, f, _ = precision_recall_fscore_support(y_true, y_pred, beta=1, labels=labels, pos_label=pos_label,
                                                 average=average, warn_for=(), sample_weight=sample_weight)
    return f


def evalPred(truth, pred, metricType='Precision@K'):
    """
        Compute loss given ground truth and prediction

        Input:
            - truth:    binary array of true labels
            - pred:     real-valued array of predictions
            - metricType: can be subset 0-1, Hamming, ranking, and Precision@K where K = # positive labels.
    """
    truth = np.asarray(truth)
    pred = np.asarray(pred)
    assert(truth.shape[0] == pred.shape[0])
    L = truth.shape[0]
    nPos = np.sum(truth)
    assert float(nPos).is_integer()
    nPos = int(nPos)
    predBin = np.array((pred > 0), dtype=np.int)

    if type(metricType) == tuple:
        # Precision@k, k is constant
        k = metricType[1]
        assert k > 0
        assert k <= L
        assert nPos > 0

        # sorted indices of the labels most likely to be +'ve
        idx = np.argsort(pred)[::-1]

        # true labels according to the sorted order
        y = truth[idx]

        # fraction of +'ves in the top K predictions
        # return np.mean(y[:k]) if nPos > 0 else 0
        return np.mean(y[:k])
    elif metricType == 'Subset01':
        return 1 - int(np.all(truth == predBin))
    elif metricType == 'Hamming':
        return np.sum(truth != predBin) / L
    elif metricType == 'Ranking':
        loss = 0
        for i in range(L):
            for j in range(L):
                if truth[i] > truth[j]:
                    if pred[i] < pred[j]:
                        loss += 1
                    elif pred[i] == pred[j]:
                        loss += 0.5
                    else:
                        pass
        # denom = nPos * (L - nPos)
        # return loss / denom if denom > 0 else 0
        return loss
    elif metricType == 'TopPush':
        posInd = np.nonzero(truth)[0].tolist()
        negInd = sorted(set(np.arange(L)) - set(posInd))
        return np.mean(pred[posInd] <= np.max(pred[negInd]))
    elif metricType == 'BottomPush':
        posInd = np.nonzero(truth)[0].tolist()
        negInd = sorted(set(np.arange(L)) - set(posInd))
        return np.mean(pred[negInd] >= np.min(pred[posInd]))
    elif metricType == 'RPrecision':
        assert nPos > 0
        # sorted indices of the labels most likely to be +'ve
        idx = np.argsort(pred)[::-1]

        # true labels according to the sorted order
        y = truth[idx]

        # fraction of +'ves in the top K predictions
        return np.mean(y[:nPos])
    # elif metricType == 'Precision@3':
    #    # sorted indices of the labels most likely to be +'ve
    #    idx = np.argsort(pred)[::-1]
    #
    #    # true labels according to the sorted order
    #    y = truth[idx]
    #
    #    # fraction of +'ves in the top K predictions
    #    return np.mean(y[:3])
    #
    # elif metricType == 'Precision@5':
    #    # sorted indices of the labels most likely to be +'ve
    #    idx = np.argsort(pred)[::-1]
    #
    #    # true labels according to the sorted order
    #    y = truth[idx]
    #
    #    # fraction of +'ves in the top K predictions
    #    return np.mean(y[:5])
    else:
        assert(False)


def calcLoss(allTruths, allPreds, metricType, njobs=-1):
    N = allTruths.shape[0]
    losses = Parallel(n_jobs=njobs)(delayed(evalPred)(allTruths[i, :], allPreds[i, :], metricType)
                                    for i in range(N))
    return np.asarray(losses)


def avgPrecision(allTruths, allPreds, k):
    L = allTruths.shape[1]
    assert k <= L
    losses = []
    metricType = ('Precision', k)
    for i in range(allPreds.shape[0]):
        pred = allPreds[i, :]
        truth = allTruths[i, :]
        losses.append(evalPred(truth, pred, metricType))
    return np.mean(losses)
