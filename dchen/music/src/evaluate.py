import sys
import numpy as np
# from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from joblib import Parallel, delayed
from scipy.sparse import issparse


def evaluate_minibatch(clf, eval_func, X_test, Y_test, threshold=None, batch_size=100, verbose=0):
    assert X_test.shape[0] == Y_test.shape[0]

    N = X_test.shape[0]
    metrics_all = []
    n_batches = int((N-1) / batch_size) + 1
    indices = np.arange(N)

    for nb in range(n_batches):
        if verbose > 0:
            sys.stdout.write('\r %d / %d' % (nb+1, n_batches))
            sys.stdout.flush()

        ix_start = nb * batch_size
        ix_end = min((nb+1) * batch_size, N)
        ix = indices[ix_start:ix_end]

        X = X_test[ix]
        Y_true = Y_test[ix].astype(np.bool)
        #if issparse(Y_true):
        #    Y_true = Y_true.toarray()
        Y_pred = clf.decision_function(X)
        #if issparse(Y_pred):
        #    Y_pred = Y_pred.toarray()
        if threshold is not None:
            Y_pred = Y_pred >= threshold

        metrics = eval_func(Y_true, Y_pred)
        metrics_all = np.concatenate((metrics_all, metrics), axis=-1)

    return metrics_all


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
    OneK = np.ones(K)

    # n_true = np.dot(Y_true, OneK)
    n_true = np.sum(Y_true, axis=1)
    # n_positive = np.dot(Y_pred, OneK)
    n_positive = np.sum(Y_pred, axis=1)
    # true_positive = np.dot(np.multiply(Y_true, Y_pred), OneK)
    # true_positive = np.sum(np.multiply(Y_true, Y_pred), axis=1)
    true_positive = np.sum(np.logical_and(Y_true, Y_pred), axis=1)

    numerator = 2 * true_positive
    denominator = n_true + n_positive
    nonzero_ix = np.nonzero(denominator)[0]

    f1 = np.zeros(N)
    f1[nonzero_ix] = np.divide(numerator[nonzero_ix], denominator[nonzero_ix])

    return f1


def calc_precisionK(Y_true, Y_pred):
    """
    Compute Precision@K, one score for each example.
    - thresholding predictions using the K-th largest predicted score, K is #positives in ground truth
    - Precision@K: #true_positives / #positives_in_ground_truth
      where by the definition of Precision@K, #positives_in_ground_truth = #positive_in_prediction
    """
    assert Y_true.shape == Y_pred.shape
    assert Y_true.dtype == np.bool
    N, K = Y_true.shape
    OneK = np.ones(K)
    #KPosAll = np.dot(Y_true, OneK).astype(np.int)
    KPosAll = np.sum(Y_true, axis=1).astype(np.int)
    assert np.all(KPosAll > 0)

    rows = np.arange(N)
    sortedIx = np.argsort(-Y_pred, axis=1)
    cols = sortedIx[rows, KPosAll-1]  # index of thresholds (the K-th largest scores, NOTE index starts at 0)
    thresholds = Y_pred[rows, cols]   # the K-th largest scores
    Y_pred_bin = Y_pred >= thresholds[:, None]  # convert scores to binary predictions

    #true_positives = np.multiply(Y_true, Y_pred_bin)
    true_positives = np.logical_and(Y_true, Y_pred_bin)
    #return np.dot(true_positives, OneK) / KPosAll
    return np.sum(true_positives, axis=1) / KPosAll


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

        # sorted indices of the labels most likely to be +'ve
        idx = np.argsort(pred)[::-1]

        # true labels according to the sorted order
        y = truth[idx]

        # fraction of +'ves in the top K predictions
        return np.mean(y[:k]) if nPos > 0 else 0

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

    elif metricType == 'Precision@K':
        # sorted indices of the labels most likely to be +'ve
        idx = np.argsort(pred)[::-1]

        # true labels according to the sorted order
        y = truth[idx]

        # fraction of +'ves in the top K predictions
        return np.mean(y[:nPos]) if nPos > 0 else 0

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


def avgPrecisionK(allTruths, allPreds):
    losses = []
    metricType = 'Precision@K'
    for i in range(allPreds.shape[0]):
        pred = allPreds[i, :]
        truth = allTruths[i, :]
        losses.append(evalPred(truth, pred, metricType))
    return np.mean(losses)


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


def evaluatePrecision(allTruths, allPreds, verbose=0, n_jobs=-1):
    N = allTruths.shape[0]
    perf_dict = dict()
    for metricType in [('Precision@3', 3), ('Precision@5', 5), ('Precision@10', 10), 'Precision@K']:
        losses = Parallel(n_jobs=n_jobs)(delayed(evalPred)(allTruths[i, :], allPreds[i, :], metricType)
                                         for i in range(N))
        # losses = []
        # for i in range(allPreds.shape[0]):
        #    pred = allPreds[i, :]
        #    truth = allTruths[i, :]
        #    losses.append(evalPred(truth, pred, metricType))

        metricStr = metricType[0] if type(metricType) == tuple else metricType
        mean = np.mean(losses)
        stderr = np.std(losses) / np.sqrt(N)
        perf_dict[metricStr] = (mean, stderr)
        if verbose > 0:
            print('%s: %.4f, %.3f' % ('Average %s' % metricStr, mean, stderr))
    return perf_dict


def evaluateF1(allTruths, allPreds):
    N = allTruths.shape[0]
    f1 = []
    for i in range(allPreds.shape[0]):
        pred = allPreds[i, :]
        truth = allTruths[i, :]
        f1.append(f1_score_nowarn(truth, pred))
    mean = np.mean(f1)
    stderr = np.std(f1) / np.sqrt(N)
    print('%s: %.4f, %.3f' % ('Average F1', mean, stderr))
    return {'F1': (mean, stderr)}


def evaluateRankingLoss(allTruths, allPreds, n_jobs=-1):
    N = allTruths.shape[0]
    losses = Parallel(n_jobs=n_jobs)(delayed(evalPred)(allTruths[i, :], allPreds[i, :], metricType='Ranking')
                                     for i in range(N))
    # losses = []
    # for i in range(N):
    #    pred = allPreds[i, :]
    #    truth = allTruths[i, :]
    #    losses.append(evalPred(truth=truth, pred=pred, metricType='Ranking'))
    mean = np.mean(losses)
    stderr = np.std(losses) / np.sqrt(N)
    print('%s: %.4f, %.3f' % ('Average RankingLoss', mean, stderr))
    return {'RankingLoss': (mean, stderr)}


def printEvaluation(allTruths, allPreds):
    N = allTruths.shape[0]
    # print(N)

    for metricType in [('Precision@3', 3), ('Precision@5', 5), 'Precision@K']:
        # ['Subset01', 'Hamming', 'Ranking', 'Precision@K', 'Precision@3', 'Precision@5']:
        losses = []
        for i in range(allPreds.shape[0]):
            pred = allPreds[i, :]
            truth = allTruths[i, :]
            losses.append(evalPred(truth, pred, metricType))

        # print('%24s: %1.4f' % ('Average %s Loss' % metricType, np.mean(losses)))
        metricStr = metricType[0] if type(metricType) == tuple else metricType
        print('%s: %.4f, %.3f' % ('Average %s' % metricStr, np.mean(losses), np.std(losses) / np.sqrt(N)))
        # plt.hist(aucs, bins = 10);
