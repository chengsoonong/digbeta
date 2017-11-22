import numpy as np
# from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support


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


def evaluatePrecision(allTruths, allPreds, verbose=0):
    N = allTruths.shape[0]
    perf_dict = dict()
    for metricType in [('Precision@3', 3), ('Precision@5', 5), ('Precision@10', 10), 'Precision@K']:
        losses = []
        for i in range(allPreds.shape[0]):
            pred = allPreds[i, :]
            truth = allTruths[i, :]
            losses.append(evalPred(truth, pred, metricType))

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


def evaluateRankingLoss(allTruths, allPreds):
    N = allTruths.shape[0]
    loss = []
    for i in range(N):
        pred = allPreds[i, :]
        truth = allTruths[i, :]
        loss.append(evalPred(truth=truth, pred=pred, metricType='Ranking'))
    mean = np.mean(loss)
    stderr = np.std(loss) / np.sqrt(N)
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
