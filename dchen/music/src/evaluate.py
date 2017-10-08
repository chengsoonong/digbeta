import numpy as np


def evalPred(truth, pred, lossType='Precision@K'):
    """
        Compute loss given ground truth and prediction

        Input:
            - truth:    binary array of true labels
            - pred:     real-valued array of predictions
            - lossType: can be subset 0-1, Hamming, ranking, and Precision@K where K = # positive labels.
    """

    assert(len(truth) == len(pred))
    L = len(truth)
    nPos = np.sum(truth)

    predBin = np.array((pred > 0), dtype=np.int)

    if type(lossType) == tuple:
        # Precision@k, k is constant
        k = lossType[1]
        assert k > 0
        assert k <= L

        # sorted indices of the labels most likely to be +'ve
        idx = np.argsort(pred)[::-1]

        # true labels according to the sorted order
        y = truth[idx]

        # fraction of +'ves in the top K predictions
        return np.mean(y[:k]) if nPos > 0 else 0

    elif lossType == 'Subset01':
        return 1 - int(np.all(truth == predBin))

    elif lossType == 'Hamming':
        return np.sum(truth != predBin) / L

    elif lossType == 'Ranking':
        loss = 0
        for i in range(L-1):
            for j in range(i+1, L):
                if truth[i] > truth[j]:
                    if pred[i] < pred[j]:
                        loss += 1
                    if pred[i] == pred[j]:
                        loss += 0.5
        # return loss / (nPos * (L-nPos))
        return loss

    elif lossType == 'Precision@K':
        # sorted indices of the labels most likely to be +'ve
        idx = np.argsort(pred)[::-1]

        # true labels according to the sorted order
        y = truth[idx]

        # fraction of +'ves in the top K predictions
        return np.mean(y[:nPos]) if nPos > 0 else 0

    # elif lossType == 'Precision@3':
    #    # sorted indices of the labels most likely to be +'ve
    #    idx = np.argsort(pred)[::-1]
    #
    #    # true labels according to the sorted order
    #    y = truth[idx]
    #
    #    # fraction of +'ves in the top K predictions
    #    return np.mean(y[:3])
    #
    # elif lossType == 'Precision@5':
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
    lossType = 'Precision@K'
    for i in range(allPreds.shape[0]):
        pred = allPreds[i, :]
        truth = allTruths[i, :]
        losses.append(evalPred(truth, pred, lossType))
    return np.mean(losses)


def avgPrecision(allTruths, allPreds, k):
    L = allTruths.shape[1]
    assert k <= L

    losses = []
    lossType = ('Precision', k)
    for i in range(allPreds.shape[0]):
        pred = allPreds[i, :]
        truth = allTruths[i, :]
        losses.append(evalPred(truth, pred, lossType))
    return np.mean(losses)


def printEvaluation(allTruths, allPreds):
    N = allTruths.shape[0]
    # print(N)

    for lossType in ['Precision@K']:
        # ['Subset01', 'Hamming', 'Ranking', 'Precision@K', 'Precision@3', 'Precision@5']:
        losses = []
        for i in range(allPreds.shape[0]):
            pred = allPreds[i, :]
            truth = allTruths[i, :]
            losses.append(evalPred(truth, pred, lossType))

        # print('%24s: %1.4f' % ('Average %s Loss' % lossType, np.mean(losses)))
        print('%s: %1.4f, %.3f' % ('Average %s' % lossType, np.mean(losses), np.std(losses) / np.sqrt(N)))
        # plt.hist(aucs, bins = 10);
