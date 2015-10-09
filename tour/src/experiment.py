import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt
#from scipy import stats
from ijcai15 import PersTour


def enum_traj(ptobj, usr, seqofpoi):
    assert(isinstance(ptobj, PersTour))
    assert(usr in range(len(ptobj.usrmap)))
    assert(len(seqofpoi) in {3, 4, 5})
    pois = [x for x in range(len(ptobj.poimap))]
    p0 = seqofpoi[0]
    pN = seqofpoi[-1]

    # enumerate sequences with length 3
    if len(seqofpoi) == 3:
        return [[p0, p, pN] \
                for p in pois if p not in {p0, pN}]

    # enumerate sequences with length 4
    if len(seqofpoi) == 4:
        return [[p0, p1, p2, pN] \
                for p1 in pois if p1 not in {p0, pN} \
                for p2 in pois if p2 not in {p0, p1, pN}]

    # enumerate sequences with length 5
    if len(seqofpoi) == 5:
        return [[p0, p1, p2, p3, pN] \
                for p1 in pois if p1 not in {p0, pN} \
                for p2 in pois if p2 not in {p0, p1, pN} \
                for p3 in pois if p3 not in {p0, p1, p2, pN}]


def calc_scores(ptobj, usr, seqofpoi, transmat):  
    """Compute scores as features"""
    assert(isinstance(ptobj, PersTour))
    assert(usr in range(len(ptobj.usrmap)))
    assert(len(seqofpoi) > 2)
    assert(isinstance(transmat, np.ndarray))
    assert(transmat.shape == (len(ptobj.catmap), len(ptobj.catmap)))
    # User Interest: time-based/freq-based
    # POI Popularity
    # Travelling Cost
    # Trajectory log(probability) based on the transition matrix between different POI categories 
    # and the following rules for choosing a specific POI within certain category:
    #  - The Nearest Neighbor of the current POI
    #  - The most Popular POI
    #  - A random POI choosing with probability proportional to the reciprocal of its distance to current POI
    #  - A random POI choosing with probability proportional to its popularity
    cols = 8
    scores = np.zeros(cols, dtype=np.float64)
    for poi in seqofpoi:
        cat = ptobj.poicat[poi]
        scores[0] += ptobj.time_usr_interest[usr, cat] # score based on user interest: time-based interest
        scores[1] += ptobj.freq_usr_interest[usr, cat] # frequency based interest
        scores[2] += ptobj.poi_pop[poi]                # score based on POI popularity

    for i in range(len(seqofpoi)-1):
        p1 = seqofpoi[i]
        p2 = seqofpoi[i+1]
        scores[3] += -1*ptobj.traveltime[p1, p2] # score based on travelling cost

    EPS = 1e-9 # smooth, deal with 0 probability
    for i in range(len(seqofpoi)-1):
        p1 = seqofpoi[i]
        p2 = seqofpoi[i+1]
        cat1 = ptobj.poicat[p1]
        cat2 = ptobj.poicat[p2]
        #print('prob:', transmat[cat1, cat2])
        logprob = math.log10(transmat[cat1, cat2] + EPS) # log of transition probability

        poicat2 = [p2] # for convenience, put 'p2' as the first in candidate POI list with category 'cat2'
        for p, cat in ptobj.poicat.items():
            if p not in {p1, p2} and cat == cat2: # p1 and p2 could be within the same category (i.e. cat1 == cat2)
                poicat2.append(p)
        dvec = np.zeros(len(poicat2), dtype=np.float64) # reciprocal of distance p1-->p for p in poicat2
        pvec = np.zeros(len(poicat2), dtype=np.float64) # popularity of p
        speed = 4. # 4km/h
        for j in range(len(poicat2)):
            p = poicat2[j]
            #print('travel time', p1, '->', p, ':', ptobj.traveltime[p1, p])
            assert(ptobj.traveltime[p1, p] > 0.)
            dvec[j] = 60. * 60. / (ptobj.traveltime[p1, p] * speed) # travel time in seconds
            pvec[j] = ptobj.poi_pop[p]

        # score = log of transition probability + log of probability choose a specific POI given a rule
        if dvec.argmax() == 0: # POI 'p2' is the nearest neighbor
            scores[4] += logprob + math.log10(1. + EPS)
        else:
            scores[4] += logprob + math.log10(0. + EPS)

        if pvec.argmax() == 0: # POI 'p2' is the most popular within category 'cat2'
            scores[5] += logprob + math.log10(1. + EPS)
        else:
            scores[5] += logprob + math.log10(0. + EPS)

        scores[6] += math.log10(EPS + dvec[0] / np.sum(dvec))
        scores[7] += math.log10(EPS + pvec[0] / np.sum(pvec))

    return scores


def calc_traj_score(paramvec, scorevec):
    assert(paramvec.ndim == 1)
    if scorevec.ndim == 1:
        assert(len(paramvec) == len(scorevec))
    else:
        assert(len(paramvec) == len(scorevec[0]))

    # normalize scores
    #zscores = stats.zscore(scorevec)
    #normscores = zscore / abs(zscores).max()
    scorevec1 = scorevec / abs(scorevec).max() # normalize score, range [-1, 1]
    
    # normalize parameters ?, range[-1, 1]
    #paramvec /= np.sum(paramvec)

    return np.dot(paramvec, scorevec1)


def calc_traj_F1score(seqofpoi_act, seqofpoi_rec):
    assert(len(seqofpoi_act) > 0)
    assert(len(seqofpoi_rec) > 0)
    actset = set(seqofpoi_act)
    recset = set(seqofpoi_rec)
    intersect = actset & recset
    recall = len(intersect) / len(seqofpoi_act)
    precision = len(intersect) / len(seqofpoi_rec)
    f1score = 2. * precision * recall / (precision + recall)
    return f1score


def gen_train_test_set(ptobj):
    """For each user[i], leave one of his/her traj in testing set, and aggregate other traj into training set"""
    assert(isinstance(ptobj, PersTour))
    useqdict = dict()

    for seq in ptobj.sequences.keys():
        if len(ptobj.sequences[seq]) not in {3, 4, 5}: continue
        usr = ptobj.sequsr[seq]
        if usr not in useqdict: useqdict[usr] = []
        useqdict[usr].append(seq)

    seqtrainset = set()
    seqtestset = set()
    for usr, seqlist in useqdict.items():
        if len(seqlist) < 2: continue
        idx = random.sample(range(len(seqlist)), 1)[0] # leave one out
        #print(idx)
        seqtestset.add(seqlist[idx])
        seqtrainset = seqtrainset.union(set([seqlist[x] for x in range(len(seqlist)) if x != idx]))
        #seqtrainset = seqtrainset.union(set(seqlist[1:]))
        #seqtestset.add(seqlist[0])
    return seqtrainset, seqtestset


def save_train_test_set(fname, seqtrainset, seqtestset):
    with open(fname, 'w') as f:
        f.write(str(seqtrainset) + '\n')
        f.write(str(seqtestset) + '\n')


def load_train_test_set(fname):
    seqtrainset = set()
    seqtestset = set()
    with open(fname, 'r') as f:
        trainstr = f.readline().strip()
        teststr = f.readline().strip()
    for seq in trainstr[1:-1].split(','): # remove first character '{' and last character '}'
        seqtrainset.add(int(seq.strip()))
    for seq in teststr[1:-1].split(','):
        seqtestset.add(int(seq.strip()))
    return seqtrainset, seqtestset


def calc_mean_F1score(ptobj, trainset, enumseqs_dict, scorevec_dict, scoremat, paramvec):
    traj_scores = scoremat.dot(paramvec)
    totalF1 = 0.

    for i in range(len(trainset)):
        seq = trainset[i]
        usr = ptobj.sequsr[seq]
        seqofpoi = ptobj.sequences[seq]
        enumseqs = enumseqs_dict[seq]
        assert(len(enumseqs) > 0)
        scores = np.zeros(len(enumseqs), dtype=np.float64)
        for j in range(len(enumseqs)):
            scoreidx = scorevec_dict[seq][j]
            scores[j] = traj_scores[scoreidx]
        bestseq = enumseqs[ scores.argmax() ]
        F1score = calc_traj_F1score(seqofpoi, bestseq)
        #print('length:', len(seqofpoi), ', F1score:', F1score)
        totalF1 += F1score
    avgF1 = totalF1 / len(trainset)
    return avgF1
     

def main(ptobj, seqtrainset):
    assert(isinstance(ptobj, PersTour))
    assert(len(seqtrainset) > 0)

    transmat = ptobj.gen_transmat(seqtrainset)
    trainset = sorted(list(seqtrainset))
    enumseqs_dict = dict()
    scorevec_dict = dict()

    print('compute features...')

    # compute the enumerated sequences and the corresponding score vector
    for i in range(len(trainset)):
        seq = trainset[i]
        usr = ptobj.sequsr[seq]
        seqofpoi = ptobj.sequences[seq]
        enumseqs = enum_traj(ptobj, usr, seqofpoi)
        enumseqs_dict[seq] = enumseqs

    # the big score matrix for all enumerated sequences
    cols = 8
    rows = 0
    for seq in enumseqs_dict.keys():
        rows += len(enumseqs_dict[seq])
    assert(rows > 0)
    scoremat = np.zeros((rows, cols), dtype=np.float64)
    sidx = 0 # row index of the score vector in the big score matrix
    for i in range(len(trainset)):
        seq = trainset[i]
        assert(seq in enumseqs_dict)
        usr = ptobj.sequsr[seq]
        enumseqs = enumseqs_dict[seq]
        indeces = []
        for j in range(len(enumseqs)):
            idx = sidx + j
            indeces.append(idx)
            scoremat[idx] = calc_scores(ptobj, usr, enumseqs[j], transmat)
            scoremat[idx] /= abs(scoremat[idx]).max() # normalize score, range [-1, 1]
        scorevec_dict[seq] = indeces
        sidx += len(enumseqs) # for the next iteration

    K = 8 # number of features

    # sanity check
    #paramdata = [[1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8],
    #             [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    #             [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
    #             [0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    #             [0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    #             [0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1],
    #             [0.15, 0., 0.05, 0.25, 0.4, 0., 0., 0.9]]

    # run with random weights
    #N = 10000
    #params = np.zeros((N, K), dtype=np.float64)
    #values = np.zeros(N, dtype=np.float64)
    #for j in range(N):
    #    paramvec = np.random.uniform(-1, 1, K)
    #    params[j] = paramvec
    #    values[j] = calc_mean_F1score(ptobj, trainset, enumseqs_dict, scorevec_dict, scoremat, paramvec)
    #maxidx = values.argmax()
    #minidx = values.argmin()
    #print('best  F1-score:', values[maxidx], ', best  weights:', params[maxidx])
    #print('worst F1-score:', values[minidx], ', worst weights:', params[minidx])
    #np.savetxt('params.txt', params, delimiter=',')
    #np.savetxt('values.txt', values, delimiter=',')
    #plt.scatter(list(range(N)), values, marker='+')

    # run with single features
    weights = np.linspace(-1, 1, 41)
    randh = 0.667
    for k in range(K):
        print('run with the features[' + str(k) + '] only...')
        paramvec = np.zeros(K, dtype=np.float64)
        values = np.zeros(weights.shape[0], dtype=np.float64)
        for l in range(weights.shape[0]):
            paramvec[k] = weights[l]
            values[l] = calc_mean_F1score(ptobj, trainset, enumseqs_dict, scorevec_dict, scoremat, paramvec)
        maxidx = values.argmax()
        minidx = values.argmin()
        print('best  F1-score:', values[maxidx], ', best  weights[' + str(k) + ']:', weights[maxidx])
        print('worst F1-score:', values[minidx], ', worst weights[' + str(k) + ']:', weights[minidx])
        plt.scatter(weights, values, marker='+')
        plt.plot([-1, 1], [randh, randh], color='g', linestyle='--', label=str(randh))
        plt.xlim([-1, 1])
        plt.ylim([values.min()-0.01, 0.01+max(values.max(), 0.667)])
        plt.xlabel('Weight of Feature ' + str(k+1))
        plt.ylabel('F1-score')
        plt.legend()
        plt.show()
        input('Press any key to continue ...')



    # search for good weights


    #paramvec = np.array([-0.41149762, 0.83146491, 0.07336859, 0.84470228, 0.29542717, -0.8729996, \
    #                     -0.76215229,-0.54824898], dtype=np.float64) # generated from np.random.uniform(-1, 1, 8)
    #weights = np.linspace(-1, 1, 41)
    #print('initial weights:', paramvec)
    #print('F1-score:', calc_mean_F1score(ptobj, trainset, enumseqs_dict, scorevec_dict, scoremat, paramvec))
    
    #for l in range(paramvec.shape[0]):
    #    print('search for coordinate', l, '...')
    #    values = np.zeros(weights.shape[0], dtype=np.float64)
    #    for k in range(weights.shape[0]):
    #        paramvec[l] = weights[k]
    #        values[k] = calc_mean_F1score(ptobj, trainset, enumseqs_dict, scorevec_dict, scoremat, paramvec)
    #    midx = values.argmax()
    #    print('best F1-score:', values[midx], ', best weight:', weights[midx])
    #    paramvec[l] = weights[midx]
    #plt.scatter(np.linspace(-1, 1, 21), values, marker='+')

    plt.show()
    input('...')


if __name__ == '__main__':
    dirname = './Edinburgh'
    basefilename = 'userVisits-Edin.csv'
    ptobj = PersTour(dirname, basefilename)
    ptobj.calc_metrics({x for x in range(len(ptobj.sequences))})

    # train and test sequences
    #seqtrainset, seqtestset = gen_train_test_set(ptobj)
    #print(len(seqtrainset), len(seqtestset))

    fname = 'train_test.set'
    seqtrainset, seqtestset = load_train_test_set(fname)

    #lengths = []
    #for seq in seqtrainset:
    #    lengths.append(len(ptobj.sequences[seq]))
    #plt.hist(lengths, bins=20)
    #plt.show()
    #input('...')

    #fname = 'train_test.set1'
    #save_train_test_set(fname, seqtrainset, seqtestset)
 
    main(ptobj, seqtestset)
