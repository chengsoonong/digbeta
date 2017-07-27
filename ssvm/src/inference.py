import sys
import numpy as np
import pandas as pd
import heapq as hq
import itertools
import pulp


class HeapTerm:  # an item in heapq (min-heap)
    def __init__(self, priority, task):
        self.priority = priority
        self.task = task
        self.string = str(priority) + ': ' + str(task)

    def __lt__(self, other):
        return self.priority < other.priority

    def __repr__(self):
        return self.string

    def __str__(self):
        return self.string


def do_inference_brute_force(ps, L, M, unary_params, pw_params, unary_features, pw_features,
                             y_true=None, y_true_list=None, debug=False, top=5):
    """
    Inference using brute force search (for sanity check), could be:
    - Train/prediction inference for single-label SSVM
    - Train/prediction inference for multi-label SSVM
    """
    assert(L > 1)
    assert(L <= M)
    assert(ps >= 0)
    assert(ps < M)
    assert(top > 0)
    if y_true is not None:
        assert(y_true_list is not None and type(y_true_list) == list)
    if y_true is not None:
        top = 1

    Cu = np.zeros(M, dtype=np.float)       # unary_param[p] x unary_features[p]
    Cp = np.zeros((M, M), dtype=np.float)  # pw_param[pi, pj] x pw_features[pi, pj]
    # a intermediate POI should NOT be the start POI, NO self-loops
    for pi in range(M):
        Cu[pi] = np.dot(unary_params[pi, :], unary_features[pi, :])   # if pi != ps else -np.inf
        for pj in range(M):
            Cp[pi, pj] = -np.inf if (pj == ps or pi == pj) else np.dot(pw_params[pi, pj, :], pw_features[pi, pj, :])

    Q = []
    for x in itertools.permutations([p for p in range(M) if p != ps], int(L - 1)):
        y = [ps] + list(x)
        score = 0

        if y_true is not None and np.any([np.all(np.array(y) == np.asarray(yj)) for yj in y_true_list]) is True:
            continue

        for j in range(1, L):
            score += Cp[y[j - 1], y[j]] + Cu[y[j]]
        if y_true is not None:
            score += np.sum(np.asarray(y) != np.asarray(y_true))

        if len(Q) < top:
            hq.heappush(Q, HeapTerm(score, np.array(y)))
        else:
            hq.heappushpop(Q, HeapTerm(score, np.array(y)))  # pop the smallest, then push

    results = []
    scores = []
    while len(Q) > 0:
        hterm = hq.heappop(Q)
        results.append(hterm.task)
        scores.append(hterm.priority)

    # reverse the order: smallest -> largest => largest -> smallest
    results.reverse()
    scores.reverse()

    if debug is True:
        for score, y in zip(scores, results):
            print(score, y)

    if y_true is not None:
        results = results[0]

    return results


def do_inference_greedy(ps, L, M, unary_params, pw_params, unary_features, pw_features, y_true=None, y_true_list=None):
    """
    Inference using greedy search (baseline), could be:
    - Train/prediction inference for single-label SSVM
    - Prediction inference for multi-label SSVM, no guaranteed for training
    """
    assert(L > 1)
    assert(L <= M)
    assert(ps >= 0)
    assert(ps < M)
    if y_true is not None:
        assert(y_true_list is not None and type(y_true_list) == list)

    Cu = np.zeros(M, dtype=np.float)       # unary_param[p] x unary_features[p]
    Cp = np.zeros((M, M), dtype=np.float)  # pw_param[pi, pj] x pw_features[pi, pj]
    # a intermediate POI should NOT be the start POI, NO self-loops
    for pi in range(M):
        Cu[pi] = np.dot(unary_params[pi, :], unary_features[pi, :])  # if pi != ps else -np.inf
        for pj in range(M):
            Cp[pi, pj] = -np.inf if (pj == ps or pi == pj) else np.dot(pw_params[pi, pj, :], pw_features[pi, pj, :])

    y_hat = [ps]
    for t in range(1, L):
        candidate_points = [p for p in range(M) if p not in y_hat]
        p = y_hat[-1]
        # maxix = np.argmax([Cp[p, p1] + Cu[p1] + float(p1 != y_true[t]) if y_true is not None else \
        #                    Cp[p, p1] + Cu[p1] for p1 in candidate_points])
        scores = [Cp[p, p1] + Cu[p1] + float(p1 != y_true[t]) if y_true is not None
                  else Cp[p, p1] + Cu[p1] for p1 in candidate_points]
        indices = list(np.argsort(-np.asarray(scores)))
        if t < L - 1 or y_true is None:
            y_hat.append(candidate_points[indices[0]])
        else:
            for j in range(len(candidate_points)):
                y = y_hat + [candidate_points[indices[j]]]
                if not np.any([np.all(np.asarray(y) == np.asarray(yj)) for yj in y_true_list]):
                    y_hat.append(candidate_points[indices[j]])
                    break
            if len(y_hat) < L:
                sys.stderr.write('Greedy inference EQUALS (one of) ground truth, return ground truth\n')
                y_hat.append(candidate_points[indices[-1]])

    return [np.asarray(y_hat)]


def do_inference_viterbi_brute_force(ps, L, M, unary_params, pw_params, unary_features, pw_features):
    """
    Heuristic to skip repeated POIs in predictions by Viterbi
    """
    y_hat = do_inference_viterbi(ps, L, M, unary_params, pw_params, unary_features, pw_features)
    pois = set(y_hat[0][1:])

    Cu = np.zeros(M, dtype=np.float)       # unary_param[p] x unary_features[p]
    Cp = np.zeros((M, M), dtype=np.float)  # pw_param[pi, pj] x pw_features[pi, pj]
    # a intermediate POI should NOT be the start POI, NO self-loops
    for pi in range(M):
        Cu[pi] = np.dot(unary_params[pi, :], unary_features[pi, :])   # if pi != ps else -np.inf
        for pj in range(M):
            Cp[pi, pj] = -np.inf if (pj == ps or pi == pj) else np.dot(pw_params[pi, pj, :], pw_features[pi, pj, :])

    y_best = None
    best_score = -np.inf
    for x in itertools.permutations(sorted(pois), len(pois)):
        y = [ps] + list(x)
        score = 0
        for j in range(1, len(y)):
            score += Cp[y[j - 1], y[j]] + Cu[y[j]]
        if best_score < score:
            best_score = score
            y_best = y

    assert(y_best is not None)
    return [np.asarray(y_best)]


def do_inference_heuristic(ps, L, M, unary_params, pw_params, unary_features, pw_features):
    """
    Heuristic to skip repeated POIs in predictions by Viterbi
    """
    result = []
    y_hat = do_inference_viterbi(ps, L, M, unary_params, pw_params, unary_features, pw_features)
    for p in y_hat[0]:
        if p not in result:
            result.append(p)
    return [np.asarray(result)]


def do_inference_viterbi(ps, L, M, unary_params, pw_params, unary_features, pw_features, y_true=None, y_true_list=None):
    """
    Inference using the Viterbi algorithm, could be:
    - Train/prediction inference for single-label SSVM
    - Prediction inference for multi-label SSVM
    """
    assert(L > 1)
    assert(L <= M)
    assert(ps >= 0)
    assert(ps < M)
    if y_true is not None:
        assert(y_true_list is not None and type(y_true_list) == list)
        assert(len(y_true_list) == 1)

    Cu = np.zeros(M, dtype=np.float)       # unary_param[p] x unary_features[p]
    Cp = np.zeros((M, M), dtype=np.float)  # pw_param[pi, pj] x pw_features[pi, pj]
    # a intermediate POI should NOT be the start POI, NO self-loops
    for pi in range(M):
        Cu[pi] = np.dot(unary_params[pi, :], unary_features[pi, :])  # if pi != ps else -np.inf
        for pj in range(M):
            Cp[pi, pj] = -np.inf if (pj == ps or pi == pj) else np.dot(pw_params[pi, pj, :], pw_features[pi, pj, :])

    A = np.zeros((L - 1, M), dtype=np.float)      # scores matrix
    B = np.ones((L - 1, M), dtype=np.int) * (-1)  # backtracking pointers

    for p in range(M):  # ps--p
        A[0, p] = Cp[ps, p] + Cu[p]
        # if y_true is not None and p != ps: A[0, p] += float(p != y_true[1])/L  # loss term: normalised
        if y_true is not None and p != ps:
            A[0, p] += float(p != y_true[1])
        B[0, p] = ps

    for t in range(0, L - 2):
        for p in range(M):
            # loss = float(p != y_true[l+2])/L if y_true is not None else 0  # loss term: normlised
            loss = float(p != y_true[t + 2]) if y_true is not None else 0
            scores = [A[t, p1] + Cp[p1, p] + Cu[p] for p1 in range(M)]  # ps~~p1--p
            maxix = np.argmax(scores)
            A[t + 1, p] = scores[maxix] + loss
            # B[l+1, p] = np.array(range(N))[maxix]
            B[t + 1, p] = maxix

    y_hat = [np.argmax(A[L - 2, :])]
    p, t = y_hat[-1], L - 2
    while t >= 0:
        y_hat.append(B[t, p])
        p, t = y_hat[-1], t - 1
    y_hat.reverse()

    return [np.asarray(y_hat)]


def do_inference_ILP_topk(ps, L, M, unary_params, pw_params, unary_features, pw_features, top=10, DIVERSITY=True):
    if DIVERSITY is True:
        results = []
        good_results = []
        while top > 0:
            predicted = results if len(results) > 0 else None
            y_hat = do_inference_ILP(ps, L, M, unary_params, pw_params, unary_features, pw_features,
                                     predicted_list=predicted)
            results.append(y_hat[0])
            if len(good_results) == 0 or len(set(y_hat[0]) - set(good_results[-1])) > 0:
                good_results.append(y_hat[0])
                top -= 1
        return good_results
    else:
        results = []
        for k in range(top):
            predicted = results if len(results) > 0 else None
            y_hat = do_inference_ILP(ps, L, M, unary_params, pw_params, unary_features, pw_features,
                                     predicted_list=predicted)
            results.append(y_hat[0])
        return results


def do_inference_ILP(ps, L, M, unary_params, pw_params, unary_features, pw_features, y_true=None,
                     y_true_list=None, predicted_list=None, n_threads=4, USE_GUROBI=True):
    """
    Inference using integer linear programming (ILP), could be:
    - Train/prediction inference for single-label SSVM (NOTE: NOT Hamming loss)
    - Prediction inference for multi-label SSVM
    """
    assert(L > 1)
    assert(L <= M)
    assert(ps >= 0)
    assert(ps < M)
    if y_true is not None:
        assert(y_true_list is not None and type(y_true_list) == list)
        assert(len(y_true_list) == 1)
        assert(predicted_list is None)
    if predicted_list is not None:
        assert(y_true is None and y_true_list is None)

    # when the parameters are very small, GUROBI will suffer from precision problems
    # scaling parameters
    unary_params = 1e6 * unary_params
    pw_params = 1e6 * pw_params

    p0 = str(ps)
    pois = [str(p) for p in range(M)]  # create a string list for each POI
    pb = pulp.LpProblem('Inference_ILP', pulp.LpMaximize)  # create problem
    # visit_i_j = 1 means POI i and j are visited in sequence
    visit_vars = pulp.LpVariable.dicts('visit', (pois, pois), 0, 1, pulp.LpInteger)
    # isend_l = 1 means POI l is the END POI of trajectory
    isend_vars = pulp.LpVariable.dicts('isend', pois, 0, 1, pulp.LpInteger)
    # a dictionary contains all dummy variables
    dummy_vars = pulp.LpVariable.dicts('u', [x for x in pois if x != p0], 2, M, pulp.LpInteger)

    # add objective
    objlist = []
    for pi in pois:      # from
        for pj in pois:  # to
            objlist.append(visit_vars[pi][pj] * (np.dot(unary_params[int(pj)], unary_features[int(pj)]) +
                                                 np.dot(pw_params[int(pi), int(pj)], pw_features[int(pi), int(pj)])))
    if y_true is not None:  # Loss: normalised number of mispredicted POIs, Hamming loss is non-linear of 'visit'
        objlist.append(1)
        for j in range(M):
            pj = pois[j]
            for k in range(1, L):
                pk = str(y_true[k])
                # objlist.append(-1.0 * visit_vars[pj][pk] / L) # loss term: normalised
                objlist.append(-1.0 * visit_vars[pj][pk])
    pb += pulp.lpSum(objlist), 'Objective'

    # add constraints, each constraint should be in ONE line
    pb += pulp.lpSum([visit_vars[pi][pi] for pi in pois]) == 0, 'NoSelfLoops'
    pb += pulp.lpSum([visit_vars[p0][pj] for pj in pois]) == 1, 'StartAt_p0'
    pb += pulp.lpSum([visit_vars[pi][p0] for pi in pois]) == 0, 'NoIncoming_p0'
    pb += pulp.lpSum([visit_vars[pi][pj] for pi in pois for pj in pois]) == L - 1, 'Length'
    pb += pulp.lpSum([isend_vars[pi] for pi in pois]) == 1, 'OneEnd'
    pb += isend_vars[p0] == 0, 'StartNotEnd'

    for pk in [x for x in pois if x != p0]:
        pb += pulp.lpSum([visit_vars[pi][pk] for pi in pois]) == isend_vars[pk] + \
            pulp.lpSum([visit_vars[pk][pj] for pj in pois if pj != p0]), 'ConnectedAt_' + pk
        pb += pulp.lpSum([visit_vars[pi][pk] for pi in pois]) <= 1, 'Enter_' + pk + '_AtMostOnce'
        pb += pulp.lpSum([visit_vars[pk][pj] for pj in pois if pj != p0]) + isend_vars[pk] <= 1, \
            'Leave_' + pk + '_AtMostOnce'
    for pi in [x for x in pois if x != p0]:
        for pj in [y for y in pois if y != p0]:
            pb += dummy_vars[pi] - dummy_vars[pj] + 1 <= (M - 1) * (1 - visit_vars[pi][pj]), \
                'SubTourElimination_' + pi + '_' + pj

    # additional constraints/cuts to filtering out specified sequences
    if predicted_list is not None:
        for j in range(len(predicted_list)):
            y = predicted_list[j]
            pb += pulp.lpSum([visit_vars[str(y[k])][str(y[k + 1])] for k in range(len(y) - 1)]) <= (len(y) - 2), \
                'exclude_%dth' % j

    pb.writeLP("traj_tmp.lp")

    # solve problem: solver should be available in PATH, default solver is CBC
    if USE_GUROBI is True:
        # gurobi_options = [('TimeLimit', '7200'), ('Threads', str(n_threads)), ('NodefileStart', '0.2'), ('Cuts', '2')]
        gurobi_options = [('TimeLimit', '10800'), ('Threads', str(n_threads)), ('NodefileStart', '0.5')]
        pb.solve(pulp.GUROBI_CMD(path='gurobi_cl', options=gurobi_options))  # GUROBI
    else:
        pb.solve(pulp.COIN_CMD(path='cbc', options=['-threads', str(n_threads), '-strategy', '1', '-maxIt', '2000000']))
    visit_mat = pd.DataFrame(data=np.zeros((len(pois), len(pois)), dtype=np.float), index=pois, columns=pois)
    isend_vec = pd.Series(data=np.zeros(len(pois), dtype=np.float), index=pois)
    for pi in pois:
        isend_vec.loc[pi] = isend_vars[pi].varValue
        for pj in pois:
            visit_mat.loc[pi, pj] = visit_vars[pi][pj].varValue

    # build the recommended trajectory
    recseq = [p0]
    while True:
        pi = recseq[-1]
        pj = visit_mat.loc[pi].idxmax()
        value = visit_mat.loc[pi, pj]
        assert(int(round(value)) == 1)
        recseq.append(pj)
        if len(recseq) == L:
            assert(int(round(isend_vec[pj])) == 1)
            return [np.asarray([int(x) for x in recseq])]
