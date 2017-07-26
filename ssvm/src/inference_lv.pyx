import numpy as np
import heapq as hq
import sys
cimport numpy as np

"""
Inference using the List Viterbi algorithm, which sequentially find the (k+1)-th best path/walk given the 1st, 2nd, ..., k-th best paths/walks. 
Implementation is adapted from references:
- Sequentially finding the N-Best List in Hidden Markov Models, Dennis Nilsson and Jacob Goldberger, IJCAI 2001.
- A tutorial on hidden Markov models and selected applications in speech recognition, L.R. Rabiner, Proceedings of the IEEE, 1989.
"""

cdef class HeapItem:  # an item in heapq (min-heap)
    cdef readonly float priority
    cdef readonly object task, string
    
    def __init__(self, float priority, task):
        self.priority = priority
        self.task = task
        self.string = str(priority) + ': ' + str(task)
        
    #def __lt__(self, other):
    #    return self.priority < other.priority
    
    def __richcmp__(self, other, int op):
        if op == 2: # ==
            return self.priority == other.priority
        elif op == 3: # !=
            return self.priority != other.priority
        elif op == 0: # <
            return self.priority < other.priority
        elif op == 1: # <=
            return self.priority <= other.priority
        elif op == 4: # >
            return self.priority > other.priority
        elif op == 5: # >=
            return self.priority >= other.priority
        else:
            assert False
            
    def __repr__(self):
        return self.string
    
    def __str__(self):
        return self.string


cpdef do_inference_list_viterbi(int ps, int L, int M,
                                np.ndarray[dtype=np.float64_t, ndim=2] unary_params, 
                                np.ndarray[dtype=np.float64_t, ndim=3] pw_params, 
                                np.ndarray[dtype=np.float64_t, ndim=2] unary_features, 
                                np.ndarray[dtype=np.float64_t, ndim=3] pw_features, 
                                y_true=None, y_true_list=None, int top=10, path4train=False, DIVERSITY=True):
    """ 
    Inference using the list Viterbi algorithm, could be:
    - Train/prediction inference for single-label SSVM
    - Train/prediction inference for multi-label SSVM
    """
    assert(L > 1)
    assert(M >= L)
    assert(ps >= 0)
    assert(ps < M)
    assert(top > 0)
    if y_true is not None: assert(y_true_list is not None and type(y_true_list) == list)

    # scaling parameters as it is too small
    unary_params = 1e6 * unary_params
    pw_params = 1e6 * pw_params
    
    cdef int pi, pj, t, pk, parix, partition_index, partition_index_start, k_partition_index
    cdef long k, nIter, maxIter = long(5e6)
    cdef float loss, priority, new_priority
    
    Cu = np.zeros(M, dtype=np.float)      # unary_param[p] x unary_features[p]
    Cp = np.zeros((M, M), dtype=np.float) # pw_param[pi, pj] x pw_features[pi, pj]
    
    # a intermediate POI should NOT be the start POI, NO self-loops
    for pi in range(M):
        Cu[pi] = np.dot(unary_params[pi, :], unary_features[pi, :]) # if pi != ps else -np.inf
        for pj in range(M):
            Cp[pi, pj] = -np.inf if (pj == ps or pi == pj) else np.dot(pw_params[pi, pj, :], pw_features[pi, pj, :])
            
    # forward-backward procedure: adapted from the Rabiner paper
    Alpha = np.zeros((L, M), dtype=np.float)  # alpha_t(p_i)
    Beta  = np.zeros((L, M), dtype=np.float)  # beta_t(p_i)
    
    for pj in range(M): Alpha[1, pj] = Cp[ps, pj] + Cu[pj] + (0 if y_true is None else float(pj != y_true[1]))
    for t in range(2, L):
        for pj in range(M): # ps~~pi--pj
            loss = 0 if y_true is None else float(pj != y_true[t])  # pi varies, pj fixed
            Alpha[t, pj] = loss + np.max([Alpha[t-1, pi] + Cp[pi, pj] + Cu[pj] for pi in range(M)])
    
    for pi in range(M): Beta[L-1, pi] = 0 if y_true is None else float(pi != y_true[L-1])
    for t in range(L-1, 1, -1):
        for pi in range(M): # ps~~pi--pj
            loss = 0. if y_true is None else float(pi != y_true[t-1])  # pi fixed, pj varies
            Beta[t-1, pi] = loss + np.max([Cp[pi, pj] + Cu[pj] + Beta[t, pj] for pj in range(M)])
    Beta[0, ps] = np.max([Cp[ps, pj] + Cu[pj] + Beta[1, pj] for pj in range(M)])
    
    Fp = np.zeros((L-1, M, M), dtype=np.float)  # f_{t, t+1}(p, p')
    
    for t in range(L-1):
        for pi in range(M):
            for pj in range(M):
                Fp[t, pi, pj] = Alpha[t, pi] + Cp[pi, pj] + Cu[pj] + Beta[t+1, pj]
                
    # identify the best path/walk: adapted from the IJCAI01 paper
    y_best = np.ones(L, dtype=np.int) * (-1)
    y_best[0] = ps
    y_best[1] = np.argmax(Fp[0, ps, :])  # the start POI is specified
    for t in range(2, L): 
        y_best[t] = np.argmax(Fp[t-1, y_best[t-1], :])
    
    Q = []  # priority queue (min-heap)
    #with np.errstate(invalid='raise'):  # deal with overflow
    #    try: nIter = np.power(M, L-1) - np.prod([M-kx for kx in range(1,L)]) + top + \
    #                 (0 if y_true is None else len(y_true_list))
    #    except: nIter = maxIter
    #nIter = np.min([nIter, maxIter])
    nIter = maxIter
        
    # heap item for the best path/walk
    priority = -np.max(Alpha[L-1, :])  # -1 * score as priority
    partition_index = -1
    exclude_set = set()  
    hq.heappush(Q, HeapItem(priority, (y_best, partition_index, exclude_set)))
    
    results = []
    k = 0; y_last = None
    while len(Q) > 0 and k < nIter:
        hitem = hq.heappop(Q)
        k_priority = hitem.priority
        (k_best, k_partition_index, k_exclude_set) = hitem.task
        k += 1; y_last = k_best
     
        # allow sub-tours for training
        if path4train == False:
            if y_true is None: 
                #print(-k_priority)
                if len(set(k_best)) == L:
                    if DIVERSITY is True:
                        if len(results) == 0 or len(set(k_best) - set(results[-1][0])) > 0:
                            results.append((k_best, k)); top -= 1
                    else:
                        results.append((k_best, k)); top -= 1
                    if top == 0: return results
            else: # return k_best if it is NOT one of the ground truth labels
                if not np.any([np.all(np.asarray(k_best) == np.asarray(yj)) for yj in y_true_list]): return k_best
        
        # disallow sub-tours for training
        else:
            if len(set(k_best)) == L:
                if y_true is None: 
                    #print(-k_priority); 
                    if DIVERSITY is True:
                        if len(results) == 0 or len(set(k_best) - set(results[-1][0])) > 0:
                            results.append((k_best, k)); top -= 1
                    else:
                        results.append((k_best, k)); top -= 1
                    if top == 0: return results
                else: # return k_best if it is NOT one of the ground truth labels
                    if not np.any([np.all(np.asarray(k_best) == np.asarray(yj)) for yj in y_true_list]): return k_best

        # identify the (k+1)-th best path/walk given the 1st, 2nd, ..., k-th best: adapted from the IJCAI01 paper
        partition_index_start = 1
        if k_partition_index > 0:
            assert(k_partition_index < L)
            partition_index_start = k_partition_index

        for parix in range(partition_index_start, L):
            # new_best[:parix]
            new_best = np.zeros(L, dtype=np.int) * (-1)
            new_best[:parix] = k_best[:parix]
            if len(set(new_best[:parix])) < parix: break  # here break is more efficient than continue (skip all trajectories with sub-tours at once)
            
            # new_best[parix]
            #new_exclude_set = set({k_best[parix]})
            new_exclude_set = set(k_best[:parix+1]) # exclude all visited POIs
            if parix == partition_index_start: new_exclude_set = new_exclude_set | k_exclude_set
            candidate_points = [p for p in range(M) if p not in new_exclude_set]
            if len(candidate_points) == 0: continue
            candidate_maxix = np.argmax([Fp[parix-1, k_best[parix-1], p] for p in candidate_points])
            new_best[parix] = candidate_points[candidate_maxix]
            
            # new_best[parix+1:]
            for pk in range(parix+1, L):
                new_best[pk] = np.argmax([Fp[pk-1, new_best[pk-1], p] for p in range(M)])
                #new_best[pk] = np.argmax([Fp[pk-1, new_best[pk-1], p] for p in range(M) if p not in new_best[:pk]])  # incorrect
                #new_best[pk] = np.argmax([Fp[pk-1, new_best[pk-1], p] for p in range(M) if p not in new_best[:parix+1]]) # incorrect
            
            new_priority = Fp[parix-1, k_best[parix-1], new_best[parix]]
            if k_partition_index > 0:
                new_priority += (-k_priority) - Fp[parix-1, k_best[parix-1], k_best[parix]]
            new_priority *= -1.0   # NOTE: -np.inf - np.inf + np.inf = nan

            #print('push: %s, %d' % (str(new_best), parix))
            
            hq.heappush(Q, HeapItem(new_priority, (new_best, parix, new_exclude_set)))
            
    if k >= nIter: 
        sys.stderr.write('WARN: reaching max number of iterations, NO optimal solution found, return suboptimal solution.\n')
    if len(Q) == 0:
        sys.stderr.write('WARN: empty queue, return the last one\n')
    if y_true is None: 
        results.append((y_last, k)); top -= 1
        while len(Q) > 0 and top > 0:
            hitem = hq.heappop(Q)
            results.append((hitem.task[0], k))
            top -= 1
        return results
    else: 
        return y_last
