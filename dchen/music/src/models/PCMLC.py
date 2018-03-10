import sys
import time
import numpy as np
from sklearn.base import BaseEstimator
from scipy.sparse import issparse, isspmatrix_coo
from lbfgs import LBFGS, LBFGSError  # pip install pylbfgs
from joblib import Parallel, delayed

VERBOSE = 1
N_JOBS = 3


def risk_pclassification(W, b, X, Y, P, Q, p=1):
    """
        Empirical risk of p-classification loss for multilabel classification

        Input:
            - W: current weight matrix, K by D
            - b: current bias
            - X: feature matrix, N x D
            - Y: positive label matrix, N x K
            - p: constant for p-classification push loss
            - loss_type: valid assignment is 'example' or 'label'
                - 'example': compute a loss for each example, by the #positive or #negative labels per example
                - 'label'  : compute a loss for each label, by the #positive or #negative examples per label

        Output:
            - risk: empirical risk
            - db  : gradient of bias term
            - dW  : gradients of weights
    """
    assert p > 0
    assert Y.dtype == np.bool
    assert isspmatrix_coo(Y)  # scipy.sparse.coo_matrix type
    N, D = X.shape
    K = Y.shape[1]
    assert W.shape == (K, D)
    # shape = (N, 1) if loss_type == 'example' else (1, K)
    assert P.shape == Q.shape
    if P.shape[0] == 1:
        assert P.shape[1] == K
    else:
        assert P.shape == (N, 1)

    T1 = np.dot(X, W.T) + b
    T1p = np.zeros((N, K), dtype=np.float)
    T1p[Y.row, Y.col] = T1[Y.row, Y.col]
    T1n = T1 - T1p

    T2 = np.exp(-T1p)
    T2p = np.zeros((N, K), dtype=np.float)
    T2p[Y.row, Y.col] = T2[Y.row, Y.col]
    T2 = T2p * P

    T3 = np.exp(p * T1n)
    T3[Y.row, Y.col] = 0
    T3 = T3 * Q

    risk = np.sum(T2 + T3 / p)
    T4 = T3 - T2
    db = np.sum(T4)
    dW = np.dot(T4.T, X)

    if np.isnan(risk) or np.isinf(risk):
        sys.stderr('risk_pclassification(): risk is NaN or inf!\n')
        sys.exit(0)
    return risk, db, dW


class DataHelper:
    """
        SciPy sparse matrix slicing is slow, as stated here:
        https://stackoverflow.com/questions/42127046/fast-slicing-and-multiplication-of-scipy-sparse-csr-matrix
        Profiling confirms this inefficient slicing.
        This iterator aims to do slicing only once and cache the results.
    """
    def __init__(self, Y, ax=0, batch_size=256):
        assert ax in [0, 1]
        assert issparse(Y)
        self.init = False
        self.ax = ax
        self.starts = []
        self.ends = []
        self.Ys = []
        self.Ps = []
        self.Qs = []
        num = Y.shape[self.ax]
        bs = num if batch_size > num else batch_size
        self.n_batches = int((num-1) / bs) + 1
        Y = Y.tocsr() if self.ax == 0 else Y.tocsc()
        for nb in range(self.n_batches):
            ix_start = nb * bs
            ix_end = min((nb + 1) * bs, num)
            Yi = Y[ix_start:ix_end, :] if self.ax == 0 else Y[:, ix_start:ix_end]

            numPos = Yi.sum(axis=1-self.ax).A.reshape(-1)
            numNeg = Yi.shape[1-self.ax] - numPos
            nz_pix = np.nonzero(numPos)[0]  # taking care of zeros
            nz_nix = np.nonzero(numNeg)[0]
            P = np.zeros_like(numPos, dtype=np.float)
            Q = np.zeros_like(numNeg, dtype=np.float)
            P[nz_pix] = 1. / numPos[nz_pix]  # P = 1 / numPos
            Q[nz_nix] = 1. / numNeg[nz_nix]  # Q = 1 / numNeg
            shape = (len(P), 1) if self.ax == 0 else (1, len(P))

            self.starts.append(ix_start)
            self.ends.append(ix_end)
            self.Ys.append(Yi.tocoo())
            self.Ps.append(P.reshape(shape))
            self.Qs.append(Q.reshape(shape))
        self.init = True

    def get_data(self):
        assert self.init is True
        return self.starts, self.ends, self.Ys, self.Ps, self.Qs


def accumulate_risk_label(Wt, bt, X, Y, p, data_helper):
    assert data_helper is not None
    assert data_helper.ax == 1
    assert Wt.shape == (Y.shape[1], X.shape[1])
    starts, ends, Ys, Ps, Qs = data_helper.get_data()
    num = len(Ys)
    results = Parallel(n_jobs=N_JOBS)(delayed(risk_pclassification)
                                      (Wt[starts[i]:ends[i], :], bt, X, Ys[i], Ps[i], Qs[i], p=p) for i in range(num))
    denom = Y.shape[1]
    risk = 0.
    db = 0.
    dW_slices = []
    for t in results:
        risk += t[0] / denom
        db += t[1] / denom
        dW_slices.append(t[2] / denom)
    dW = np.vstack(dW_slices)
    return risk, db, dW


def accumulate_risk_example(Wt, bt, X, Y, p, data_helper):
    assert data_helper is not None
    assert data_helper.ax == 0
    assert Wt.shape == (Y.shape[1], X.shape[1])
    starts, ends, Ys, Ps, Qs = data_helper.get_data()
    denom = Y.shape[0]
    risk = 0.
    db = 0.
    dW = np.zeros_like(Wt)
    num = len(Ys)
    bs = 8
    n_batches = int((num-1) / bs) + 1
    indices = np.arange(num)
    for nb in range(n_batches):
        ixs = nb * bs
        ixe = min((nb + 1) * bs, num)
        ix = indices[ixs:ixe]
        res = Parallel(n_jobs=N_JOBS)(delayed(risk_pclassification)
                                      (Wt, bt, X[starts[i]:ends[i], :], Ys[i], Ps[i], Qs[i], p) for i in ix)
        assert len(res) <= bs
        for t in res:
            assert len(t) == 3
            risk += t[0] / denom
            db += t[1] / denom
            dW += t[2] / denom
    return risk, db, dW


# def accumulate_risk(Wt, bt, X, Y, p, loss, data_helper, verbose=0):
#     assert loss in ['example', 'label']
#     assert data_helper is not None
#     assert Wt.shape == (Y.shape[1], X.shape[1])
#     ax = 0 if loss == 'example' else 1
#     assert data_helper.ax == ax
#     risk = 0.
#     db = 0.
#     dW = np.zeros_like(Wt)
#     nb = 0
#     for ix_start, ix_end, Yi, Pi, Qi in zip(*(data_helper.get_data())):
#         nb += 1
#         if verbose > 2:
#             sys.stdout.write('\r%d / %d' % (nb, data_helper.n_batches))
#             sys.stdout.flush()
#         Xi = X[ix_start:ix_end, :] if ax == 0 else X
#         Wb = Wt if ax == 0 else Wt[ix_start:ix_end, :]
#         riski, dbi, dWi = risk_pclassification(Wb, bt, Xi, Yi, Pi, Qi, p=p, loss_type=loss)
#         assert dWi.shape == Wb.shape
#         denom = Y.shape[ax]
#         risk += riski / denom
#         db += dbi / denom
#         if ax == 0:
#             dW += dWi / denom
#         else:
#             dW[ix_start:ix_end, :] = dWi / denom
#     if verbose > 2:
#         print()
#     return risk, db, dW


def multitask_regulariser(Wt, bt, cliques):
    assert cliques is not None
    denom = 0.
    cost_mt = 0.
    dW_mt = np.zeros_like(Wt)
    for clq in cliques:
        npl = len(clq)
        if npl < 2:
            continue
        denom += npl * (npl - 1)
        M = -1 * np.ones((npl, npl), dtype=np.float)
        np.fill_diagonal(M, npl-1)
        Wu = Wt[clq, :]
        cost_mt += np.multiply(M, np.dot(Wu, Wu.T)).sum()
        dW_mt[clq, :] = np.dot(M, Wu)  # assume one playlist belongs to only one user
    cost_mt /= denom
    dW_mt = dW_mt * 2. / denom
    return cost_mt, dW_mt


def objective(w, dw, X, Y, C1=1, C2=1, C3=1, p=1, loss_type='example', cliques=None,
              data_helper_example=None, data_helper_label=None, fnpy=None):
        """
            - w : np.ndarray, current weights
            - dw: np.ndarray, OUTPUT array for gradients of w
            - cliques: a list of arrays, each array is the indices of playlists of the same user.
                       To require the parameters of label_i and label_j be similar by regularising
                       their diff if entry (i,j) is 1 (i.e. belong to the same user).
        """
        assert loss_type in ['example', 'label', 'both']
        assert C1 > 0
        assert C2 > 0
        assert C3 > 0
        assert p > 0
        t0 = time.time()
        N, D = X.shape
        K = Y.shape[1]
        assert w.shape[0] == K * D + 1
        b = w[0]
        W = w[1:].reshape(K, D)

        if loss_type == 'both':
            risk1, db1, dW1 = accumulate_risk_label(W, b, X, Y, p, data_helper=data_helper_label)
            risk2, db2, dW2 = accumulate_risk_example(W, b, X, Y, p, data_helper=data_helper_example)
            risk = risk1 + C2 * risk2
            db = db1 + C2 * db2
            dW = dW1 + C2 * dW2
        elif loss_type == 'label':
            risk, db, dW = accumulate_risk_label(W, b, X, Y, p, data_helper=data_helper_label)
        else:
            risk, db, dW = accumulate_risk_example(W, b, X, Y, p, data_helper=data_helper_example)

        J = risk + np.dot(W.ravel(), W.ravel()) * 0.5 / C1
        dW += W / C1

        if cliques is not None:
            cost_mt, dW_mt = multitask_regulariser(W, b, cliques)
            J += cost_mt / C3
            dW += dW_mt / C3

        dw[:] = np.r_[db, dW.ravel()]  # in-place assignment

        if VERBOSE > 0:
            print('Eval f, g: %.1f seconds used.' % (time.time() - t0))

        return J


def progress(x, g, f_x, xnorm, gnorm, step, k, ls, *args):
    """
        Report optimization progress.
        progress: callable(x, g, fx, xnorm, gnorm, step, k, num_eval, *args)
                  If not None, called at each iteration after the call to f with
                  the current values of x, g and f(x), the L2 norms of x and g,
                  the line search step, the iteration number,
                  the number of evaluations at this iteration and args.
    """
    print('Iter {:3d}:  f = {:15.9f},  |g| = {:15.9f},  {}'.format(k, f_x, gnorm, time.strftime('%Y-%m-%d %H:%M:%S')))

    # save intermediate weights
    fnpy = args[-1]
    if fnpy is not None and k > 20 and k % 10 == 0:
        try:
            print(fnpy)
            np.save(fnpy, x, allow_pickle=False)
        except (OSError, IOError, ValueError):
            sys.stderr.write('Save weights to .npy file failed\n')


class PCMLC(BaseEstimator):
    """All methods are necessary for a scikit-learn estimator"""

    def __init__(self, C1=1, C2=1, C3=1, p=1, loss_type='example'):
        """Initialisation"""
        assert C1 > 0
        assert C2 > 0
        assert C3 > 0
        assert p > 0
        assert loss_type in ['example', 'label', 'both'], \
            'Valid assignment for "loss_type" are: "example", "label", "both".'
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.p = p
        self.loss_type = loss_type
        self.trained = False

    def fit(self, X_train, Y_train, user_playlist_indices=None, batch_size=256, verbose=0, w0=None, fnpy=None):
        assert X_train.shape[0] == Y_train.shape[0]
        N, D = X_train.shape
        K = Y_train.shape[1]
        VERBOSE = verbose  # set verbose output, use a global variable in this case

        if VERBOSE > 0:
            t0 = time.time()

        if w0 is None:
            if fnpy is not None:
                try:
                    w0 = np.load(fnpy, allow_pickle=False)
                    assert w0.shape[0] == K * D + 1
                    print('Restore from %s' % fnpy)
                except (IOError, ValueError):
                    w0 = np.zeros(K * D + 1)
        else:
            assert w0.shape[0] == K * D + 1

        data_helper_example = None if self.loss_type == 'label' else DataHelper(Y_train, ax=0, batch_size=batch_size)
        data_helper_label = None if self.loss_type == 'example' else DataHelper(Y_train, ax=1, batch_size=batch_size)

        try:
            # f: callable(x, g, *args)
            # LBFGS().minimize(f, x0, progress=progress, args=args)
            optim = LBFGS()
            optim.linesearch = 'wolfe'
            res = optim.minimize(objective, w0, progress,
                                 args=(X_train, Y_train, self.C1, self.C2, self.C3, self.p, self.loss_type,
                                       user_playlist_indices, data_helper_example, data_helper_label, fnpy))
            self.b = res[0]
            self.W = res[1:].reshape(K, D)
            self.trained = True
        except (LBFGSError, MemoryError) as err:
            self.trained = False
            sys.stderr.write('LBFGS failed: {0}\n'.format(err))
            sys.stderr.flush()

        if VERBOSE > 0:
            print('Training finished in %.1f seconds' % (time.time() - t0))

    def decision_function(self, X_test):
        """Make predictions (score is a real number)"""
        assert self.trained is True, "Cannot make prediction before training"
        return np.dot(X_test, self.W.T) + self.b  # log of prediction score

    def predict(self, X_test):
        return self.decision_function(X_test)
    #    """Make predictions (score is boolean)"""
    #    preds = sigmoid(self.decision_function(X_test))
    #    return preds >= Threshold

    # inherit from BaseEstimator instead of re-implement
    # def get_params(self, deep = True):
    # def set_params(self, **params):
