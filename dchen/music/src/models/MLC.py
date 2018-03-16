import sys
import time
import numpy as np
from sklearn.base import BaseEstimator
from scipy.sparse import issparse, isspmatrix_coo
from lbfgs import LBFGS, LBFGSError  # pip install pylbfgs
from joblib import Parallel, delayed


def risk_pclassification(W, b, X, Y, P, Q, p=1):
    """
        Empirical risk of p-classification loss for multilabel classification

        Input:
            - W: current weight matrix, K by D
            - b: current bias
            - X: feature matrix, N x D
            - Y: positive label matrix, N x K
            - P: 1/#positive_example, N x 1
            - Q: 1/#negative_example, N x 1
            - p: constant for p-classification push loss

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
    assert b.shape == (1, K)
    assert P.shape == Q.shape == (1, K)

    T1 = np.dot(X, W.T) + b  # N x K
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
    db = np.sum(T4, axis=0)
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
    def __init__(self, Y, ax=1, batch_size=256):
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


def accumulate_risk_label(W, b, X, Y, p, data_helper, njobs):
    assert data_helper is not None
    assert data_helper.ax == 1
    N, D = X.shape
    K = Y.shape[1]
    assert W.shape == (K, D)
    assert b.shape == (1, K)
    starts, ends, Ys, Ps, Qs = data_helper.get_data()
    num = len(Ys)
    if num == 1:
        njobs = 1

    results = Parallel(n_jobs=njobs)(delayed(risk_pclassification)(W[starts[i]:ends[i], :], b[:, starts[i]:ends[i]],
                                                                   X, Ys[i], Ps[i], Qs[i], p=p) for i in range(num))
    risk = 0.
    db_slices = []
    dW_slices = []
    for t in results:
        risk += t[0] / K
        db_slices.append(t[1])
        dW_slices.append(t[2])
    db = np.hstack(db_slices) / K
    dW = np.vstack(dW_slices) / K
    return risk, db, dW


def multitask_regulariser(W, cliques):
    assert cliques is not None
    denom = 0.
    cost_mt = 0.
    dW_mt = np.zeros_like(W)
    for clq in cliques:
        npl = len(clq)
        if npl < 2:
            continue
        denom += npl * (npl - 1)
        M = -1 * np.ones((npl, npl), dtype=np.float)
        np.fill_diagonal(M, npl-1)
        Wu = W[clq, :]
        cost_mt += np.multiply(M, np.dot(Wu, Wu.T)).sum()
        dW_mt[clq, :] = np.dot(M, Wu)  # assume one playlist belongs to only one user
    cost_mt /= denom
    dW_mt = dW_mt * 2. / denom
    return cost_mt, dW_mt


def objective(w, dw, X, Y, C1=1, C3=1, p=1, cliques=None, data_helper=None, njobs=1, verbose=0, fnpy=None):
        """
            - w : np.ndarray, current weights
            - dw: np.ndarray, OUTPUT array for gradients of w
            - cliques: a list of arrays, each array is the indices of playlists of the same user.
                       To require the parameters of label_i and label_j be similar by regularising
                       their diff if entry (i,j) is 1 (i.e. belong to the same user).
        """
        assert C1 > 0
        assert C3 > 0
        assert p > 0
        t0 = time.time()
        N, D = X.shape
        K = Y.shape[1]
        assert w.shape[0] == K * (D + 1)
        b = w[:K].reshape(1, K)
        W = w[K:].reshape(K, D)

        risk, db, dW = accumulate_risk_label(W, b, X, Y, p, data_helper=data_helper, njobs=njobs)
        J = risk + np.dot(W.ravel(), W.ravel()) * 0.5 / C1
        dW += W / C1

        if cliques is not None:
            cost_mt, dW_mt = multitask_regulariser(W, cliques)
            J += cost_mt / C3
            dW += dW_mt / C3

        dw[:] = np.r_[db.ravel(), dW.ravel()]  # in-place assignment

        if verbose > 1:
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
    verbose = args[-2]
    if verbose > 1:
        print('Iter {:3d}: f ={:15.9f};  |g| ={:15.9f};  {}'.format(k, f_x, gnorm, time.strftime('%Y-%m-%d %H:%M:%S')))

    # save intermediate weights
    fnpy = args[-1]
    assert type(fnpy) == str
    if fnpy.endswith('.npy') and k > 20 and k % 10 == 0:
        try:
            np.save(fnpy, x, allow_pickle=False)
            if verbose > 1:
                print('Save to %s' % fnpy)
        except (OSError, IOError, ValueError) as err:
            sys.stderr.write('Save intermediate weights failed: {0}\n'.format(err))


class MLC(BaseEstimator):
    """All methods are necessary for a scikit-learn estimator"""

    def __init__(self, C1=1, C3=1, p=1):
        """Initialisation"""
        assert C1 > 0
        assert C3 > 0
        assert p > 0
        self.C1 = C1
        self.C3 = C3
        self.p = p
        self.trained = False

    def fit(self, X_train, Y_train, user_playlist_indices=None, batch_size=256, w0=None, njobs=1, verbose=0, fnpy='_'):
        assert X_train.shape[0] == Y_train.shape[0]
        N, D = X_train.shape
        K = Y_train.shape[1]

        if verbose > 1:
            t0 = time.time()

        if verbose > 0:
            if user_playlist_indices is None:
                print('\nC: %g, p: %g' % (self.C1, self.p))
            else:
                print('\nC1: %g, C3: %g, p: %g' % (self.C1, self.C3, self.p))

        if w0 is not None:
            assert w0.shape[0] == K * (D + 1)
        else:
            if fnpy is not None:
                try:
                    w0 = np.load(fnpy, allow_pickle=False)
                    assert w0.shape[0] == K * (D + 1)
                    print('Restore from %s' % fnpy)
                except (IOError, ValueError):
                    w0 = np.zeros(K * (D + 1))

        data_helper = DataHelper(Y_train, ax=1, batch_size=batch_size)

        try:
            # f: callable(x, g, *args)
            # LBFGS().minimize(f, x0, progress=progress, args=args)
            optim = LBFGS()
            optim.linesearch = 'wolfe'
            res = optim.minimize(objective, w0, progress,
                                 args=(X_train, Y_train, self.C1, self.C3, self.p,
                                       user_playlist_indices, data_helper, njobs, verbose, fnpy))
            self.b = res[:K].reshape(1, K)
            self.W = res[K:].reshape(K, D)
            self.trained = True
        except (LBFGSError, MemoryError) as err:
            self.trained = False
            sys.stderr.write('LBFGS failed: {0}\n'.format(err))
            sys.stderr.flush()

        if verbose > 1:
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
