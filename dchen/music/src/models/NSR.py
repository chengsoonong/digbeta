import sys
import time
import numpy as np
from scipy.sparse import issparse, isspmatrix_coo
# from cvxopt import matrix, spmatrix
from lbfgs import LBFGS, LBFGSError  # pip install pylbfgs
from joblib import Parallel, delayed


def risk_pclassification(mu, v, Wu, X, Yu, C, p, N, U):
    """
        Empirical risk of p-classification loss for multilabel classification

        Input:
            - W: current weight matrix, N by D
            - X: feature matrix, M x D
            - Y: positive label matrix, M x N
            - p: constant for p-classification push loss

        Output:
            - risk: empirical risk
            - dW  : gradients of weights
    """
    assert C > 0
    assert p > 0
    assert Yu.dtype == np.bool
    assert isspmatrix_coo(Yu)  # scipy.sparse.coo_matrix type
    M, D = X.shape
    Nu = Yu.shape[1]
    assert v.shape == mu.shape == (D,)
    assert Wu.shape == (Nu, D)
    Wt = Wu + (v + mu).reshape(1, D)

    T1 = np.dot(X, Wt.T)
    T1p = np.zeros(T1.shape, dtype=np.float)
    T1p[Yu.row, Yu.col] = T1[Yu.row, Yu.col]
    T1n = T1 - T1p

    T2p = np.exp(-T1p)
    T2 = np.zeros(T1.shape, dtype=np.float)
    T2[Yu.row, Yu.col] = T2p[Yu.row, Yu.col]

    T3 = np.exp(p * T1n)
    T3[Yu.row, Yu.col] = 0

    risk = np.dot(v, v) / U + np.sum([np.dot(Wu[k, :], Wu[k, :]) for k in range(Nu)]) / N
    risk = risk * C / 2 + np.sum(T2 + T3 / p) / N
    T4 = T3 - T2
    T5 = np.dot(T4.T, X)
    dW = T5 / N
    dv = np.sum(T5, axis=0) / N
    dmu = dv.copy()

    if np.isnan(risk) or np.isinf(risk):
        sys.stderr('risk_pclassification(): risk is NaN or inf!\n')
        sys.exit(0)
    return risk, dmu, dv, dW


class DataHelper:
    """
        SciPy sparse matrix slicing is slow, as stated here:
        https://stackoverflow.com/questions/42127046/fast-slicing-and-multiplication-of-scipy-sparse-csr-matrix
        Profiling confirms this inefficient slicing.
        This iterator aims to do slicing only once and cache the results.
    """
    def __init__(self, Y, cliques):
        assert issparse(Y)
        U = len(cliques)
        self.init = False
        self.Ys = []
        Y = Y.tocsc()
        for u in range(U):
            clq = cliques[u]
            Yu = Y[:, clq]
            Yu.eliminate_zeros()
            self.Ys.append(Yu.tocoo())
        self.init = True

    def get_data(self):
        assert self.init is True
        return self.Ys


def accumulate_risk(mu, V, W, X, Y, C, p, cliques, data_helper, njobs):
    M, D = X.shape
    N = Y.shape[1]
    U = len(cliques)
    assert mu.shape == (D,)
    assert V.shape == (U, D)
    assert W.shape == (N, D)
    Ys = data_helper.get_data()
    assert U == len(Ys)
    if U == 1:
        njobs = 1

    results = Parallel(n_jobs=njobs)(delayed(risk_pclassification)(mu, V[u, :], W[cliques[u], :], X, Ys[u],
                                                                   C, p, N, U) for u in range(U))
    J = np.dot(mu, mu) * C / 2
    dV_slices = []
    dW_slices = []
    dmu = C * mu
    for t in results:
        J += t[0]
        dmu += t[1]
        dV_slices.append(t[2])
        dW_slices.append(t[3])
    dV = V * C / U + np.vstack(dV_slices)
    dW = W * C / N + np.vstack(dW_slices)
    return J, dmu, dV, dW


def objective_clf(w, dw, X, Y, C, p, cliques, data_helper, njobs=1, verbose=0, fnpy=None):
    t0 = time.time()
    assert C > 0
    assert p > 0
    M, D = X.shape
    N = Y.shape[1]
    U = len(cliques)
    assert w.shape == ((U + N + 1) * D,)
    mu = w[:D]
    V = w[D:(U+1)*D].reshape(U, D)
    W = w[(U+1)*D:].reshape(N, D)

    J, dmu, dV, dW = accumulate_risk(mu, V, W, X, Y, C, p, cliques, data_helper, njobs)
    dw[:] = np.r_[dmu.ravel(), dV.ravel(), dW.ravel()]  # in-place assignment

    if verbose > 0:
        print('Eval f, g: %.1f seconds used.' % (time.time() - t0))

    return J

# ncalls = 0

def obj_clf_loop(w, dw, X, Y, C, p, cliques, data_helper, njobs=1, verbose=0, fnpy=None):
    # global ncalls; ncalls += 1; sys.stdout.write('\r%d' % ncalls); sys.stdout.flush()
    assert C > 0
    assert p > 0
    assert Y.dtype == np.bool
    M, D = X.shape
    N = Y.shape[1]
    U = len(cliques)
    assert w.shape == ((U + N + 1) * D,)
    mu = w[:D]
    V = w[D:(U+1)*D].reshape(U, D)
    W = w[(U+1)*D:].reshape(N, D)
    Jv = np.sum([np.dot(V[u, :], V[u, :]) for u in range(U)]) / U
    Jw = np.sum([np.dot(W[i, :], W[i, :]) for i in range(N)]) / N
    Jmu = np.dot(mu, mu)
    pl2u = dict()
    for u in range(U):
        for k in cliques[u]:
            pl2u[k] = u
    Jn = 0.
    dmu = np.zeros_like(mu)
    dV = np.zeros_like(V)
    dW = np.zeros_like(W)
    for k in range(N):
        u = pl2u[k]
        for m in range(M):
            score = np.dot(V[u, :] + W[k, :] + mu, X[m, :])
            if Y[m, k] is True:
                s = np.exp(-score)
                g = -X[m, :] * s
            else:
                s = np.exp(p * score) / p
                g = X[m, :] * s * p
            Jn += s
            dV[u, :] += g
            dW[k, :] += g
            dmu += g
    J = (Jv + Jw + Jmu) * C / 2 + Jn / N
    dV = V * C / U + dV / N
    dW = W * C / N + dW / N
    dmu = C * mu + dmu / N
    dw[:] = np.r_[dmu.ravel(), dV.ravel(), dW.ravel()]
    
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
    if verbose > 0:
        print('Iter {:3d}: f ={:15.9f};  |g| ={:15.9f};  {}'.format(k, f_x, gnorm, time.strftime('%Y-%m-%d %H:%M:%S')))

    # save intermediate weights
    fnpy = args[-1]
    assert type(fnpy) == str
    if fnpy.endswith('.npy') and k > 20 and k % 10 == 0:
        try:
            np.save(fnpy, x, allow_pickle=False)
            if verbose > 0:
                print('Save to %s' % fnpy)
        except (OSError, IOError, ValueError) as err:
            sys.stderr.write('Save intermediate weights failed: {0}\n'.format(err))


class NSR():
    """Class for New Song Recommendation task"""

    def __init__(self, C=1, p=1):
        """Initialisation"""
        assert C > 0
        assert p > 0
        self.C = C
        self.p = p
        self.trained = False

#     def preprocess(self):
#         self.XX = np.dot(self.X_train, self.X_train.T)
#         self.P = np.log(self.N) + np.log(self.Y_train.sum(axis=0))
#         num = self.X_train.shape[0] * self.N
#         self.H0 = spmatrix(x=[0], I=[0], J=[0], size=(num, num), tc='d')
#         for i in range(self.N):

#     def dual_objective(self, theta, grad=False, hessian=False):
#         M, D = self.X_train.shape
#         N = self.N
#         U = self.U
#         theta = np.asarray(theta)
#         assert len(theta) == M * N
#         T1 = theta.reshape(M, N)  # M x N
#         T2 = np.dot(T1.T, self.X_train)      # N x D
#         T3 = np.dot(T2, T2.T)     # N x N

#         J = N * np.trace(T3) + np.sum(T3)
#         for clq in self.cliques:
#             assert len(clq) > 0
#             Tu = T3[clq, clq]
#             J += Tu.sum()
#         J /= 2 * C

#         if grad is True:
#             dT = np.zeros(T1.shape, dtype=np.float)
#             Ts = T1.sum(axis=1)

#         for k in range(N):
#             J += (1 - self.P[k]) * self.Y_train[:, k].dot(T1[:, k])
#             R = np.log(self.Y_train[:, k].multiply(-T1[:, k]) + 1 - self.Y_train[:, k])
#             J -= np.dot(T1[:, k], R)
#             if grad is True:
#                 u = self.playlist2user[k]
#                 Q = U * T1[:, self.cliques[u]].sum(axis=1) + N * T1[:, k] + Ts
#                 dT[:, k] = np.dot(self.XX, Q.T) / C - P[k] * self.Y_train[:, k] - R

    def fit(self, X_train, Y_train, user_playlist_indices, w0=None, njobs=1, verbose=0, fnpy='_'):
        assert X_train.shape[0] == Y_train.shape[0]
        M, D = X_train.shape
        N = Y_train.shape[1]
        U = len(user_playlist_indices)
        assert N == np.sum([len(clq) for clq in user_playlist_indices])

        if verbose > 0:
            t0 = time.time()

        if verbose > 0:
            print('\nC: %g, p: %g' % (self.C, self.p))

        if w0 is not None:
            assert w0.shape[0] == (U + N + 1) * D
        else:
            if fnpy is not None:
                try:
                    w0 = np.load(fnpy, allow_pickle=False)
                    assert w0.shape[0] == (U + N + 1) * D
                    print('Restore from %s' % fnpy)
                except (IOError, ValueError):
                    w0 = np.zeros((U + N + 1) * D)

        data_helper = DataHelper(Y_train, user_playlist_indices)

        try:
            # f: callable(x, g, *args)
            # LBFGS().minimize(f, x0, progress=progress, args=args)
            optim = LBFGS()
            optim.linesearch = 'wolfe'
            res = optim.minimize(objective_clf, w0, progress,
                                 args=(X_train, Y_train, self.C, self.p, user_playlist_indices,
                                       data_helper, njobs, verbose, fnpy))
            self.mu = res[:D]
            self.V = res[D:(U+1)*D].reshape(U, D)
            self.W = res[(U+1)*D:].reshape(N, D)
            self.trained = True
        except (LBFGSError, MemoryError) as err:
            self.trained = False
            sys.stderr.write('LBFGS failed: {0}\n'.format(err))
            sys.stderr.flush()

        if verbose > 0:
            print('Training finished in %.1f seconds' % (time.time() - t0))

    def predict(self, X_test, user_playlist_indices):
        """Make predictions (score is a real number)"""
        assert self.trained is True, "Cannot make prediction before training"
        U = len(user_playlist_indices)
        D = X_test.shape[1]
        preds = []
        for u in range(U):
            clq = user_playlist_indices[u]
            Wt = self.W[clq, :] + (self.V[u, :] + self.mu).reshape(1, D)
            preds.append(np.dot(X_test, Wt.T))
        return np.hstack(preds)
