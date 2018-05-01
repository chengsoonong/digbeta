import sys
import time
import numpy as np
from scipy.sparse import isspmatrix_coo, isspmatrix_csc
from lbfgs import LBFGS, LBFGSError  # pip install pylbfgs


def risk_per_user(v, Wu, mu, X, Yu, Pu, Qu, p, N):
    """
        Empirical risk of bottom version of p-classification loss

        Input:
            - W: current weight matrix, N by D
            - X: feature matrix, M x D
            - Y: positive label matrix, M x N
            - p: constant for p-classification push loss

        Output:
            - risk: empirical risk
            - dW  : gradients of weights
    """
    assert p > 0
    assert Yu.dtype == np.bool
    assert isspmatrix_coo(Yu)  # scipy.sparse.coo_matrix type
    M, D = X.shape
    Nu = Yu.shape[1]

    assert v.shape == mu.shape == (D,)
    assert Wu.shape == (Nu, D)
    assert Pu.shape == Qu.shape == (Nu,)
    Wt = Wu + (v + mu).reshape(1, D)
    T1 = np.dot(X, Wt.T)
    T1p = np.zeros(T1.shape)
    T1p[Yu.row, Yu.col] = T1[Yu.row, Yu.col]
    T1n = T1 - T1p

    T2p = np.exp(-p * T1p)
    T2 = np.zeros(T1.shape)
    T2[Yu.row, Yu.col] = T2p[Yu.row, Yu.col]
    T2 *= Pu

    T3 = np.exp(T1n)
    T3[Yu.row, Yu.col] = 0
    T3 *= Qu

    risk = np.sum(T2 / p + T3) / N
    T4 = T3 - T2
    T5 = np.dot(T4.T, X) / N
    dv = np.sum(T5, axis=0)
    dW = T5
    dmu = dv

    if np.isnan(risk) or np.isinf(risk):
        sys.stderr.write('risk_pclassification(): risk is NaN or inf!\n')
        sys.exit(0)
    return risk, dv, dW, dmu


class DataHelper:
    """
        SciPy sparse matrix slicing is slow, as stated here:
        https://stackoverflow.com/questions/42127046/fast-slicing-and-multiplication-of-scipy-sparse-csr-matrix
        Profiling confirms this inefficient slicing.
        This iterator aims to do slicing only once and cache the results.
    """
    def __init__(self, Y, cliques):
        assert isspmatrix_csc(Y)
        M, N = Y.shape
        assert np.all(np.arange(N) == np.asarray(sorted([k for clq in cliques for k in clq])))
        U = len(cliques)
        self.init = False
        self.Ys = []
        self.Ps = []
        self.Qs = []
        Mplus = Y.sum(axis=0).A.reshape(-1)
        P = 1. / Mplus
        Q = 1. / (M - Mplus)
        for u in range(U):
            clq = cliques[u]
            Yu = Y[:, clq]
            self.Ys.append(Yu.tocoo())
            self.Ps.append(P[clq])
            self.Qs.append(Q[clq])
        self.init = True

    def get_data(self):
        assert self.init is True
        return self.Ys, self.Ps, self.Qs


def objective(w, dw, X, Y, C1, p, cliques, data_helper, verbose=0, fnpy=None):
    t0 = time.time()
    assert C1 > 0
    assert p > 0
    M, D = X.shape
    N = Y.shape[1]
    U = len(cliques)
    assert w.shape == ((U + N + 1) * D,)
    V = w[:U * D].reshape(U, D)
    W = w[U * D:(U + N) * D].reshape(N, D)
    mu = w[(U + N) * D:]
    assert mu.shape == (D,)

    Ys, Ps, Qs = data_helper.get_data()
    assert U == len(Ys) == len(Ps) == len(Qs)

    dV = V * C1
    dW = np.zeros(W.shape)
    dmu = np.zeros(D)
    J = 0.
    for u in range(U):
        clq = cliques[u]
        v = V[u, :]
        J += v.dot(v) * C1 / 2
        res = risk_per_user(v, W[clq, :], mu, X, Ys[u], Ps[u], Qs[u], p, N)
        J += res[0]
        dV[u, :] += res[1]
        dW[clq, :] = res[2]
        dmu += res[3]

    dw[:] = np.r_[dV.ravel(), dW.ravel(), dmu.ravel()]  # in-place assignment

    if verbose > 0:
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


class MTC():
    """Multitask classification"""

    def __init__(self, X_train, Y_train, C1, C2, p, user_playlist_indices):
        if not isspmatrix_csc(Y_train):
            raise ValueError('ERROR: %s\n' % 'Y_train should be a parse csc_matrix.')
        if not np.all(np.array([C1, C2]) > 0):
            raise ValueError('ERROR: %s\n' % 'Regularisation parameters should be positive.')
        if p <= 0:
            raise ValueError('ERROR: %s\n' % 'parameter "p" should be positive.')

        self.X, self.Y = X_train, Y_train
        self.M, self.D = self.X.shape
        self.N = self.Y.shape[1]
        assert self.M == self.Y.shape[0]
        self.C1 = C1
        self.C2 = C2
        self.p = p

        self.cliques = user_playlist_indices
        self.U = len(self.cliques)
        self.u2pl = self.cliques
        self.pl2u = np.zeros(self.N, dtype=np.int32)
        for u in range(self.U):
            clq = self.cliques[u]
            self.pl2u[clq] = u
        self.data_helper = DataHelper(self.Y, self.cliques)

    def fit(self, w0=None, verbose=0, fnpy='_'):
        N, U, D = self.N, self.U, self.D

        if verbose > 0:
            t0 = time.time()

        if verbose > 0:
            print('\nC: %g, %g, p: %g' % (self.C1, self.C2, self.p))

        num_vars = (U + N + 1) * D
        if w0 is None:
            np.random.seed(0)
            if fnpy is not None:
                try:
                    w0 = np.load(fnpy, allow_pickle=False)
                    print('Restore from %s' % fnpy)
                except (IOError, ValueError):
                    w0 = 1e-3 * np.random.randn(num_vars)
            else:
                w0 = 1e-3 * np.random.randn(num_vars)
        if w0.shape != (num_vars,):
            raise ValueError('ERROR: incorrect dimention for initial weights.')

        try:
            # f: callable(x, g, *args)
            # LBFGS().minimize(f, x0, progress=progress, args=args)
            optim = LBFGS()
            optim.linesearch = 'wolfe'
            optim.orthantwise_c = self.C2
            optim.orthantwise_start = U * D  # start index to compute L1 regularisation (included)
            optim.orthantwise_end = w0.shape[0]    # end   index to compute L1 regularisation (not included)
            res = optim.minimize(objective, w0, progress,
                                 args=(self.X, self.Y, self.C1, self.p, self.cliques,
                                       self.data_helper, verbose, fnpy))
            self.V = res[:U * D].reshape(U, D)
            self.W = res[U * D:(U + N) * D].reshape(N, D)
            self.mu = res[(U + N) * D:]
            assert self.mu.shape == (D,)
            self.trained = True
        except (LBFGSError, MemoryError) as err:
            self.trained = False
            sys.stderr.write('LBFGS failed: {0}\n'.format(err))
            sys.stderr.flush()

        if verbose > 0:
            print('Training finished in %.1f seconds' % (time.time() - t0))
