import sys
import time
import numpy as np
from scipy.sparse import issparse, isspmatrix_coo, isspmatrix_csc
from lbfgs import LBFGS, LBFGSError  # pip install pylbfgs
# from joblib import Parallel, delayed


def risk_pclassification(mu, v, Wu, X, Yu, Pu, Qu, param_dict, u, UF=None):
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
    C, p, N, U = param_dict['C'], param_dict['p'], param_dict['N'], param_dict['U']
    assert C > 0
    assert p > 0
    assert Yu.dtype == np.bool
    assert isspmatrix_coo(Yu)  # scipy.sparse.coo_matrix type
    M, D = X.shape
    Nu = Yu.shape[1]
    if UF is not None:
        assert UF.shape == (M, U)
        assert 0 <= u < U
        assert UF.shape == (M, U)
        Xu = np.concatenate([X, np.delete(UF, u, axis=1)], axis=1)
        D += U - 1
    else:
        Xu = X

    assert v.shape == mu.shape == (D,)
    assert Wu.shape == (Nu, D)
    assert Pu.shape == Qu.shape == (Nu,)
    Wt = Wu + (v + mu).reshape(1, D)
    T1 = np.dot(Xu, Wt.T)
    T1p = np.zeros(T1.shape)
    T1p[Yu.row, Yu.col] = T1[Yu.row, Yu.col]
    T1n = T1 - T1p

    T2p = np.exp(-T1p)
    T2 = np.zeros(T1.shape)
    T2[Yu.row, Yu.col] = T2p[Yu.row, Yu.col]
    T2 *= Pu

    T3 = np.exp(p * T1n)
    T3[Yu.row, Yu.col] = 0
    T3 *= Qu

    risk = np.dot(v, v) / U + np.sum([np.dot(Wu[k, :], Wu[k, :]) for k in range(Nu)]) / N
    risk = risk * C / 2 + np.sum(T2 + T3 / p) / N
    T4 = T3 - T2
    T5 = np.dot(T4.T, Xu) / N
    dW = T5
    dv = np.sum(T5, axis=0)
    dmu = dv

    if np.isnan(risk) or np.isinf(risk):
        sys.stderr.write('risk_pclassification(): risk is NaN or inf!\n')
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


def accumulate_risk(mu, V, W, X, Y, C, p, cliques, data_helper, UF=None, njobs=1):
    M, D = X.shape
    N = Y.shape[1]
    U = len(cliques)
    if UF is not None:
        assert UF.shape == (M, U)
        D += U - 1
    assert mu.shape == (D,)
    assert V.shape == (U, D)
    assert W.shape == (N, D)
    Ys, Ps, Qs = data_helper.get_data()
    assert U == len(Ys) == len(Ps) == len(Qs)
    param_dict = {'C': C, 'p': p, 'N': N, 'U': U}
    # if U == 1:
    #    njobs = 1

    # results = Parallel(n_jobs=njobs)(delayed(risk_pclassification)(mu, V[u, :], W[cliques[u], :], X, Ys[u], Ps[u],
    #                                                               Qs[u], param_dict, u, UF) for u in range(U))
    J = np.dot(mu, mu) * C / 2
    dV_slices = []
    dW_slices = []
    dmu = C * mu
    for u in range(U):
        results = risk_pclassification(mu, V[u, :], W[cliques[u], :], X, Ys[u], Ps[u], Qs[u], param_dict, u, UF)
        J += results[0]
        dmu += results[1]
        dV_slices.append(results[2])
        dW_slices.append(results[3])
    dV = V * C / U + np.vstack(dV_slices)
    dW = W * C / N + np.vstack(dW_slices)
    return J, dmu, dV, dW


def objective(w, dw, X, Y, C, p, cliques, data_helper, UF=None, njobs=1, verbose=0, fnpy=None):
    t0 = time.time()
    assert C > 0
    assert p > 0
    M, D = X.shape
    N = Y.shape[1]
    U = len(cliques)
    if UF is not None:
        assert UF.shape == (M, U)
        D += U - 1
    assert w.shape == ((U + N + 1) * D,)
    mu = w[:D]
    V = w[D:(U + 1) * D].reshape(U, D)
    W = w[(U + 1) * D:].reshape(N, D)

    J, dmu, dV, dW = accumulate_risk(mu, V, W, X, Y, C, p, cliques, data_helper, UF, njobs)
    dw[:] = np.r_[dmu.ravel(), dV.ravel(), dW.ravel()]  # in-place assignment

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

    def __init__(self, X_train, Y_train, C, p, user_playlist_indices, label_feature=False):
        assert issparse(Y_train)
        assert C > 0
        assert p > 0
        self.X, self.Y = X_train, Y_train
        self.M, self.D = self.X.shape
        self.N = self.Y.shape[1]
        assert self.M == self.Y.shape[0]
        self.C = C
        self.p = p

        self.cliques = user_playlist_indices
        self.U = len(self.cliques)
        self.u2pl = self.cliques
        self.pl2u = np.zeros(self.N, dtype=np.int32)
        for u in range(self.U):
            clq = self.cliques[u]
            self.pl2u[clq] = u

        Ycsc = self.Y.tocsc()
        self.data_helper = DataHelper(Ycsc, self.cliques)
        if label_feature is True:
            self.UF = np.zeros((self.M, self.U), dtype=np.float)
            for u in range(self.U):
                clq = self.u2pl[u]
                self.UF[:, u] = Ycsc[:, clq].sum(axis=1).A.reshape(-1)
            UF_mean = np.mean(self.UF, axis=0).reshape((1, -1))
            UF_std = np.std(self.UF, axis=0).reshape((1, -1)) + 10 ** (-6)
            self.UF -= UF_mean
            self.UF /= UF_std
            self.D += self.U - 1
        else:
            self.UF = None

    def fit(self, w0=None, njobs=1, verbose=0, fnpy='_'):
        N, U, D = self.N, self.U, self.D

        if verbose > 0:
            t0 = time.time()

        if verbose > 0:
            print('\nC: %g, p: %g' % (self.C, self.p))

        if w0 is not None:
            assert w0.shape[0] == (U + N + 1) * D
        else:
            if fnpy is None:
                w0 = np.zeros((U + N + 1) * D)
            else:
                try:
                    w0 = np.load(fnpy, allow_pickle=False)
                    assert w0.shape[0] == (U + N + 1) * D
                    print('Restore from %s' % fnpy)
                except (IOError, ValueError):
                    w0 = np.zeros((U + N + 1) * D)

        try:
            # f: callable(x, g, *args)
            # LBFGS().minimize(f, x0, progress=progress, args=args)
            optim = LBFGS()
            optim.linesearch = 'wolfe'
            res = optim.minimize(objective, w0, progress,
                                 args=(self.X, self.Y, self.C, self.p, self.cliques, self.data_helper,
                                       self.UF, njobs, verbose, fnpy))
            self.mu = res[:D]
            self.V = res[D:(U + 1) * D].reshape(U, D)
            self.W = res[(U + 1) * D:].reshape(N, D)
            self.trained = True
        except (LBFGSError, MemoryError) as err:
            self.trained = False
            sys.stderr.write('LBFGS failed: {0}\n'.format(err))
            sys.stderr.flush()

        if verbose > 0:
            print('Training finished in %.1f seconds' % (time.time() - t0))

    def predict(self, X_test):
        """Make predictions (score is a real number)"""
        assert self.trained is True, 'Cannot make prediction before training'
        U, D = self.U, self.D
        assert D == X_test.shape[1]
        preds = []
        for u in range(U):
            clq = self.cliques[u]
            Wt = self.W[clq, :] + (self.V[u, :] + self.mu).reshape(1, D)
            preds.append(np.dot(X_test, Wt.T))
        return np.hstack(preds)
