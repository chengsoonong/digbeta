import sys
import time
import numpy as np
from scipy.sparse import isspmatrix_csc, isspmatrix_coo
from lbfgs import LBFGS, LBFGSError  # pip install pylbfgs


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
        Mplus = Y.sum(axis=0).A.reshape(-1)
        self.Q = 1. / (N * Mplus)
        for u in range(U):
            clq = cliques[u]
            Yu = Y[:, clq]
            self.Ys.append(Yu.tocoo())
        self.init = True

    def get_data(self):
        assert self.init is True
        return self.Ys, self.Q


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


def objective(w, dw, X, cliques, data_helper, param_dict, verbose, fnpy):
    """
        The callback for calculating the objective
    """
    t0 = time.time()
    D = X.shape[1]
    U = len(cliques)
    N = param_dict['N']
    C1, C2, C3 = param_dict['C']
    assert w.shape == ((U + N + 1) * D,)
    mu = w[:D]
    V = w[D:(U + 1) * D].reshape(U, D)
    W = w[(U + 1) * D:].reshape(N, D)

    Ys, Q = data_helper.get_data()
    assert U == len(Ys)
    assert Q.shape == (N,)

    J = 0.
    dV = V * C1 / U
    dW = W * C2 / N
    dmu = C3 * mu

    J += np.sum([np.dot(V[u, :], V[u, :]) for u in range(U)]) * 0.5 * C1 / U
    J += np.sum([np.dot(W[k, :], W[k, :]) for k in range(N)]) * 0.5 * C2 / N
    J += np.dot(mu, mu) * 0.5 * C3
    for u in range(U):
        clq = cliques[u]
        Nu = len(clq)
        Yu = Ys[u]
        assert isspmatrix_coo(Yu)

        Wt = W[clq, :] + (V[u, :] + mu).reshape(1, D)
        T1 = np.dot(X, Wt.T)  # M by Nu
        T1p = np.zeros(T1.shape)
        T1p[Yu.row, Yu.col] = T1[Yu.row, Yu.col]
        T1n = T1 - T1p  # M by Nu
        T1n[Yu.row, Yu.col] = -np.inf  # mask entries for positive labels
        max_rowix = T1n.argmax(axis=0)  # Nu

        assert max_rowix.shape == (Nu,)
        Qu = Q[clq] * np.exp(T1n[max_rowix, np.arange(Nu)])

        T2 = np.exp(-T1p)  # M by Nu
        T2p = np.zeros(T2.shape)
        T2p[Yu.row, Yu.col] = T2[Yu.row, Yu.col]

        J += np.dot(Qu, T2p.sum(axis=0))

        T3 = np.dot(T2p.T, X)  # Nu by D

        Tn = X[max_rowix, :]   # Nu by D
        T4 = Tn * T2p.sum(axis=0).reshape(Nu, 1)  # Nu by D

        T5 = T4 - T3  # Nu by D
        T6 = T5 * Qu.reshape(Nu, 1)  # Nu by D

        dv = T6.sum(axis=0)
        dV[u, :] += dv
        dW[clq, :] += T6
        dmu += dv

    if np.isnan(J) or np.isinf(J):
        raise ValueError('objective(): objective is NaN or inf!\n')

    if verbose > 2:
        print('Eval f, g: %.1f seconds used.' % (time.time() - t0))

    dw[:] = np.r_[dmu, dV.ravel(), dW.ravel()]
    return J


class MTR(object):
    """Primal Problem of Multitask Ranking"""
    def __init__(self, X_train, Y_train, C1, C2, C3, cliques):
        assert C1 > 0
        assert C2 > 0
        assert C3 > 0
        assert X_train.shape[0] == Y_train.shape[0]
        assert isspmatrix_csc(Y_train)

        self.X = X_train
        self.Y = Y_train
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.M, self.D = self.X.shape
        self.N = self.Y.shape[1]
        self.cliques = cliques
        self.U = len(self.cliques)
        self.u2pl = self.cliques
        self.pl2u = np.zeros(self.N, dtype=np.int32)
        for u in range(self.U):
            clq = self.cliques[u]
            self.pl2u[clq] = u

        self.data_helper = DataHelper(self.Y, self.cliques)
        self.trained = False

    def _init_vars(self):
        np.random.seed(0)
        w0 = 0.001 * np.random.randn((self.U + self.N + 1) * self.D)
        return w0

    def fit(self, w0=None, verbose=0, fnpy=None):
        N, U, D = self.N, self.U, self.D
        if verbose > 0:
            t0 = time.time()
            print('\nC: %g, %g, %g' % (self.C1, self.C2, self.C3))

        num_vars = (U + N + 1) * D
        if w0 is None:
            if fnpy is None:
                w0 = self._init_vars()
            else:
                try:
                    assert type(fnpy) == str
                    assert fnpy.endswith('.npy')
                    w0 = np.load(fnpy, allow_pickle=False)
                    print('Restore from %s' % fnpy)
                except (IOError, ValueError):
                    w0 = self._init_vars()
        assert w0.shape == (num_vars,)

        try:
            # f: callable(x, g, *args)
            # LBFGS().minimize(f, x0, progress=progress, args=args)
            optim = LBFGS()
            optim.linesearch = 'wolfe'
            optim.max_linesearch = 100
            param_dict = {'N': self.N, 'C': (self.C1, self.C2, self.C3)}
            res = optim.minimize(objective, w0, progress,
                                 args=(self.X, self.cliques, self.data_helper, param_dict, verbose, fnpy))
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
