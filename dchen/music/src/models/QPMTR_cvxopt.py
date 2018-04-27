import sys
import time
import numpy as np
from scipy.sparse import isspmatrix_csc, coo_matrix
import cvxopt


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
        self.Minus = M - Y.sum(axis=0).A.reshape(-1)
        for u in range(U):
            clq = cliques[u]
            Yu = Y[:, clq]
            self.Ys.append(Yu.tocoo())
        self.init = True

    def get_data(self):
        assert self.init is True
        return self.Ys, self.Minus


class QPMTR(object):
    def __init__(self, X_train, Y_train, C1, C2, C3, cliques):
        if not isspmatrix_csc(Y_train):
            raise ValueError('ERROR: %s\n' % 'Y_train should be a parse csc_matrix.')
        if not np.all(np.array([C1, C2, C3]) > 0):
            raise ValueError('ERROR: %s\n' % 'Regularisation parameters should be positive.')
        if X_train.shape[0] != Y_train.shape[0]:
            raise ValueError('X_train.shape[0] != Y_train.shape[0]')

        self.X = X_train
        self.Y = Y_train
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.M, self.D = self.X.shape
        self.N = self.Y.shape[1]
        self.cliques = cliques
        self.U = len(self.cliques)
        self.pl2u = np.zeros(self.N, dtype=np.int32)
        for u in range(self.U):
            clq = self.cliques[u]
            self.pl2u[clq] = u

        self.num_vars = (self.U + self.N + 1) * self.D + self.N * 2
        self.num_cons = self.Y.sum() + self.N * 2
        self.data_helper = DataHelper(self.Y, self.cliques)
        self.trained = False

    def _init_vars(self):
        np.random.seed(0)
        return 0.001 * np.random.randn(self.num_vars)
        N, U, D = self.N, self.U, self.D
        # w0 = np.zeros((U + N + 1) * D + N)
        # w0 = 1e-5 * np.random.rand((U + N + 1) * D + N)
        # w0 = np.r_[1e-3 * np.random.randn((U + N + 1) * D), np.random.rand(N)]
        # w0 = np.r_[1e-3 * np.random.randn((U + N + 1) * D), np.zeros(N)]
        # w0 = np.r_[1e-3 * np.random.rand((U + 1) * D), np.zeros(N * (D + 1))]
        mu = 1e-3 * np.random.randn(D)
        V = 1e-3 * np.random.randn(U, D)
        W = 1e-3 * np.random.randn(N, D)
        xi = np.zeros(N)
        delta = np.zeros(N)
        Ys, Minus = self.data_helper.get_data()
        assert U == len(Ys)
        for u in range(U):
            clq = self.cliques[u]
            Yu = Ys[u]
            Wt = W[clq, :] + (V[u, :] + mu).reshape(1, D)
            T1 = np.dot(self.X, Wt.T)
            T2 = np.full(T1.shape, np.inf, dtype=np.float)
            T2[Yu.row, Yu.col] = T1[Yu.row, Yu.col]  # mask entry (n,i) if y_n^i = 0
            xi[clq] = T2.min(axis=0)
            T1p = np.zeros(T1.shape)
            T1p[Yu.row, Yu.col] = T1[Yu.row, Yu.col]
            T1n = T1 - T1p
            delta[clq] = T1n.sum(axis=0) / Minus[clq]
        return np.r_[mu, V.ravel(), W.ravel(), xi, delta]

    def _create_P(self):
        N, U, D, C1, C2, C3 = self.N, self.U, self.D, self.C1, self.C2, self.C3
        rows, cols, values = [], [], []
        values.append(np.full(D, C3, dtype=np.float))
        values.append(np.full(U * D, C1 / U, dtype=np.float))
        values.append(np.full(N * D, C2 / N, dtype=np.float))
        ix_mu = np.arange(D)
        rows.append(ix_mu)
        cols.append(ix_mu)
        ix_V = np.arange(D, (U + 1) * D)
        rows.append(ix_V)
        cols.append(ix_V)
        ix_W = np.arange((U + 1) * D, (U + 1 + N) * D)
        rows.append(ix_W)
        cols.append(ix_W)
        P = cvxopt.spmatrix(np.concatenate(values, axis=-1),
                            np.concatenate(rows, axis=-1),
                            np.concatenate(cols, axis=-1),
                            size=(self.num_vars, self.num_vars))
        return P

    def _create_q(self):
        N = self.N
        q = cvxopt.matrix(0., size=(self.num_vars, 1))  # should be dense matrix
        q[-N:] = 1. / N
        return q

    def _create_Gh(self):
        N, D, U = self.N, self.D, self.U
        Ys, Minus = self.data_helper.get_data()
        pl2u = self.pl2u
        rows, cols, G = [], [], []
        ix = 0
        print('---- stage 1 --------------')
        sys.stdout.flush()
        for k in range(N):
            # constraint: \delta_k >= 1 - \xi_k + \sum_{n: y_n^k=0} f(u(k), k, n) / M_-^k
            # which is equivalent to
            # \xi_k + \delta_k - (v_{u(k)} + w_k + \mu)^T \sum_{n: y_n^k=0} x_n / M_-^k >= 1
            #
            u = pl2u[k]
            ny = 1 - self.Y[:, k].A.reshape(-1)
            vec = ny.dot(self.X) / Minus[k]
            G += [vec, vec, vec, [-1., -1.]]
            rows.append(np.full(3 * D + 2, ix, dtype=np.int32))
            cols.append(np.arange(D))
            cols.append(np.arange((u + 1) * D, (u + 2) * D))
            cols.append(np.arange((U + 1 + k) * D, (U + 2 + k) * D))
            cols.append([(U + N + 1) * D + k])
            cols.append([(U + N + 1) * D + N + k])
            ix += 1

        print('---- stage 2 --------------')
        sys.stdout.flush()
        # constraint: \delta_k >= 0
        G.append(np.full(N, -1., dtype=np.float))
        rows.append(np.arange(ix, ix + N))
        cols.append(np.arange(self.num_vars - N, self.num_vars))
        ix += N

        # constraint: f(u(k), k, m) - \xi_k >= 0
        # which is equivalent to
        # (v_{u(k)} + w_k + \mu)^T x_m - \xi_k >= 0
        #
        # for k in range(self.N):
        #     u = self.pl2u[k]
        #     for m in range(self.M):
        #         if self.Y[m, k] == 1:
        #             vec = -self.X[m, :]
        #             G += [vec, vec, vec, [1.]]
        #             rows.append(np.full(3 * D + 1, ix, dtype=np.int32))
        #             cols.append(np.arange(D))
        #             cols.append(np.arange((u + 1) * D, (u + 2) * D))
        #             cols.append(np.arange((U + 1 + k) * D, (U + 2 + k) * D))
        #             cols.append([(U + N + 1) * D + k])
        #             ix += 1
        print('---- stage 3 --------------')
        sys.stdout.flush()
        Ycoo = self.Y.tocoo(copy=False)
        print('---- stage 4 --------------')
        sys.stdout.flush()
        m_ix = Ycoo.row
        k_ix = Ycoo.col
        mat = -self.X[m_ix, :]
        npos = mat.shape[0]
        coef = np.hstack([np.tile(mat, (1, 3)), np.ones((npos, 1))]).ravel()
        G.append(coef)
        rowix_p0 = np.arange(ix, ix + npos, dtype=np.int32).reshape(npos, 1)
        rowix = np.tile(rowix_p0, (1, 3 * D + 1)).ravel()
        rows.append(rowix)
        colix_p0 = np.tile(np.arange(D), (npos, 1))
        colix_p1 = np.array([np.arange((pl2u[k] + 1) * D, (pl2u[k] + 2) * D, dtype=np.int32) for k in k_ix])
        colix_p2 = np.array([np.arange((U + 1 + k) * D, (U + 2 + k) * D, dtype=np.int32) for k in k_ix])
        colix_p3 = (U + N + 1) * D + k_ix
        colix = np.hstack([colix_p0, colix_p1, colix_p2, colix_p3.reshape(npos, 1)]).ravel()
        cols.append(colix)
        assert coef.shape == rowix.shape == colix.shape
        ix += npos

        print('---- stage 5 --------------')
        sys.stdout.flush()
        assert ix == self.num_cons
        # coo_matrix((data, (row, col)), shape=())
        Gcoo = coo_matrix((np.concatenate(G, axis=-1),
                           (np.concatenate(rows, axis=-1), np.concatenate(cols, axis=-1))),
                          shape=(self.num_cons, self.num_vars))
        print('---- stage 6 --------------')
        G = cvxopt.spmatrix(Gcoo.data, Gcoo.row.tolist(), Gcoo.col.tolist(), size=Gcoo.shape)
        # np.concatenate(rows, axis=-1),
        # np.concatenate(cols, axis=-1),
        # size=(self.num_cons, self.num_vars))
        h = cvxopt.matrix(0., size=(self.num_cons, 1))  # should be dense matrix
        h[:N] = -1.
        return G, h

    def fit(self, w0=None, verbose=0):
        N, U, D = self.N, self.U, self.D
        if verbose > 0:
            t0 = time.time()
            print('\nC: %g, %g, %g' % (self.C1, self.C2, self.C3))

        if verbose > 0:
            sys.stdout.write('Creating P ... ')
            sys.stdout.flush()

        P = self._create_P()

        if verbose > 0:
            t1 = time.time()
            print('%d by %d sparse matrix created in %.1f seconds.' % (P.size[0], P.size[1], t1 - t0))

        if verbose > 0:
            sys.stdout.write('Creating q ... ')
            sys.stdout.flush()

        q = self._create_q()

        if verbose > 0:
            t2 = time.time()
            print('%d dense vector created in %.1f seconds.' % (q.size[0], t2 - t1))

        if verbose > 0:
            sys.stdout.write('Creating G, h ... ')
            sys.stdout.flush()

        G, h = self._create_Gh()

        if verbose > 0:
            t3 = time.time()
            print('%d by %d sparse matrix, %d dense vector created in %.1f seconds.' %
                  (G.size[0], G.size[1], h.size[0], t3 - t2))

        w0 = self._init_vars()
        assert w0.shape == (self.num_vars,)

        if verbose > 0:
            sys.stdout.write('Solving QP: %d variables, %d constraints ...\n' % (self.num_vars, self.num_cons))
            sys.stdout.flush()

        cvxopt.solvers.options['show_progress'] = False
        cvxopt.solvers.options['feastol'] = 1e-5
        try:
            solution = cvxopt.solvers.qp(P, q, G, h, initvals=self._init_vars())
        except ValueError:
            solution = {'status': 'error'}
        if solution['status'] != "optimal":
            raise ValueError('QP solver failed.')
        w = np.ravel(solution['x'])

        self.mu = w[:D]
        self.V = w[D:(U + 1) * D].reshape(U, D)
        self.W = w[(U + 1) * D:(U + N + 1) * D].reshape(N, D)
        # self.xi = w[(U + N + 1) * D:(U + N + 1) * D + N]
        # self.di = w[self.num_vars - N:self.num_vars]
        self.trained = True

        if verbose > 0:
            print('Training finished in %.1f seconds' % (time.time() - t0))

    def predict(self, X_test):
        """
            Make predictions (score is a real number)
        """
        assert self.trained is True, 'Cannot make prediction before training'
        U, D = self.U, self.D
        assert D == X_test.shape[1]
        preds = []
        for u in range(U):
            clq = self.cliques[u]
            Wt = self.W[clq, :] + (self.V[u, :] + self.mu).reshape(1, D)
            preds.append(np.dot(X_test, Wt.T))
        return np.hstack(preds)
